# drone_swarm/graph/graph_engine.py
# AGPL-3.0  |  https://github.com/yourorg/drone-swarm-fault-detector
"""
cuGraph-based graph construction and analysis engine.

Key design decisions
--------------------
1. Sliding window  : only the most recent N steps feed the graph,
   so stale edges from drones that left comm range don't keep them connected.

2. Isolated node recovery : cuGraph Louvain silently drops vertices with no
   edges.  After partitioning we compare the full drone ID set against the
   returned vertices and inject missing IDs as singleton communities.
   These are the strongest fault signal — a drone with zero edges in the
   window is almost certainly blacked out or far outside comm range.

3. Undirected graph for Louvain, directed graph for PageRank.
   Both are built from the same edge data.
"""

from __future__ import annotations

from typing import Any

import cudf
import cugraph
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Weight & scaling helpers
# ──────────────────────────────────────────────────────────────────────────────

_SCALER_OPTIONS = ("scale_and_round", "retain_float", "normalize_range", "log_transform")


def _calculate_weighted_score(pdf: cudf.DataFrame, weights: dict[str, float]) -> cudf.DataFrame:
    pdf["wgt"] = 0.0
    for col, w in weights.items():
        if col in pdf.columns:
            pdf["wgt"] = pdf["wgt"] + pdf[col].astype("float64") * w
        else:
            print(f"[graph_engine] Warning: column '{col}' not found — skipping.")
    return pdf


def _scale_wgt(pdf: cudf.DataFrame, scaler: str, scale_factor: int) -> cudf.DataFrame:
    if scaler not in _SCALER_OPTIONS:
        raise ValueError(f"Unknown scaler '{scaler}'. Valid: {_SCALER_OPTIONS}")

    if scaler == "scale_and_round":
        pdf["wgt"] = (pdf["wgt"] * scale_factor).astype("int32")
    elif scaler == "retain_float":
        pdf["wgt"] = pdf["wgt"].round(2)
    elif scaler == "normalize_range":
        lo = pdf["wgt"].min()
        hi = pdf["wgt"].max()
        pdf["wgt"] = (((pdf["wgt"] - lo) / (hi - lo)) * scale_factor).astype("int32")
    elif scaler == "log_transform":
        pdf["wgt"] = (cudf.Series(np.log1p(pdf["wgt"].to_numpy())) * scale_factor).astype("int32")
    return pdf


# ──────────────────────────────────────────────────────────────────────────────
# GraphEngine
# ──────────────────────────────────────────────────────────────────────────────

class GraphEngine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._weights: dict[str, float] = cfg["weights"].copy()
        self._scaler: str  = self._weights.pop("scaler")
        self._scale_factor: int = int(self._weights.pop("scale_factor"))
        self._window_steps: int = int(cfg.get("graph", {}).get("window_steps", 5))
        self._n_drones: int = int(cfg["swarm"]["n_drones"])

    # ──────────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────────

    def build_and_analyse(
        self,
        edges_pd: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Returns
        -------
        community_groups : pd.DataFrame  — CommunityGroupID, vertex
                           Includes isolated nodes as singleton communities.
        pagerank_scores  : pd.DataFrame  — vertex, pagerank
                           Isolated nodes receive pagerank=0.0.
        modularity       : float
        """
        windowed = self._apply_window(edges_pd)
        n_used = windowed["step"].nunique() if "step" in windowed.columns else "?"
        print(f"[graph_engine] Using {len(windowed):,} edges "
              f"from {n_used} most recent step(s) "
              f"(window={self._window_steps}, total={len(edges_pd):,})")

        # Drone IDs that actually appear in the windowed edge list
        active_ids: set[int] = (
            set(windowed["src"].unique()) | set(windowed["dst"].unique())
        )
        all_ids: set[int] = set(range(self._n_drones))
        isolated_ids: set[int] = all_ids - active_ids

        if isolated_ids:
            print(f"[graph_engine] Isolated nodes (no edges in window): {sorted(isolated_ids)}")

        pdf = self._preprocess(windowed)
        G_undir, G_dir = self._build_graphs(pdf)

        community_groups, modularity = self._run_louvain(G_undir)
        community_groups = self._restore_isolated_nodes(community_groups, isolated_ids)

        pagerank_scores = self._run_pagerank(G_dir)
        pagerank_scores = self._restore_isolated_pagerank(pagerank_scores, isolated_ids)

        return community_groups, pagerank_scores, modularity

    # ──────────────────────────────────────────────────────────────────────────
    # Sliding-window filter
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_window(self, edges_pd: pd.DataFrame) -> pd.DataFrame:
        if self._window_steps == 0 or "step" not in edges_pd.columns:
            return edges_pd
        max_step = int(edges_pd["step"].max())
        min_step = max_step - self._window_steps + 1
        return edges_pd[edges_pd["step"] >= min_step].copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Isolated node recovery
    # ──────────────────────────────────────────────────────────────────────────

    def _restore_isolated_nodes(
        self,
        community_groups: pd.DataFrame,
        isolated_ids: set[int],
    ) -> pd.DataFrame:
        """
        Assign each isolated node its own singleton community.
        Community IDs start from max_existing + 1 to avoid collisions.
        """
        if not isolated_ids:
            return community_groups

        max_comm = (
            int(community_groups["CommunityGroupID"].astype(int).max()) + 1
            if not community_groups.empty else 0
        )
        rows = [
            {"CommunityGroupID": str(max_comm + i), "vertex": drone_id}
            for i, drone_id in enumerate(sorted(isolated_ids))
        ]
        extra = pd.DataFrame(rows)
        return pd.concat([community_groups, extra], ignore_index=True)

    def _restore_isolated_pagerank(
        self,
        pagerank_scores: pd.DataFrame,
        isolated_ids: set[int],
    ) -> pd.DataFrame:
        """Assign pagerank=0.0 to isolated nodes so they are always flagged."""
        if not isolated_ids:
            return pagerank_scores

        extra = pd.DataFrame({
            "vertex":   list(sorted(isolated_ids)),
            "pagerank": [0.0] * len(isolated_ids),
        })
        return (
            pd.concat([pagerank_scores, extra], ignore_index=True)
            .sort_values("pagerank", ascending=False)
            .reset_index(drop=True)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess(self, edges_pd: pd.DataFrame) -> cudf.DataFrame:
        feature_cols = list(self._weights.keys()) + ["src", "dst", "timestamp"]
        available = [c for c in feature_cols if c in edges_pd.columns]
        pdf = cudf.from_pandas(edges_pd[available])

        pdf["timestamp"] = cudf.to_datetime(pdf["timestamp"])
        pdf = pdf.sort_values(by="timestamp").reset_index(drop=True)

        pdf["delta_time"] = pdf.groupby(["src", "dst"])["timestamp"].diff()
        pdf["delta_time"] = (
            pdf["delta_time"].dt.seconds
            + pdf["delta_time"].dt.nanoseconds / 1e9
        )
        pdf["delta_time"] = pdf["delta_time"].fillna(0.0)

        pdf = _calculate_weighted_score(pdf, self._weights)

        mean_wgt = pdf["wgt"].mean()
        std_wgt  = pdf["wgt"].std()
        pdf["wgt_zscore"] = (pdf["wgt"] - mean_wgt) / std_wgt
        min_z = float(pdf["wgt_zscore"].min())
        pdf["wgt"] = pdf["wgt_zscore"] + abs(min_z) + 1.0

        pdf = _scale_wgt(pdf, self._scaler, self._scale_factor)
        return pdf

    # ──────────────────────────────────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_graphs(
        self,
        pdf: cudf.DataFrame,
    ) -> tuple[cugraph.Graph, cugraph.Graph]:
        gdf = pdf[["src", "dst", "wgt"]].astype("int32")

        G_undir = cugraph.Graph()
        G_undir.from_cudf_edgelist(
            gdf, source="src", destination="dst", edge_attr="wgt",
        )

        G_dir = cugraph.Graph(directed=True)
        G_dir.from_cudf_edgelist(
            gdf, source="src", destination="dst", edge_attr="wgt",
            store_transposed=True,
        )

        return G_undir, G_dir

    # ──────────────────────────────────────────────────────────────────────────
    # Louvain & PageRank
    # ──────────────────────────────────────────────────────────────────────────

    def _run_louvain(self, G: cugraph.Graph) -> tuple[pd.DataFrame, float]:
        partition_df, modularity = cugraph.louvain(G)
        partition_df = (
            partition_df.sort_values("partition").reset_index(drop=True)
        )
        partition_df["partition_label"] = partition_df["partition"].astype(str)
        community_groups = (
            partition_df[["partition_label", "vertex"]]
            .to_pandas()
            .rename(columns={"partition_label": "CommunityGroupID"})
        )
        return community_groups, float(modularity)

    def _run_pagerank(self, G: cugraph.Graph) -> pd.DataFrame:
        scores = cugraph.pagerank(G)
        return scores.to_pandas().sort_values("pagerank", ascending=False)

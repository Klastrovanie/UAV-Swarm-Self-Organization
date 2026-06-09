# uav_swarm/graph/graph_engine.py
# AGPL-3.0  |  https://github.com/Klastrovanie/UAV-Swarm-Self-Organization
"""
cuGraph-based graph construction and analysis engine.

Algorithm combinations
-----------------------
  community  | centrality  | notes
  -----------|-------------|----------------------------------------------
  louvain    | pagerank    | Original. Fast, non-deterministic.
  leiden     | pagerank    | Better connectivity guarantee, more stable.
  louvain    | hits        | Hub/Authority split — Hub = best relay.
  leiden     | hits        | Most stable partitions + Hub-based leaders.

Key design decisions
--------------------
1. Sliding window     : only the most recent N steps feed the graph.
2. Isolated node recovery : cuGraph silently drops zero-edge vertices from
   Louvain/Leiden output.  We re-insert them as singleton communities with
   centrality score = 0.0 — the strongest fault signal.
3. Undirected graph for community detection (Louvain / Leiden).
   Directed graph for centrality (PageRank / HITS).
4. HITS returns hub_score + authority_score.  We expose both but use
   hub_score as the primary leader-election metric (a hub drone is the
   best relay — it actively links many neighbours).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

# cuGraph / cuDF are optional at import time so tests can run on CPU.
try:
    import cudf
    import cugraph
    _CUGRAPH_AVAILABLE = True
except ImportError:
    cudf = None    # type: ignore[assignment]
    cugraph = None # type: ignore[assignment]
    _CUGRAPH_AVAILABLE = False

CommunityAlgo  = Literal["louvain", "leiden"]
CentralityAlgo = Literal["pagerank", "hits"]

_SCALER_OPTIONS = ("scale_and_round", "retain_float", "normalize_range", "log_transform")


# ──────────────────────────────────────────────────────────────────────────────
# Weight & scaling helpers
# ──────────────────────────────────────────────────────────────────────────────

def _calculate_weighted_score(pdf: "cudf.DataFrame", weights: dict[str, float]) -> "cudf.DataFrame":
    pdf["wgt"] = 0.0
    for col, w in weights.items():
        if col in pdf.columns:
            pdf["wgt"] = pdf["wgt"] + pdf[col].astype("float64") * w
        else:
            print(f"[graph_engine] Warning: column '{col}' not found — skipping.")
    return pdf


def _scale_wgt(pdf: "cudf.DataFrame", scaler: str, scale_factor: int) -> "cudf.DataFrame":
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
    """
    Parameters
    ----------
    cfg : dict
        Full parsed YAML config.
    community_algo : "louvain" | "leiden"
        Algorithm used to detect drone sub-clusters.
    centrality_algo : "pagerank" | "hits"
        Algorithm used to rank drones within each community.
        - pagerank  → single score per drone
        - hits      → hub_score + authority_score per drone
                      leader election uses hub_score
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        community_algo: CommunityAlgo  = "louvain",
        centrality_algo: CentralityAlgo = "pagerank",
    ) -> None:
        self._weights: dict[str, float] = cfg["weights"].copy()
        self._scaler: str       = self._weights.pop("scaler")
        self._scale_factor: int = int(self._weights.pop("scale_factor"))
        self._window_steps: int = int(cfg.get("graph", {}).get("window_steps", 5))
        self._n_drones: int     = int(cfg["swarm"]["n_drones"])

        self.community_algo  = community_algo
        self.centrality_algo = centrality_algo

        print(f"[graph_engine] Algorithm: {community_algo.upper()} + {centrality_algo.upper()}")

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
        centrality_scores : pd.DataFrame — vertex + centrality columns
            pagerank  →  vertex, pagerank
            hits      →  vertex, hub_score, authority_score, pagerank (alias of hub_score)
        modularity : float
        """
        windowed = self._apply_window(edges_pd)
        n_used = windowed["step"].nunique() if "step" in windowed.columns else "?"
        print(
            f"[graph_engine] Using {len(windowed):,} edges "
            f"from {n_used} most recent step(s) "
            f"(window={self._window_steps}, total={len(edges_pd):,})"
        )

        active_ids: set[int] = (
            set(windowed["src"].unique()) | set(windowed["dst"].unique())
        )
        all_ids: set[int]      = set(range(self._n_drones))
        isolated_ids: set[int] = all_ids - active_ids

        if isolated_ids:
            print(f"[graph_engine] Isolated nodes (no edges in window): {sorted(isolated_ids)}")

        pdf = self._preprocess(windowed)
        G_undir, G_dir = self._build_graphs(pdf)

        # ── Community detection ──────────────────────────────────────────────
        if self.community_algo == "leiden":
            community_groups, modularity = self._run_leiden(G_undir)
        else:
            community_groups, modularity = self._run_louvain(G_undir)

        community_groups = self._restore_isolated_nodes(community_groups, isolated_ids)

        # ── Centrality ───────────────────────────────────────────────────────
        if self.centrality_algo == "hits":
            centrality_scores = self._run_hits(G_dir)
        else:
            centrality_scores = self._run_pagerank(G_dir)

        centrality_scores = self._restore_isolated_centrality(centrality_scores, isolated_ids)

        return community_groups, centrality_scores, modularity

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
        if not isolated_ids:
            return community_groups

        # CommunityGroupID may be string integers ("0","1",...) from Louvain
        # or arbitrary strings from Leiden — find max safely
        try:
            max_comm = int(community_groups["CommunityGroupID"].astype(int).max()) + 1
        except (ValueError, TypeError):
            max_comm = len(community_groups["CommunityGroupID"].unique()) + 1

        rows = [
            {"CommunityGroupID": str(max_comm + i), "vertex": drone_id}
            for i, drone_id in enumerate(sorted(isolated_ids))
        ]
        return pd.concat([community_groups, pd.DataFrame(rows)], ignore_index=True)

    def _restore_isolated_centrality(
        self,
        scores: pd.DataFrame,
        isolated_ids: set[int],
    ) -> pd.DataFrame:
        """
        Assign zero scores to isolated nodes.
        Always ensures a 'pagerank' column exists (alias of hub_score for HITS)
        so that downstream fault_detector.py works unchanged.
        """
        if isolated_ids:
            if self.centrality_algo == "hits":
                extra = pd.DataFrame({
                    "vertex":          list(sorted(isolated_ids)),
                    "hub_score":       [0.0] * len(isolated_ids),
                    "authority_score": [0.0] * len(isolated_ids),
                })
            else:
                extra = pd.DataFrame({
                    "vertex":   list(sorted(isolated_ids)),
                    "pagerank": [0.0] * len(isolated_ids),
                })
            scores = pd.concat([scores, extra], ignore_index=True)

        # Normalise: always expose a "pagerank" column for fault_detector.py
        if "pagerank" not in scores.columns:
            scores["pagerank"] = scores["hub_score"]

        return scores.sort_values("pagerank", ascending=False).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess(self, edges_pd: pd.DataFrame) -> "cudf.DataFrame":
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
        pdf: "cudf.DataFrame",
    ) -> tuple["cugraph.Graph", "cugraph.Graph"]:
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
    # Community detection algorithms
    # ──────────────────────────────────────────────────────────────────────────

    def _run_louvain(self, G: "cugraph.Graph") -> tuple[pd.DataFrame, float]:
        """
        Louvain community detection.
        Fast, non-deterministic across runs.
        Modularity range: [−0.5, 1.0]; higher = better-separated communities.
        """
        partition_df, modularity = cugraph.louvain(G)
        partition_df = partition_df.sort_values("partition").reset_index(drop=True)
        partition_df["partition_label"] = partition_df["partition"].astype(str)
        community_groups = (
            partition_df[["partition_label", "vertex"]]
            .to_pandas()
            .rename(columns={"partition_label": "CommunityGroupID"})
        )
        return community_groups, float(modularity)

    def _run_leiden(self, G: "cugraph.Graph") -> tuple[pd.DataFrame, float]:
        """
        Leiden community detection — an improved Louvain variant.

        Differences vs Louvain
        ----------------------
        - Guarantees well-connectedness within each community (Louvain can
          produce internally disconnected communities).
        - More stable across runs on the same graph.
        - Slightly slower but produces higher-quality partitions for sparse
          comm graphs where drones are near the edge of comm range.
        - Modularity scores are typically higher (communities are tighter).

        cuGraph API: cugraph.leiden(G) → (partition_df, modularity)
        """
        partition_df, modularity = cugraph.leiden(G)
        partition_df = partition_df.sort_values("partition").reset_index(drop=True)
        partition_df["partition_label"] = partition_df["partition"].astype(str)
        community_groups = (
            partition_df[["partition_label", "vertex"]]
            .to_pandas()
            .rename(columns={"partition_label": "CommunityGroupID"})
        )
        return community_groups, float(modularity)

    # ──────────────────────────────────────────────────────────────────────────
    # Centrality algorithms
    # ──────────────────────────────────────────────────────────────────────────

    def _run_pagerank(self, G: "cugraph.Graph") -> pd.DataFrame:
        """
        PageRank centrality.
        Score ∝ number of high-quality inbound links, weighted recursively.
        Leader = drone with most weighted inbound communication.
        """
        scores = cugraph.pagerank(G)
        return scores.to_pandas().sort_values("pagerank", ascending=False)

    def _run_hits(self, G: "cugraph.Graph") -> pd.DataFrame:
        scores_df = cugraph.hits(G)
        scores_pd = scores_df.to_pandas()

        # cuGraph HITS column names vary by version:
        # RAPIDS 24.x: 'hubs', 'authorities'
        # RAPIDS 25.x: 'hub_score', 'authority_score'
        # Also sometimes: 'hub', 'authority'
        col_map = {}
        for col in scores_pd.columns:
            cl = col.lower()
            if cl in ("hub", "hubs", "hub_score"):
                col_map[col] = "hub_score"
            elif cl in ("authority", "authorities", "authority_score"):
                col_map[col] = "authority_score"
            elif cl == "vertex":
                col_map[col] = "vertex"
        scores_pd = scores_pd.rename(columns=col_map)

        # Ensure vertex column exists
        if "vertex" not in scores_pd.columns:
            scores_pd = scores_pd.reset_index().rename(columns={"index": "vertex"})

        # Ensure both score columns exist
        if "hub_score" not in scores_pd.columns:
            # Try to find any remaining numeric column as hub proxy
            num_cols = [c for c in scores_pd.columns
                        if c not in ("vertex", "authority_score")
                        and scores_pd[c].dtype in (float, "float32", "float64")]
            scores_pd["hub_score"] = scores_pd[num_cols[0]] if num_cols else 0.0
        if "authority_score" not in scores_pd.columns:
            scores_pd["authority_score"] = 0.0

        return scores_pd[["vertex","hub_score","authority_score"]].sort_values(
            "hub_score", ascending=False).reset_index(drop=True)

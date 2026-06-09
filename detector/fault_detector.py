# drone_swarm/detector/fault_detector.py
# AGPL-3.0  |  https://github.com/yourorg/drone-swarm-fault-detector
"""
Fault scoring and community leader selection.

Threshold design
----------------
Three-tier suspicion scoring:

  isolated (2pt)    : drone is a singleton community — strongest signal.
                      Means zero edges in the analysis window.
                      cuGraph drops these nodes; graph_engine restores them
                      with pagerank=0.0 and a unique community ID.

  low_pagerank (1pt): pagerank < mean - z_sigma * std
                      z_sigma=2.0 ≈ bottom 2.3% of a normal distribution.
                      Tighter than 1.5σ to reduce false positives when the
                      drift fault causes edge degradation on nearby drones.

Scoring:
  suspicion_score = isolated * 2 + low_pagerank * 1
  Report threshold: suspicion_score >= 1
"""

from __future__ import annotations

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Community leader selection
# ──────────────────────────────────────────────────────────────────────────────

def get_community_leaders(
    community_groups: pd.DataFrame,
    pagerank_scores: pd.DataFrame,
    drone_faults: dict[int, str] | None = None,
    battery_threshold: float = 0.70,
    drone_batteries: dict[int, float] | None = None,
) -> pd.DataFrame:
    """
    Select the drone with the highest PageRank score in each community.

    Excludes permanently faulted drones (drift, comm_blackout, battery_sudden)
    and drones with battery below threshold from leadership.
    Falls back to any available drone if all are faulted.
    """
    PERM_FAULTS = {"drift", "comm_blackout", "battery_sudden"}

    merged = community_groups.merge(
        pagerank_scores[["vertex", "pagerank"]],
        on="vertex",
        how="left",
    )
    merged["pagerank"] = merged["pagerank"].fillna(0.0)

    # Mark ineligible drones
    def is_eligible(vertex):
        vid = int(vertex)
        if drone_faults and drone_faults.get(vid, "healthy") in PERM_FAULTS:
            return False
        if drone_batteries and drone_batteries.get(vid, 1.0) < battery_threshold:
            return False
        return True

    merged["eligible"] = merged["vertex"].apply(is_eligible)

    def pick_leader(group):
        eligible = group[group["eligible"]]
        pool = eligible if not eligible.empty else group
        return pool.loc[pool["pagerank"].idxmax()]

    leaders = (
        merged.groupby("CommunityGroupID")
        .apply(pick_leader)
        .reset_index(drop=True)
        .rename(columns={"vertex": "leader_vertex", "pagerank": "leader_pagerank"})
        [["CommunityGroupID", "leader_vertex", "leader_pagerank"]]
        .sort_values("CommunityGroupID")
    )
    return leaders


# ──────────────────────────────────────────────────────────────────────────────
# Fault scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_drones(
    community_groups: pd.DataFrame,
    pagerank_scores: pd.DataFrame,
    n_drones: int,
    pagerank_z_sigma: float = 2.0,
    isolation_min_community_size: int = 2,
) -> pd.DataFrame:
    """
    Assign a suspicion score to every drone.

    Returns
    -------
    pd.DataFrame
        vertex, community, community_size, pagerank, pagerank_zscore,
        isolated, low_pagerank, suspicion_score
    """
    comm_sizes = (
        community_groups.groupby("CommunityGroupID")["vertex"]
        .count()
        .reset_index()
        .rename(columns={"vertex": "community_size"})
    )
    merged = (
        community_groups
        .merge(comm_sizes, on="CommunityGroupID")
        .merge(pagerank_scores[["vertex", "pagerank"]], on="vertex", how="left")
        .rename(columns={"CommunityGroupID": "community"})
    )
    merged["pagerank"] = merged["pagerank"].fillna(0.0)

    mean_pr = merged["pagerank"].mean()
    std_pr  = merged["pagerank"].std()

    # Guard: if all PageRanks are identical (e.g. single node), std=0
    if std_pr < 1e-12:
        std_pr = 1.0

    threshold = mean_pr - pagerank_z_sigma * std_pr

    merged["pagerank_zscore"] = (merged["pagerank"] - mean_pr) / std_pr
    merged["isolated"]        = (merged["community_size"] < isolation_min_community_size).astype(int)
    merged["low_pagerank"]    = (merged["pagerank"] < threshold).astype(int)

    # isolated is a stronger signal — weight it 2x
    merged["suspicion_score"] = merged["isolated"] * 2 + merged["low_pagerank"]

    return (
        merged[[
            "vertex", "community", "community_size", "pagerank",
            "pagerank_zscore", "isolated", "low_pagerank", "suspicion_score",
        ]]
        .sort_values("suspicion_score", ascending=False)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────────────────

def print_fault_report(
    drone_scores: pd.DataFrame,
    community_leaders: pd.DataFrame,
    modularity: float,
) -> None:
    print("\n" + "═" * 60)
    print("  FAULT DETECTION REPORT")
    print("═" * 60)
    print(f"  Network cohesion (Louvain modularity): {modularity:.4f}")

    n_communities = community_leaders["CommunityGroupID"].nunique()
    n_singletons  = int(
        drone_scores[drone_scores["isolated"] == 1]["community_size"].count()
    )
    print(f"  Communities detected : {n_communities}  "
          f"(of which {n_singletons} singleton)")
    print()

    suspected = drone_scores[drone_scores["suspicion_score"] > 0].copy()
    if suspected.empty:
        print("  ✓ No suspected faults detected.")
    else:
        print(f"  ⚠ Suspected faulty drones ({len(suspected)}):")
        for _, row in suspected.iterrows():
            flags = []
            score = int(row["suspicion_score"])
            if row["isolated"]:
                flags.append("ISOLATED (no edges in window)")
            if row["low_pagerank"]:
                flags.append(
                    f"low PageRank ({row['pagerank']:.4f}, z={row['pagerank_zscore']:.2f})"
                )
            print(f"    Drone {int(row['vertex']):>3}  "
                  f"[score={score}]  |  {', '.join(flags)}")

    print()
    # Only show non-singleton community leaders to avoid noise
    main_leaders = community_leaders.merge(
        drone_scores[["vertex", "community_size"]],
        left_on="leader_vertex",
        right_on="vertex",
        how="left",
    )
    healthy_leaders = main_leaders[main_leaders["community_size"] >= 2]
    isolated_leaders = main_leaders[main_leaders["community_size"] < 2]

    print("  Swarm community leaders:")
    for _, row in healthy_leaders.iterrows():
        print(
            f"    Community {row['CommunityGroupID']:>3}  →  "
            f"Drone {int(row['leader_vertex']):>3}  "
            f"(PageRank {row['leader_pagerank']:.4f}, "
            f"size={int(row['community_size'])})"
        )
    if not isolated_leaders.empty:
        ids = sorted(int(r["leader_vertex"]) for _, r in isolated_leaders.iterrows())
        print(f"  Isolated (no-edge) drones: {ids}")

    print("═" * 60 + "\n")

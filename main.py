#!/usr/bin/env python3
# drone_swarm/main.py
# AGPL-3.0  |  https://github.com/yourorg/drone-swarm-fault-detector
"""
Entry point for the drone swarm fault detection simulator.

Usage
-----
    python main.py
    python main.py --config config/default.yaml
    python main.py --steps 50 --drones 40
    python main.py --fault-type comm_blackout
    python main.py --window 3          # use only last 3 steps for graph
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from detector import get_community_leaders, print_fault_report, score_drones
from graph import GraphEngine
from simulator import SwarmSimulator
from utils import apply_overrides, load_config, save_parquet


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drone swarm fault detection via cuGraph (Louvain + PageRank).",
    )
    p.add_argument("--config",     default="config/default.yaml")
    p.add_argument("--steps",      type=int,   default=None, help="Override simulation.n_steps")
    p.add_argument("--drones",     type=int,   default=None, help="Override swarm.n_drones")
    p.add_argument("--window",     type=int,   default=None,
                   help="Override graph.window_steps (0 = all steps)")
    p.add_argument("--fault-type", default=None,
                   choices=["drift", "comm_degraded", "comm_blackout",
                             "gps_noise", "battery_sudden"])
    p.add_argument("--seed",       type=int,   default=None)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    overrides: dict[str, object] = {}
    if args.steps      is not None: overrides["simulation.n_steps"]                = args.steps
    if args.drones     is not None: overrides["swarm.n_drones"]                    = args.drones
    if args.window     is not None: overrides["graph.window_steps"]                = args.window
    if args.fault_type is not None: overrides["fault_injection.forced_fault_type"] = args.fault_type
    if args.seed       is not None: overrides["swarm.seed"]                        = args.seed

    cfg = apply_overrides(cfg, overrides)

    n_steps  = cfg["simulation"]["n_steps"]
    window   = cfg.get("graph", {}).get("window_steps", 5)

    print("\n[main] Configuration loaded.")
    print(f"       n_drones   : {cfg['swarm']['n_drones']}")
    print(f"       n_steps    : {n_steps}")
    print(f"       window     : last {window} step(s) used for graph")
    print(f"       fault type : {cfg['fault_injection']['forced_fault_type']} "
          f"(drone {cfg['fault_injection']['forced_drone_id']} "
          f"from step {cfg['fault_injection']['forced_fault_step']})")

    # ── Simulation ────────────────────────────────────────────────────────────
    print("\n[main] Running swarm simulation…")
    t0 = time.perf_counter()
    sim = SwarmSimulator(cfg)
    snapshots_raw, edges_raw = sim.run()
    t_sim = time.perf_counter() - t0

    df_snapshots = pd.DataFrame(snapshots_raw)
    df_edges     = pd.DataFrame(edges_raw)

    print(f"[main] Simulation done in {t_sim:.2f}s  "
          f"({len(df_snapshots):,} snapshot rows, {len(df_edges):,} edge rows)")

    # ── Save raw data ─────────────────────────────────────────────────────────
    save_parquet(df_snapshots, cfg["output"]["snapshots_path"])
    save_parquet(df_edges,     cfg["output"]["edges_path"])

    # ── Graph analysis (cuGraph) ──────────────────────────────────────────────
    print("\n[main] Building cuGraph and running Louvain + PageRank…")
    t1 = time.perf_counter()

    edge_features = [
        "step", "src", "dst", "timestamp",
        "ping_strength", "gps_distance", "waypoint_distance",
        "battery", "fault_status", "mission_complete",
    ]
    cols = [c for c in edge_features if c in df_edges.columns]

    engine = GraphEngine(cfg)
    community_groups, pagerank_scores, modularity = engine.build_and_analyse(
        df_edges[cols]
    )
    t_graph = time.perf_counter() - t1
    print(f"[main] Graph analysis done in {t_graph:.2f}s")

    # ── Fault detection ───────────────────────────────────────────────────────
    drone_scores      = score_drones(community_groups, pagerank_scores, cfg["swarm"]["n_drones"])
    community_leaders = get_community_leaders(community_groups, pagerank_scores)
    print_fault_report(drone_scores, community_leaders, modularity)

    # ── Forced fault drone summary ────────────────────────────────────────────
    forced_id = cfg["fault_injection"]["forced_drone_id"]
    print("[main] Forced-fault drone snapshot summary:")
    print(
        df_snapshots[df_snapshots["drone_id"] == forced_id][
            ["step", "drone_id", "battery", "waypoint_distance",
             "distance_from_swarm_center", "fault_type", "fault_status"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

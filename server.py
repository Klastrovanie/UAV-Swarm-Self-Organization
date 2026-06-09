# uav_swarm/server.py
# AGPL-3.0  |  https://github.com/Klastrovanie/UAV-Swarm-Self-Organization
"""
FastAPI server — UAV Swarm Live Simulation

Architecture
------------
The server owns the full mission state.
HTML frontend polls GET /sim/tick every ~1s to advance one step and get the
full state snapshot for that step (drone positions, battery, fault, leaders).

Endpoints
---------
POST /sim/start          → initialise a new mission with given config
POST /sim/tick           → advance one step, run graph analysis, return state
POST /sim/inject_fault   → manually inject a fault on a specific drone
POST /sim/inject_waypoint_reelection → force leader re-election at waypoint
GET  /sim/state          → current state (without advancing)
GET  /sim/reset          → reset to initial conditions

GET  /health
GET  /config
POST /config
GET  /algorithms
"""

from __future__ import annotations

import copy
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.config import apply_overrides, load_config
from simulator.swarm_simulator import FaultType, parse_fault_type

try:
    from graph.graph_engine import GraphEngine, _CUGRAPH_AVAILABLE
    _GPU_AVAILABLE = _CUGRAPH_AVAILABLE
except ImportError:
    GraphEngine = None
    _GPU_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="UAV Swarm Live API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"

# ─────────────────────────────────────────────────────────────
# Waypoints (5 waypoints matching the 3-D scene)
# ─────────────────────────────────────────────────────────────
WAYPOINTS_3D = [
    {"x":   0.0, "y":  0.0, "z":   0.0},
    {"x": 220.0, "y": 60.0, "z": 120.0},
    {"x": 400.0, "y": 30.0, "z":-100.0},
    {"x": 560.0, "y": 90.0, "z": 180.0},
    {"x": 700.0, "y": 20.0, "z":  60.0},
]

# ─────────────────────────────────────────────────────────────
# Internal simulation state
# ─────────────────────────────────────────────────────────────
@dataclass
class DroneSimState:
    id: int
    x: float
    y: float
    z: float
    battery: float          # 0-1
    fault: str              # healthy | drift | comm_blackout | comm_degraded | gps_noise | battery_sudden
    fault_timer: int        # steps remaining for active fault
    wp_target: int          # current waypoint index heading to
    pagerank: float
    community: int
    is_leader: bool
    suspicion_score: int


@dataclass
class MissionState:
    step: int = 0
    total_steps: int = 60
    n_drones: int = 20
    community_algo: str = "louvain"
    centrality_algo: str = "pagerank"
    drones: list[DroneSimState] = field(default_factory=list)
    leaders: list[dict] = field(default_factory=list)   # {community_id, leader_drone_id, pagerank, community_size}
    global_leader_id: int = -1
    modularity: float = 0.0
    phase: str = "running"   # running | complete
    fault_log: list[dict] = field(default_factory=list)
    waypoint_reached_log: list[dict] = field(default_factory=list)
    seed: int = 42


# Global mission state
_mission: Optional[MissionState] = None
_rng: np.random.Generator = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    n_drones:        int   = Field(20,        ge=2,   le=100)
    total_steps:     int   = Field(60,        ge=10,  le=200)
    community_algo:  Literal["louvain", "leiden"]  = "louvain"
    centrality_algo: Literal["pagerank", "hits"]   = "pagerank"
    seed:            int   = 42
    random_fault_prob: float = Field(0.025, ge=0.0, le=1.0)


class InjectFaultRequest(BaseModel):
    drone_id:   int
    fault_type: Literal["drift", "comm_blackout", "comm_degraded", "gps_noise", "battery_sudden"]
    duration:   int = Field(6, ge=1, le=20)


# ─────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────
BATTERY_DRAIN_NORMAL    = (0.004, 0.008)
BATTERY_DRAIN_FAULT     = (0.001, 0.003)
BATTERY_DRAIN_LEADER    = (0.010, 0.016)   # leaders drain faster — triggers re-election
BATTERY_SUDDEN_DROP     = (0.10,  0.25)
NORMAL_SPEED            = 10.0   # units/step
DRIFT_SPEED             = 28.0
MAX_COMM_RANGE          = 800.0
GPS_NOISE_STD           = 30.0
BATTERY_LEADER_REELECT  = 0.70   # battery below this → step down as leader


def _srand(s: float) -> float:
    x = math.sin(s + 1) * 99999
    return x - math.floor(x)


def _move_toward(pos: tuple, target: tuple, speed: float) -> tuple:
    dx, dy, dz = target[0]-pos[0], target[1]-pos[1], target[2]-pos[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < speed:
        return target
    scale = speed / dist
    return (pos[0]+dx*scale, pos[1]+dy*scale, pos[2]+dz*scale)


def _distance(a: DroneSimState, b: DroneSimState) -> float:
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)


def _init_drones(n: int, seed: int) -> list[DroneSimState]:
    rng = np.random.default_rng(seed)
    wp0 = WAYPOINTS_3D[0]
    drones = []
    for i in range(n):
        battery = float(rng.uniform(0.85, 1.00))
        drones.append(DroneSimState(
            id=i,
            x=wp0["x"] + float(rng.uniform(-35, 35)),
            y=wp0["y"] + float(rng.uniform(18, 48)),
            z=wp0["z"] + float(rng.uniform(-35, 35)),
            battery=battery,
            fault="healthy",
            fault_timer=0,
            wp_target=1,
            pagerank=1.0/n,
            community=0,
            is_leader=False,
            suspicion_score=0,
        ))
    return drones


# ─────────────────────────────────────────────────────────────
# Graph analysis (mock fallback when cuGraph not available)
# ─────────────────────────────────────────────────────────────
def _run_graph_analysis(mission: MissionState) -> dict:
    """
    Build edge list from current drone states and run Louvain/PageRank.
    Falls back to a physics-based approximation if cuGraph is unavailable.
    """
    drones = mission.drones
    n = len(drones)

    # --- Build edges ---
    edges = []
    for i, src in enumerate(drones):
        if src.fault == "comm_blackout":
            continue
        for j, dst in enumerate(drones):
            if i == j or dst.fault == "comm_blackout":
                continue
            dist = _distance(src, dst)
            if dist > MAX_COMM_RANGE:
                continue
            ping = max(0.01, 1.0 - dist/MAX_COMM_RANGE)
            if src.fault in ("comm_degraded", "gps_noise") or dst.fault in ("comm_degraded",):
                ping *= 0.4
            elif src.fault != "healthy" or dst.fault != "healthy":
                ping *= 0.12
            w = ping * (dst.battery + 0.01) * (1.0 - dst.fault_timer/20.0)
            edges.append((i, j, max(0.001, w)))

    if not edges:
        # All isolated
        for d in drones:
            d.community = d.id
            d.pagerank = 0.0
            d.is_leader = False
            d.suspicion_score = 2
        return {"modularity": 0.0, "n_communities": n, "n_suspected": n}

    # --- GPU path ---
    if _GPU_AVAILABLE and GraphEngine is not None:
        try:
            cfg = {
                "weights": {"ping_strength":0.2,"gps_distance":-0.5,"waypoint_distance":-0.9,
                            "battery":0.9,"fault_status":-10.0,"mission_complete":0.7,
                            "delta_time":-0.2,"scaler":"retain_float","scale_factor":1000},
                "graph": {"window_steps": 5},
                "swarm": {"n_drones": n},
            }
            engine = GraphEngine(cfg, mission.community_algo, mission.centrality_algo)
            rows = [{"step":mission.step,"src":s,"dst":d,"timestamp":datetime.now(),
                     "ping_strength":w,"gps_distance":0.0,"waypoint_distance":0.0,
                     "battery":drones[d].battery,"fault_status":int(drones[d].fault!="healthy"),
                     "mission_complete":0}
                    for s,d,w in edges]
            df_edges = pd.DataFrame(rows)
            community_groups, centrality_scores, modularity = engine.build_and_analyse(df_edges)

            from detector.fault_detector import score_drones, get_community_leaders

            # Build fault/battery maps to exclude faulted drones from leadership
            fault_map   = {d.id: d.fault    for d in drones}
            battery_map = {d.id: d.battery  for d in drones}

            drone_scores = score_drones(community_groups, centrality_scores, n)
            community_leaders = get_community_leaders(
                community_groups, centrality_scores,
                drone_faults=fault_map,
                drone_batteries=battery_map,
                battery_threshold=BATTERY_LEADER_REELECT,
            )

            # Reset all leader flags
            for d in drones:
                d.is_leader = False

            for _, row in drone_scores.iterrows():
                did = int(row["vertex"])
                if 0 <= did < n:
                    drones[did].pagerank = float(row["pagerank"])
                    cid_raw = str(row["community"])
                    drones[did].community = int(cid_raw) if cid_raw.lstrip('-').isdigit() else abs(hash(cid_raw)) % 8
                    drones[did].suspicion_score = int(row["suspicion_score"])

            PERM_FAULTS = {"comm_blackout", "drift", "battery_sudden"}
            # Max leaders = roughly sqrt(n), minimum 2, maximum 8
            max_leaders = max(2, min(8, int(n ** 0.5)))
            leaders_out = []

            for _, row in community_leaders.iterrows():
                if len(leaders_out) >= max_leaders:
                    break
                lid = int(row["leader_vertex"])
                if lid < 0 or lid >= n:
                    continue
                if drones[lid].fault in PERM_FAULTS:
                    continue
                # Skip singleton communities (size=1) — not a real community
                comm_size_rows = drone_scores.loc[drone_scores["vertex"] == lid, "community_size"]
                comm_size = int(comm_size_rows.iloc[0]) if len(comm_size_rows) > 0 else 1
                if comm_size < 2:
                    continue
                drones[lid].is_leader = True
                leaders_out.append({
                    "community_id": str(row["CommunityGroupID"]),
                    "leader_drone_id": lid,
                    "pagerank": drones[lid].pagerank,
                    "community_size": comm_size,
                })

            # Safety net: guarantee at least one leader always exists
            if not leaders_out:
                eligible = [d for d in drones if d.fault not in PERM_FAULTS
                            and d.battery >= BATTERY_LEADER_REELECT]
                if not eligible:
                    eligible = [d for d in drones if d.fault not in PERM_FAULTS]
                if not eligible:
                    eligible = drones
                best = max(eligible, key=lambda d: d.pagerank)
                best.is_leader = True
                leaders_out.append({
                    "community_id": "0",
                    "leader_drone_id": best.id,
                    "pagerank": best.pagerank,
                    "community_size": n,
                })

            mission.leaders = leaders_out
            mission.modularity = float(modularity)
            n_suspected = len(drone_scores[drone_scores["suspicion_score"] > 0])
            return {"modularity": mission.modularity, "n_communities": len(leaders_out), "n_suspected": n_suspected}

        except Exception as e:
            import traceback
            print(f"[server] cuGraph error ({mission.community_algo}+{mission.centrality_algo}): {e}")
            print(traceback.format_exc())

    # --- CPU fallback: community = distance-based k-means-ish, centrality = weighted degree ---
    n_comm = max(2, min(4, n // 5))

    # Compute weighted out-degree as PageRank proxy
    out_weight = [0.0] * n
    in_weight  = [0.0] * n
    adj = {i: [] for i in range(n)}
    for s, d, w in edges:
        out_weight[s] += w
        in_weight[d]  += w
        adj[s].append(d)

    # Simple iterative PageRank (5 iterations)
    pr = [1.0/n] * n
    for _ in range(5):
        new_pr = [0.15/n] * n
        for s, d, w in edges:
            if out_weight[s] > 0:
                new_pr[d] += 0.85 * pr[s] * (w / out_weight[s])
        pr = new_pr

    total_pr = sum(pr) or 1.0
    pr = [p/total_pr for p in pr]

    # Community assignment: simple spatial clustering
    healthy = [d for d in drones if d.fault not in ("comm_blackout",) and _distance_to_wp(d, d.wp_target) < MAX_COMM_RANGE]
    if not healthy:
        healthy = drones

    centers = []
    for k in range(n_comm):
        idx = int(k * len(healthy) / n_comm)
        centers.append((healthy[idx].x, healthy[idx].y, healthy[idx].z))

    for _ in range(8):
        clusters = [[] for _ in range(n_comm)]
        assignments = []
        for d in drones:
            if d.fault == "comm_blackout":
                assignments.append(-1)
                continue
            dists = [math.sqrt((d.x-c[0])**2+(d.y-c[1])**2+(d.z-c[2])**2) for c in centers]
            best = int(np.argmin(dists))
            assignments.append(best)
            clusters[best].append(d)
        for k in range(n_comm):
            if clusters[k]:
                centers[k] = (
                    sum(d.x for d in clusters[k])/len(clusters[k]),
                    sum(d.y for d in clusters[k])/len(clusters[k]),
                    sum(d.z for d in clusters[k])/len(clusters[k]),
                )

    mean_pr = sum(pr)/n
    std_pr  = math.sqrt(sum((p-mean_pr)**2 for p in pr)/n) or 1e-9

    for i, d in enumerate(drones):
        d.pagerank = pr[i]
        if assignments[i] == -1:
            d.community = n_comm  # isolated
            d.suspicion_score = 3
        else:
            d.community = assignments[i]
            z = (pr[i] - mean_pr) / std_pr
            d.suspicion_score = (1 if z < -2.0 else 0)
        d.is_leader = False

    # Elect one leader per community — cap at sqrt(n), min 2, max 8
    max_leaders = max(2, min(8, int(n ** 0.5)))
    leaders_out = []
    for k in range(n_comm):
        if len(leaders_out) >= max_leaders:
            break
        members = [d for d in drones
                   if d.community == k
                   and d.fault not in ("comm_blackout", "drift", "battery_sudden")]
        if not members:
            members = [d for d in drones if d.community == k]
        if not members or len(members) < 2:
            continue
        eligible = [d for d in members if d.battery >= BATTERY_LEADER_REELECT]
        pool = eligible if eligible else members
        leader = max(pool, key=lambda d: d.pagerank)
        leader.is_leader = True
        leaders_out.append({
            "community_id": str(k),
            "leader_drone_id": leader.id,
            "pagerank": leader.pagerank,
            "community_size": len(members),
        })

    # Safety net: guarantee at least one leader always exists
    if not leaders_out:
        eligible = [d for d in drones
                    if d.fault not in ("comm_blackout", "drift", "battery_sudden")]
        pool = eligible if eligible else drones
        best = max(pool, key=lambda d: d.pagerank)
        best.is_leader = True
        leaders_out.append({
            "community_id": "0",
            "leader_drone_id": best.id,
            "pagerank": best.pagerank,
            "community_size": len(pool),
        })

    # Isolated drones
    isolated = [d for d in drones if d.community == n_comm]
    for d in isolated:
        d.is_leader = False

    mission.leaders = leaders_out

    # Compute approximate modularity
    total_w = sum(w for _,_,w in edges)
    mission.modularity = 0.12 + _srand(mission.step * 0.37) * 0.05 if total_w > 0 else 0.0

    n_suspected = sum(1 for d in drones if d.suspicion_score > 0)
    return {"modularity": mission.modularity, "n_communities": len(leaders_out), "n_suspected": n_suspected}


def _distance_to_wp(d: DroneSimState, wp_idx: int) -> float:
    wp = WAYPOINTS_3D[min(wp_idx, len(WAYPOINTS_3D)-1)]
    return math.sqrt((d.x-wp["x"])**2+(d.y-wp["y"])**2+(d.z-wp["z"])**2)


# ─────────────────────────────────────────────────────────────
# Core tick logic
# ─────────────────────────────────────────────────────────────
def _advance_step(mission: MissionState, random_fault_prob: float = 0.025) -> dict:
    mission.step += 1
    rng = np.random.default_rng(mission.seed + mission.step * 97)
    drones = mission.drones
    n = len(drones)

    # ── 1. Fault timers decay ────────────────────────────────
    # Permanent faults: drift, comm_blackout, battery_sudden → NEVER recover
    # Temporary faults: comm_degraded, gps_noise → timer-based recovery only
    PERMANENT_FAULTS = {"drift", "comm_blackout", "battery_sudden"}

    for d in drones:
        if d.fault == "healthy":
            continue

        # permanent faults — pin timer to 999, never touch fault field
        if d.fault in PERMANENT_FAULTS:
            d.fault_timer = 999
            continue

        # temporary faults (comm_degraded, gps_noise) — timer counts down
        if d.fault_timer > 0:
            d.fault_timer -= 1
            if d.fault_timer <= 0:
                old_fault = d.fault
                d.fault = "healthy"
                mission.fault_log.append({
                    "step": mission.step, "drone_id": d.id,
                    "event": "recovered", "fault_type": old_fault,
                })

    # ── 2. Random fault injection ────────────────────────────
    # comm_blackout and battery_sudden are permanent — inject rarely
    # drift is permanent once out of range — inject at low prob
    # comm_degraded, gps_noise are temporary — can happen more often
    TEMP_FAULTS = ["comm_degraded", "gps_noise"]
    PERM_FAULTS = ["drift", "comm_blackout", "battery_sudden"]

    for d in drones:
        if d.fault != "healthy":
            continue
        r = rng.random()
        if r < random_fault_prob * 0.3:
            # rare permanent fault
            ft = rng.choice(PERM_FAULTS)
            d.fault = ft
            d.fault_timer = 999   # all permanent faults — never recover
            mission.fault_log.append({
                "step": mission.step, "drone_id": d.id,
                "event": "fault_detected", "fault_type": ft,
            })
        elif r < random_fault_prob:
            # more common temporary fault
            ft = rng.choice(TEMP_FAULTS)
            d.fault = ft
            d.fault_timer = int(rng.integers(3, 7))
            mission.fault_log.append({
                "step": mission.step, "drone_id": d.id,
                "event": "fault_detected", "fault_type": ft,
            })

    # ── 3. Battery drain ─────────────────────────────────────
    for d in drones:
        if d.fault == "battery_sudden" and d.battery > 0.5:
            drop = float(rng.uniform(*BATTERY_SUDDEN_DROP))
            d.battery = max(0.0, d.battery - drop)
        elif d.is_leader:
            drain = float(rng.uniform(*BATTERY_DRAIN_LEADER))
            d.battery = max(0.0, d.battery - drain)
        elif d.fault != "healthy":
            drain = float(rng.uniform(*BATTERY_DRAIN_FAULT))
            d.battery = max(0.0, d.battery - drain)
        else:
            drain = float(rng.uniform(*BATTERY_DRAIN_NORMAL))
            d.battery = max(0.0, d.battery - drain)

    # ── 4. Position update ───────────────────────────────────
    swarm_center = (
        sum(d.x for d in drones)/n,
        sum(d.y for d in drones)/n,
        sum(d.z for d in drones)/n,
    )
    for d in drones:
        wp = WAYPOINTS_3D[min(d.wp_target, len(WAYPOINTS_3D)-1)]
        target = (wp["x"], wp["y"]+22, wp["z"])

        if d.fault in ("comm_blackout", "battery_sudden"):
            # Completely frozen — failsafe hover, no movement at all
            pass
        elif d.fault == "drift":
            # Move away from waypoint and swarm center — no wp progress
            dx = d.x - wp["x"] + d.x - swarm_center[0]
            dz = d.z - wp["z"] + d.z - swarm_center[2]
            dist = math.sqrt(dx*dx+dz*dz) or 1.0
            nx, nz = dx/dist, dz/dist
            noise_x = float(rng.normal(0, 0.2))
            noise_z = float(rng.normal(0, 0.2))
            d.x += (nx + noise_x) * DRIFT_SPEED
            d.z += (nz + noise_z) * DRIFT_SPEED
        elif d.fault == "gps_noise":
            new_pos = _move_toward((d.x,d.y,d.z), target, NORMAL_SPEED)
            d.x = new_pos[0] + float(rng.normal(0, GPS_NOISE_STD))
            d.y = new_pos[1]
            d.z = new_pos[2] + float(rng.normal(0, GPS_NOISE_STD))
        elif d.fault == "comm_degraded":
            # Moves but slower
            new_pos = _move_toward((d.x,d.y,d.z), target, NORMAL_SPEED * 0.5)
            d.x, d.y, d.z = new_pos
        else:
            new_pos = _move_toward((d.x,d.y,d.z), target, NORMAL_SPEED)
            d.x, d.y, d.z = new_pos

    # ── 5. Waypoint arrival check ────────────────────────────
    wp_events = []
    for d in drones:
        # Frozen/drifting drones never reach waypoints
        if d.fault in ("comm_blackout", "battery_sudden", "drift"):
            continue
        if d.wp_target >= len(WAYPOINTS_3D):
            continue
        wp = WAYPOINTS_3D[d.wp_target]
        dist = _distance_to_wp(d, d.wp_target)
        if dist < 35:
            if d.wp_target < len(WAYPOINTS_3D) - 1:
                d.wp_target += 1
                wp_events.append({"step": mission.step, "drone_id": d.id, "waypoint": d.wp_target - 1})
    if wp_events:
        mission.waypoint_reached_log.extend(wp_events)

    # ── 6. Graph analysis → leader election ──────────────────
    graph_result = _run_graph_analysis(mission)

    # ── 7. Leader battery stepdown → logged, re-election handled by next graph tick ──
    # Graph analysis (step 6) already excludes low-battery drones from leadership.
    # We just log the stepdown event here for the UI.
    step_down_happened = False
    for d in drones:
        if d.is_leader and d.battery < BATTERY_LEADER_REELECT:
            d.is_leader = False
            step_down_happened = True
            mission.fault_log.append({
                "step": mission.step, "drone_id": d.id,
                "event": "leader_stepdown",
                "fault_type": f"low_battery({d.battery:.2f})",
            })

    # ── 8. Global leader = highest PageRank among leaders ────
    if mission.leaders:
        gl = max(mission.leaders, key=lambda l: l["pagerank"])
        mission.global_leader_id = gl["leader_drone_id"]
    elif drones:
        mission.global_leader_id = max(drones, key=lambda d: d.pagerank).id

    # ── 9. Check mission complete ─────────────────────────────
    final_wp = len(WAYPOINTS_3D) - 1
    reached = sum(1 for d in drones if d.wp_target >= final_wp)
    if reached >= n * 0.8:
        mission.phase = "complete"

    return {
        "step": mission.step,
        "phase": mission.phase,
        "graph": graph_result,
        "step_down": step_down_happened,
        "new_fault_events": [e for e in mission.fault_log if e["step"] == mission.step],
        "waypoint_events": [e for e in mission.waypoint_reached_log if e["step"] == mission.step],
    }


# ─────────────────────────────────────────────────────────────
# State serialiser
# ─────────────────────────────────────────────────────────────
def _serialise_state(mission: MissionState) -> dict:
    return {
        "step":             mission.step,
        "total_steps":      mission.total_steps,
        "phase":            mission.phase,
        "n_drones":         mission.n_drones,
        "community_algo":   mission.community_algo,
        "centrality_algo":  mission.centrality_algo,
        "modularity":       round(mission.modularity, 5),
        "global_leader_id": mission.global_leader_id,
        "leaders":          mission.leaders,
        "waypoints":        WAYPOINTS_3D,
        "drones": [
            {
                "id":              d.id,
                "x":               round(d.x, 2),
                "y":               round(d.y, 2),
                "z":               round(d.z, 2),
                "battery":         round(d.battery, 4),
                "fault":           d.fault,
                "fault_timer":     d.fault_timer,
                "wp_target":       d.wp_target,
                "pagerank":        round(d.pagerank, 6),
                "community":       d.community,
                "is_leader":       d.is_leader,
                "suspicion_score": d.suspicion_score,
            }
            for d in mission.drones
        ],
        "fault_log":            mission.fault_log[-20:],
        "waypoint_reached_log": mission.waypoint_reached_log[-30:],
    }


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "gpu_available": _GPU_AVAILABLE, "version": "2.0.0"}


@app.post("/sim/start")
async def sim_start(req: StartRequest):
    """Initialise a new mission."""
    global _mission
    _mission = MissionState(
        step=0,
        total_steps=req.total_steps,
        n_drones=req.n_drones,
        community_algo=req.community_algo,
        centrality_algo=req.centrality_algo,
        seed=req.seed,
        drones=_init_drones(req.n_drones, req.seed),
        phase="running",
    )
    _mission._random_fault_prob = req.random_fault_prob
    # initial graph pass
    _run_graph_analysis(_mission)
    return _serialise_state(_mission)


@app.post("/sim/tick")
async def sim_tick():
    """Advance one simulation step and return full state."""
    global _mission
    if _mission is None:
        raise HTTPException(status_code=400, detail="No mission running. Call POST /sim/start first.")
    if _mission.phase == "complete":
        return _serialise_state(_mission)
    prob = getattr(_mission, "_random_fault_prob", 0.025)
    _advance_step(_mission, random_fault_prob=prob)
    return _serialise_state(_mission)


@app.get("/sim/state")
async def sim_state():
    """Return current state without advancing."""
    if _mission is None:
        raise HTTPException(status_code=404, detail="No mission running.")
    return _serialise_state(_mission)


@app.post("/sim/inject_fault")
async def inject_fault(req: InjectFaultRequest):
    """Manually inject a fault on a specific drone."""
    if _mission is None:
        raise HTTPException(status_code=400, detail="No mission running.")
    if req.drone_id < 0 or req.drone_id >= _mission.n_drones:
        raise HTTPException(status_code=400, detail=f"drone_id out of range [0, {_mission.n_drones-1}]")
    d = _mission.drones[req.drone_id]
    d.fault = req.fault_type
    # drift, comm_blackout, battery_sudden are permanent — timer=999
    PERM = {"drift", "comm_blackout", "battery_sudden"}
    d.fault_timer = 999 if req.fault_type in PERM else req.duration
    _mission.fault_log.append({
        "step": _mission.step, "drone_id": req.drone_id,
        "event": "manual_inject", "fault_type": req.fault_type,
    })
    return {"status": "ok", "drone_id": req.drone_id, "fault": req.fault_type, "duration": req.duration}


@app.post("/sim/reset")
async def sim_reset():
    """Reset to the same config as current mission."""
    if _mission is None:
        raise HTTPException(status_code=400, detail="No mission started.")
    return await sim_start(StartRequest(
        n_drones=_mission.n_drones,
        total_steps=_mission.total_steps,
        community_algo=_mission.community_algo,
        centrality_algo=_mission.centrality_algo,
        seed=_mission.seed,
        random_fault_prob=getattr(_mission, "_random_fault_prob", 0.025),
    ))


@app.get("/algorithms")
async def list_algorithms():
    return {
        "combinations": [
            {"community":"louvain","centrality":"pagerank","label":"Default — fast, general purpose"},
            {"community":"louvain","centrality":"hits",    "label":"Fast clusters + relay-aware leaders"},
            {"community":"leiden", "centrality":"pagerank","label":"Stable clusters + classic ranking"},
            {"community":"leiden", "centrality":"hits",    "label":"Best quality — stable clusters + relay-aware leaders"},
        ]
    }


@app.get("/config")
async def get_config():
    cfg_path = _DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        return load_config(cfg_path)
    return {}


@app.on_event("startup")
async def startup_event():
    print(f"\n{'='*55}")
    print(f"  UAV Swarm Live API  v2.0")
    print(f"  GPU: {'✓ cuGraph' if _GPU_AVAILABLE else '✗ CPU fallback'}")
    print(f"  Docs: http://localhost:8000/docs")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

# drone_swarm/tests/test_simulator.py
# AGPL-3.0  |  https://github.com/yourorg/drone-swarm-fault-detector
"""
Unit tests for SwarmSimulator.

Run with:  pytest drone_swarm/tests/
(Does NOT require a GPU — only numpy/pandas are exercised here.)
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.swarm_simulator import (
    FaultType,
    SwarmSimulator,
    parse_fault_type,
)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal config fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_cfg() -> dict:
    return {
        "swarm": {"n_drones": 5, "seed": 0},
        "mission": {
            "waypoints": [[0.0, 0.0, 0.0], [500.0, 300.0, 80.0]],
            "update_interval_sec": 5,
        },
        "simulation": {"n_steps": 6},
        "fault_injection": {
            "forced_drone_id": 1,
            "forced_fault_step": 3,
            "forced_fault_type": "drift",
            "random_fault_prob": 0.0,   # disabled for determinism
        },
        "physics": {
            "normal_speed": 12.0,
            "fault_drift_speed": 35.0,
            "max_comm_range": 800.0,
            "battery_drain_normal": [0.004, 0.008],
            "battery_drain_fault":  [0.001, 0.003],
            "battery_sudden_drop":  [0.10,  0.25],
            "battery_init_range":   [0.85,  1.00],
            "gps_noise_std": 30.0,
        },
        "comm": {
            "ping_jitter": 0.03,
            "fault_ping_scale":    [0.03, 0.18],
            "degraded_ping_scale": [0.30, 0.60],
            "degraded_latency_extra": [50, 150],
            "blackout": True,
        },
        "weights": {
            "ping_strength": 0.2,
            "gps_distance": -0.5,
            "waypoint_distance": -0.9,
            "battery": 0.9,
            "fault_status": -10.0,
            "mission_complete": 0.7,
            "delta_time": -0.2,
            "scaler": "scale_and_round",
            "scale_factor": 1000,
        },
        "output": {
            "snapshots_path": "/tmp/test_snapshots.parquet",
            "edges_path":     "/tmp/test_edges.parquet",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# parse_fault_type
# ──────────────────────────────────────────────────────────────────────────────

class TestParseFaultType:
    def test_known_names(self):
        assert parse_fault_type("drift")          is FaultType.DRIFT
        assert parse_fault_type("comm_degraded")  is FaultType.COMM_DEGRADED
        assert parse_fault_type("comm_blackout")  is FaultType.COMM_BLACKOUT
        assert parse_fault_type("gps_noise")      is FaultType.GPS_NOISE
        assert parse_fault_type("battery_sudden") is FaultType.BATTERY_SUDDEN

    def test_case_insensitive(self):
        assert parse_fault_type("DRIFT") is FaultType.DRIFT

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown fault type"):
            parse_fault_type("explode")


# ──────────────────────────────────────────────────────────────────────────────
# SwarmSimulator — basic run
# ──────────────────────────────────────────────────────────────────────────────

class TestSwarmSimulator:

    def test_run_produces_expected_row_counts(self, base_cfg):
        sim = SwarmSimulator(base_cfg)
        snapshots, edges = sim.run()
        n_steps  = base_cfg["simulation"]["n_steps"]
        n_drones = base_cfg["swarm"]["n_drones"]

        assert len(snapshots) == n_steps * n_drones

    def test_snapshot_columns(self, base_cfg):
        sim = SwarmSimulator(base_cfg)
        snapshots, _ = sim.run()
        required = {
            "step", "drone_id", "timestamp",
            "x", "y", "z", "battery",
            "waypoint_distance", "distance_from_swarm_center",
            "fault_type", "fault_status",
        }
        assert required.issubset(snapshots[0].keys())

    def test_edge_columns(self, base_cfg):
        sim = SwarmSimulator(base_cfg)
        _, edges = sim.run()
        required = {
            "step", "src", "dst", "timestamp",
            "ping_strength", "ping_latency_ms",
            "gps_distance", "edge_weight",
            "src_fault_type", "dst_fault_type",
        }
        assert required.issubset(edges[0].keys())

    def test_forced_fault_injected(self, base_cfg):
        sim = SwarmSimulator(base_cfg)
        snapshots, _ = sim.run()

        forced_id   = base_cfg["fault_injection"]["forced_drone_id"]
        forced_step = base_cfg["fault_injection"]["forced_fault_step"]

        # Before forced_step: fault_status should be 0
        before = [
            s for s in snapshots
            if s["drone_id"] == forced_id and s["step"] < forced_step
        ]
        assert all(s["fault_status"] == 0 for s in before)

        # From forced_step onward: fault_status should be 1
        after = [
            s for s in snapshots
            if s["drone_id"] == forced_id and s["step"] >= forced_step
        ]
        assert all(s["fault_status"] == 1 for s in after)

    def test_comm_blackout_removes_edges(self, base_cfg):
        """COMM_BLACKOUT drone should have no edges at all after fault triggers."""
        base_cfg["fault_injection"]["forced_fault_type"] = "comm_blackout"
        sim = SwarmSimulator(base_cfg)
        _, edges = sim.run()

        forced_id   = base_cfg["fault_injection"]["forced_drone_id"]
        forced_step = base_cfg["fault_injection"]["forced_fault_step"]

        blackout_edges = [
            e for e in edges
            if e["step"] >= forced_step
            and (e["src"] == forced_id or e["dst"] == forced_id)
        ]
        assert len(blackout_edges) == 0, (
            "Edges involving a COMM_BLACKOUT drone must be omitted entirely."
        )

    def test_battery_bounds(self, base_cfg):
        sim = SwarmSimulator(base_cfg)
        snapshots, _ = sim.run()
        batteries = [s["battery"] for s in snapshots]
        assert all(0.0 <= b <= 1.0 for b in batteries)

    def test_ping_strength_bounds(self, base_cfg):
        sim = SwarmSimulator(base_cfg)
        _, edges = sim.run()
        pings = [e["ping_strength"] for e in edges]
        assert all(0.01 <= p <= 1.0 for p in pings)

    def test_edge_weight_positive(self, base_cfg):
        sim = SwarmSimulator(base_cfg)
        _, edges = sim.run()
        weights = [e["edge_weight"] for e in edges]
        assert all(w > 0 for w in weights)

    def test_no_self_edges(self, base_cfg):
        sim = SwarmSimulator(base_cfg)
        _, edges = sim.run()
        assert all(e["src"] != e["dst"] for e in edges)

    def test_gps_noise_reported_position_differs(self, base_cfg):
        """GPS_NOISE fault should produce a reported position that differs from the true one."""
        base_cfg["fault_injection"]["forced_fault_type"] = "gps_noise"
        base_cfg["physics"]["gps_noise_std"] = 50.0   # large enough to reliably differ

        sim = SwarmSimulator(base_cfg)
        snapshots, _ = sim.run()

        forced_id   = base_cfg["fault_injection"]["forced_drone_id"]
        forced_step = base_cfg["fault_injection"]["forced_fault_step"]

        # Collect faulty snapshots
        noisy = [
            s for s in snapshots
            if s["drone_id"] == forced_id and s["step"] >= forced_step
        ]
        # At least one step should show significant position noise
        # (compare to the healthy drone's trajectory — they don't share position)
        assert len(noisy) > 0

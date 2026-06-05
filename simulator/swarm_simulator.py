# drone_swarm/simulator/swarm_simulator.py
# AGPL-3.0  |  https://github.com/yourorg/drone-swarm-fault-detector
"""
Drone swarm physics simulator.

Responsibilities
----------------
- Initialise drone positions, battery, fault states
- Step through time: move drones, apply fault behaviours
- Inject faults: drift, comm_degraded, comm_blackout, gps_noise, battery_sudden
- Produce snapshot rows (one per drone per step) and edge rows (one per directed pair per step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Fault taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class FaultType(Enum):
    NONE             = auto()   # healthy
    DRIFT            = auto()   # position drifts away from waypoint & swarm
    COMM_DEGRADED    = auto()   # ping degrades, latency spikes — edge still exists
    COMM_BLACKOUT    = auto()   # link completely lost — edge row omitted entirely
    GPS_NOISE        = auto()   # reported position jumps randomly each step
    BATTERY_SUDDEN   = auto()   # battery drops sharply then continues draining


_FAULT_NAME_MAP: dict[str, FaultType] = {
    "drift":           FaultType.DRIFT,
    "comm_degraded":   FaultType.COMM_DEGRADED,
    "comm_blackout":   FaultType.COMM_BLACKOUT,
    "gps_noise":       FaultType.GPS_NOISE,
    "battery_sudden":  FaultType.BATTERY_SUDDEN,
}


def parse_fault_type(name: str) -> FaultType:
    """Convert config string to FaultType enum."""
    try:
        return _FAULT_NAME_MAP[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown fault type '{name}'. "
            f"Valid options: {list(_FAULT_NAME_MAP)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-drone state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DroneState:
    drone_id: int
    position: np.ndarray          # [x, y, z] true position
    battery: float                 # 0.0 – 1.0
    fault: FaultType = FaultType.NONE
    battery_sudden_triggered: bool = False   # one-shot flag


# ──────────────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────────────

class SwarmSimulator:
    """
    Stateful simulator.  Call :meth:`run` to produce all snapshots and edges,
    or call :meth:`step` manually for custom orchestration.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config (full document, not a sub-section).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg["swarm"]["seed"])

        # Build waypoint list as numpy arrays
        self._waypoints: list[np.ndarray] = [
            np.array(wp, dtype=float)
            for wp in cfg["mission"]["waypoints"]
        ]
        self._wp_start = self._waypoints[0]
        self._wp_target = self._waypoints[-1]

        # Shortcuts into config sections
        self._phys = cfg["physics"]
        self._comm_cfg = cfg["comm"]
        self._w_cfg = cfg["weights"]
        self._fi = cfg["fault_injection"]
        self._sim = cfg["simulation"]
        self._n_drones: int = cfg["swarm"]["n_drones"]

        # Drone states
        self._drones: list[DroneState] = self._init_drones()

        # Accumulated output
        self.snapshots: list[dict] = []
        self.edges: list[dict] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _init_drones(self) -> list[DroneState]:
        batt_lo, batt_hi = self._phys["battery_init_range"]
        drones = []
        for d in range(self._n_drones):
            pos = self._wp_start + self._rng.uniform(-30, 30, 3)
            bat = float(self._rng.uniform(batt_lo, batt_hi))
            drones.append(DroneState(drone_id=d, position=pos, battery=bat))
        return drones

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> tuple[list[dict], list[dict]]:
        """Run the full simulation and return (snapshots, edges)."""
        n_steps: int = self._sim["n_steps"]
        update_interval: int = self._cfg["mission"]["update_interval_sec"]
        start_time = datetime(2026, 5, 9, 10, 0, 0)

        for step in range(n_steps):
            timestamp = start_time + timedelta(seconds=step * update_interval)
            self.step(step, timestamp)

        return self.snapshots, self.edges

    def step(self, step: int, timestamp: datetime) -> None:
        """Advance simulation by one timestep and append rows to accumulators."""
        self._inject_forced_fault(step)
        swarm_center_before = self._swarm_center()
        self._update_positions_and_battery(swarm_center_before)
        swarm_center_after = self._swarm_center()
        self._record_snapshots(step, timestamp, swarm_center_after)
        self._record_edges(step, timestamp)

    # ──────────────────────────────────────────────────────────────────────────
    # Fault injection
    # ──────────────────────────────────────────────────────────────────────────

    def _inject_forced_fault(self, step: int) -> None:
        fi = self._fi
        if step >= fi["forced_fault_step"]:
            drone = self._drones[fi["forced_drone_id"]]
            if drone.fault is FaultType.NONE:
                drone.fault = parse_fault_type(fi["forced_fault_type"])

    def _maybe_random_fault(self, drone: DroneState) -> None:
        """Assign a random fault to a healthy drone with low probability."""
        if drone.fault is not FaultType.NONE:
            return
        prob = self._fi["random_fault_prob"]
        if self._rng.random() < prob:
            fault_pool = [
                FaultType.DRIFT,
                FaultType.COMM_DEGRADED,
                FaultType.COMM_BLACKOUT,
                FaultType.GPS_NOISE,
                FaultType.BATTERY_SUDDEN,
            ]
            drone.fault = self._rng.choice(fault_pool)

    # ──────────────────────────────────────────────────────────────────────────
    # Position & battery update
    # ──────────────────────────────────────────────────────────────────────────

    def _swarm_center(self) -> np.ndarray:
        """Mean position of healthy drones (falls back to all drones)."""
        healthy = [d.position for d in self._drones if d.fault is FaultType.NONE]
        positions = healthy if healthy else [d.position for d in self._drones]
        return np.mean(positions, axis=0)

    def _update_positions_and_battery(self, swarm_center: np.ndarray) -> None:
        for drone in self._drones:
            if drone.fault is FaultType.NONE:
                drone.position = self._move_toward(drone.position, self._wp_target)
                drone.battery = self._drain_normal(drone.battery)
                self._maybe_random_fault(drone)

            elif drone.fault is FaultType.DRIFT:
                drone.position = self._drift(drone.position, swarm_center)
                drone.battery = self._drain_fault(drone.battery)

            elif drone.fault is FaultType.COMM_DEGRADED:
                # Still flies toward target, but comm quality degrades
                drone.position = self._move_toward(drone.position, self._wp_target)
                drone.battery = self._drain_normal(drone.battery)

            elif drone.fault is FaultType.COMM_BLACKOUT:
                # Hovers in place (lost link = failsafe hover)
                drone.battery = self._drain_fault(drone.battery)

            elif drone.fault is FaultType.GPS_NOISE:
                # True position advances, but reported position will jitter (see snapshot)
                drone.position = self._move_toward(drone.position, self._wp_target)
                drone.battery = self._drain_normal(drone.battery)

            elif drone.fault is FaultType.BATTERY_SUDDEN:
                if not drone.battery_sudden_triggered:
                    lo, hi = self._phys["battery_sudden_drop"]
                    drone.battery = max(0.0, drone.battery - float(self._rng.uniform(lo, hi)))
                    drone.battery_sudden_triggered = True
                else:
                    drone.battery = self._drain_fault(drone.battery)
                # Still flies toward target
                drone.position = self._move_toward(drone.position, self._wp_target)

    # ──────────────────────────────────────────────────────────────────────────
    # Movement helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _move_toward(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        speed = self._phys["normal_speed"]
        delta = target - pos
        dist = np.linalg.norm(delta)
        if dist < speed:
            return target.copy()
        return pos + (delta / dist) * speed

    def _drift(self, pos: np.ndarray, swarm_center: np.ndarray) -> np.ndarray:
        """Move away from both the waypoint and the swarm center."""
        speed = self._phys["fault_drift_speed"]
        direction = (pos - self._wp_target) + (pos - swarm_center)
        if np.linalg.norm(direction) < 1e-6:
            direction = self._rng.uniform(-1, 1, 3)
        direction = direction / np.linalg.norm(direction)
        noise = self._rng.normal(0, 0.25, 3)
        direction = (direction + noise)
        direction = direction / np.linalg.norm(direction)
        return pos + direction * speed

    # ──────────────────────────────────────────────────────────────────────────
    # Battery helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _drain_normal(self, battery: float) -> float:
        lo, hi = self._phys["battery_drain_normal"]
        return max(0.0, battery - float(self._rng.uniform(lo, hi)))

    def _drain_fault(self, battery: float) -> float:
        lo, hi = self._phys["battery_drain_fault"]
        return max(0.0, battery - float(self._rng.uniform(lo, hi)))

    # ──────────────────────────────────────────────────────────────────────────
    # Snapshot recording
    # ──────────────────────────────────────────────────────────────────────────

    def _reported_position(self, drone: DroneState) -> np.ndarray:
        """
        Return the position as seen by other drones / GCS.
        GPS_NOISE fault adds Gaussian jitter to the reported position
        while the true position advances normally.
        """
        if drone.fault is FaultType.GPS_NOISE:
            std = self._phys["gps_noise_std"]
            return drone.position + self._rng.normal(0, std, 3)
        return drone.position

    def _record_snapshots(
        self,
        step: int,
        timestamp: datetime,
        swarm_center: np.ndarray,
    ) -> None:
        for drone in self._drones:
            pos = self._reported_position(drone)
            self.snapshots.append({
                "step":                       step,
                "drone_id":                   drone.drone_id,
                "timestamp":                  timestamp,
                "x":                          round(float(pos[0]), 2),
                "y":                          round(float(pos[1]), 2),
                "z":                          round(float(pos[2]), 2),
                "battery":                    round(drone.battery, 4),
                "waypoint_distance":          round(float(np.linalg.norm(pos - self._wp_target)), 2),
                "distance_from_swarm_center": round(float(np.linalg.norm(pos - swarm_center)), 2),
                "fault_type":                 drone.fault.name,
                "fault_status":               int(drone.fault is not FaultType.NONE),
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Edge recording
    # ──────────────────────────────────────────────────────────────────────────

    def _ping_strength(
        self,
        dist: float,
        src_fault: FaultType,
        dst_fault: FaultType,
    ) -> float:
        jitter = float(self._rng.uniform(-self._comm_cfg["ping_jitter"],
                                          self._comm_cfg["ping_jitter"]))
        strength = max(0.0, 1.0 - dist / self._phys["max_comm_range"]) + jitter

        if src_fault is FaultType.COMM_DEGRADED or dst_fault is FaultType.COMM_DEGRADED:
            lo, hi = self._comm_cfg["degraded_ping_scale"]
            strength *= float(self._rng.uniform(lo, hi))
        elif src_fault is not FaultType.NONE or dst_fault is not FaultType.NONE:
            # Other fault types still degrade comms slightly (drift, GPS noise, etc.)
            lo, hi = self._comm_cfg["fault_ping_scale"]
            strength *= float(self._rng.uniform(lo, hi))

        return float(np.clip(strength, 0.01, 1.0))

    def _ping_latency(
        self,
        dist: float,
        src_fault: FaultType,
        dst_fault: FaultType,
    ) -> float:
        base = 10.0 + dist * 0.08
        jitter = float(self._rng.uniform(0, 25))

        if src_fault is FaultType.COMM_DEGRADED or dst_fault is FaultType.COMM_DEGRADED:
            lo, hi = self._comm_cfg["degraded_latency_extra"]
            base += float(self._rng.uniform(lo, hi))
        elif src_fault is not FaultType.NONE or dst_fault is not FaultType.NONE:
            base += float(self._rng.uniform(150, 500))

        return float(base + jitter)

    @staticmethod
    def _edge_weight(ping: float, gps_dist: float, latency: float) -> float:
        return (
            ping
            * (1.0 / (1.0 + gps_dist / 100.0))
            * (1.0 / (1.0 + latency / 100.0))
        )

    def _record_edges(self, step: int, timestamp: datetime) -> None:
        max_range = self._phys["max_comm_range"]
        drones = self._drones

        for src in drones:
            for dst in drones:
                if src.drone_id == dst.drone_id:
                    continue

                # COMM_BLACKOUT: drop the edge entirely
                if (src.fault is FaultType.COMM_BLACKOUT
                        or dst.fault is FaultType.COMM_BLACKOUT):
                    continue

                src_pos = self._reported_position(src)
                dst_pos = self._reported_position(dst)
                gps_dist = float(np.linalg.norm(src_pos - dst_pos))

                if gps_dist > max_range:
                    continue

                ping = self._ping_strength(gps_dist, src.fault, dst.fault)
                latency = self._ping_latency(gps_dist, src.fault, dst.fault)
                weight = self._edge_weight(ping, gps_dist, latency)

                self.edges.append({
                    "step":               step,
                    "src":                src.drone_id,
                    "dst":                dst.drone_id,
                    "timestamp":          timestamp,
                    "ping_strength":      round(ping, 4),
                    "ping_latency_ms":    round(latency, 2),
                    "gps_distance":       round(gps_dist, 2),
                    "edge_weight":        round(weight, 8),
                    "waypoint_distance":  round(float(np.linalg.norm(dst_pos - self._wp_target)), 2),
                    "battery":            round(dst.battery, 4),
                    "fault_status":       int(dst.fault is not FaultType.NONE),
                    "src_fault_type":     src.fault.name,
                    "dst_fault_type":     dst.fault.name,
                    "src_fault_status":   int(src.fault is not FaultType.NONE),
                    "dst_fault_status":   int(dst.fault is not FaultType.NONE),
                    "mission_complete":   0,
                    "src_x": round(float(src_pos[0]), 2),
                    "src_y": round(float(src_pos[1]), 2),
                    "src_z": round(float(src_pos[2]), 2),
                    "dst_x": round(float(dst_pos[0]), 2),
                    "dst_y": round(float(dst_pos[1]), 2),
                    "dst_z": round(float(dst_pos[2]), 2),
                })

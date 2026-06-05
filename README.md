# UAV-Swarm-Self-Organization

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-cuGraph%20%2F%20cuDF-green.svg)](https://rapids.ai/)

**Decentralised self-organization framework for UAV swarms**, powered by NVIDIA RAPIDS cuGraph.

When part of a swarm fails, the remaining healthy drones should reorganise themselves — without intervention from a ground station — into coherent sub-clusters, each with an elected leader that can act as a relay or sub-mission coordinator. This project provides that self-organization layer:

1. **Fault detection** — anomalous drones are identified by their communication graph topology (isolated nodes + statistical PageRank outliers), not by direct telemetry inspection.
2. **Community formation** — surviving drones are grouped into sub-clusters via Louvain community detection on the weighted comm graph.
3. **Leader election** — each sub-cluster elects the drone with the highest PageRank as its leader.

The whole pipeline runs on a single GPU and produces an updated swarm topology every timestep.

---

## Problem

In a UAV swarm, individual drones can fail in ways that are difficult to detect from a single vantage point:

- A drone may **drift away** from its assigned waypoint while still transmitting.
- A drone may suffer a **communication blackout** and silently disappear from the mesh.
- A drone may report **corrupted GPS** coordinates, misleading neighbours.
- A drone may experience a **sudden battery drop**, degrading its flight capability.

Centralised health monitoring creates a single point of failure and does not scale.  
This project treats the swarm as a **weighted communication graph** and detects anomalies from the graph topology itself.

---

## How it works

```
Each timestep
─────────────────────────────────────────────────────────
Drone telemetry   →   Directed edge per (src, dst) pair
                       weight = f(ping strength, GPS distance,
                                  latency, battery, waypoint distance)

Sliding-window filter   →   Keep only the last N steps
                             (stale edges from drones that left
                              comm range are excluded)

cuGraph Louvain         →   Community partitions + modularity score
cuGraph PageRank        →   Per-drone influence score

Isolated node recovery  →   Drones absent from Louvain output
                             (zero edges in window) are re-inserted
                             as singleton communities with pagerank = 0

Fault scoring
  isolated     +2 pt   drone is a singleton community
  low PageRank +1 pt   pagerank < mean − 2σ

Leader election         →   Highest-PageRank drone per community
                             becomes the community leader
─────────────────────────────────────────────────────────
```

### Why sliding window?

Using all historical edges keeps dead links alive.
A DRIFT drone that left comm range at step 12 is still "connected"
via its step 0–11 edges if the full history is used.
The sliding window ensures the graph reflects the **current topology**.

### Why two separate graphs?

cuGraph Louvain requires an **undirected** graph.  
cuGraph PageRank requires a **directed** graph (`store_transposed=True`).  
Both are built from the same edge data in the same preprocessing pass.

### Why isolated node recovery?

cuGraph silently drops vertices with no edges from Louvain output.
A drone with zero edges in the window is the **strongest fault signal** —
it is almost certainly blacked out or beyond comm range.
`GraphEngine._restore_isolated_nodes()` re-inserts such drones
as singleton communities with `pagerank = 0.0` before scoring.

---

## Fault types simulated

| Type | Drone behaviour | Graph signature |
|---|---|---|
| `drift` | Position drifts away from waypoint and swarm center | Edges thin as distance grows → isolated after comm range exceeded |
| `comm_degraded` | Ping quality drops, latency spikes (link stays up) | Low edge weight → low PageRank |
| `comm_blackout` | RF link completely lost | Edge rows omitted entirely → immediately isolated |
| `gps_noise` | Reported position jumps ±N metres per step | Noisy GPS distance on all adjacent edges |
| `battery_sudden` | Battery drops sharply then drains normally | Low battery weight contribution on edges |

---

## Suspicion scoring

| Signal | Points | Condition |
|---|---|---|
| `isolated` | **+2** | Drone is a singleton Louvain community (zero edges in window) |
| `low_pagerank` | **+1** | PageRank < mean − 2σ across the swarm |

A drone with `suspicion_score ≥ 1` is reported as a suspected fault.  
`isolated` is weighted 2× because zero edges is an unambiguous structural signal,
whereas low PageRank can arise from transient edge degradation.

---

## Leader election

After Louvain partitioning, the drone with the **highest PageRank** in each community is elected as the community leader.

PageRank in this context measures how many high-quality communication links a drone has, weighted by the quality of its neighbours' links in turn. A leader drone is:

- Centrally positioned in its sub-cluster (low average GPS distance to peers)
- Reachable with low latency and high ping strength from most community members
- Operating with healthy battery

Leaders are reported per community and can serve as relay nodes or sub-swarm coordinators after the main swarm splits due to a fault event.

---

## Requirements

### Hardware
- NVIDIA GPU with **compute capability 7.0+** (Volta or later — V100, T4, A100, H100, RTX 20-series and above)
- CUDA-capable Linux distribution or WSL2 on Windows 11

### Software
- **CUDA 12.2+** with a compatible NVIDIA driver (check with `nvidia-smi`)
- **Python 3.10 / 3.11 / 3.12 / 3.13**
- **RAPIDS** (`cudf` + `cugraph`) — latest stable release recommended

### Installing RAPIDS

The recommended way is via conda. Run the [RAPIDS install selector](https://docs.rapids.ai/install/) to get the exact command for your CUDA version, or use the latest stable:

```bash
# CUDA 12
conda install -c rapidsai -c conda-forge \
    cudf cugraph cuda-version=12.9

# CUDA 13
conda install -c rapidsai -c conda-forge \
    cudf cugraph cuda-version=13.1
```

Or with pip (CUDA 12 example):

```bash
pip install cudf-cu12 cugraph-cu12 \
    --extra-index-url https://pypi.nvidia.com
```

Pinning a specific RAPIDS version (e.g. `cugraph=25.10`) is fine if you need reproducibility — this project tracks the stable cuGraph API and should work on any RAPIDS release from 24.x onward.

### Python packages

```bash
pip install -r requirements.txt
```

---

## Project structure

```
UAV-Swarm-Self-Organization/
├── config/
│   └── default.yaml          # all tunable parameters — no magic numbers in source
├── simulator/
│   └── swarm_simulator.py    # drone physics + 5 fault types (FaultType enum)
├── graph/
│   └── graph_engine.py       # sliding window, cuGraph Louvain + PageRank,
│                             # isolated node recovery
├── detector/
│   └── fault_detector.py     # z-score scoring, leader election, report
├── utils/
│   ├── config.py             # YAML loader + dot-notation CLI overrides
│   └── io.py                 # Parquet save / load
├── data/                     # output Parquet files (gitignored)
├── examples/
│   ├── README.md             # scenario descriptions and result tables
│   └── drift_40drones_50steps.log
├── tests/
│   └── test_simulator.py     # pytest — GPU not required
├── main.py                   # CLI entry point
├── requirements.txt
├── NOTICE                    # copyright + third-party licenses
└── CHANGELOG.md
```

---

## Usage

```bash
# Default: 20 drones, 10 steps, drift fault injected on drone 7
python main.py

# Larger scenario with tighter window
python main.py --steps 50 --drones 40 --window 3

# Test a different fault type
python main.py --steps 50 --drones 40 --fault-type comm_blackout

# Reproducible run (disable random faults)
# Set random_fault_prob: 0.0 in config/default.yaml, then:
python main.py --steps 50 --drones 40 --seed 42
```

### CLI options

| Option | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Path to YAML config |
| `--steps` | 10 | Number of simulation timesteps |
| `--drones` | 20 | Number of drones in the swarm |
| `--window` | 5 | Recent steps included in the graph (0 = all) |
| `--fault-type` | `drift` | Fault type for the forced drone |
| `--seed` | 42 | RNG seed |

### Key config parameters

```yaml
simulation:
  n_steps: 10

graph:
  window_steps: 5       # 0 = cumulative (all steps)

fault_injection:
  forced_drone_id: 7
  forced_fault_step: 4
  forced_fault_type: drift
  random_fault_prob: 0.001   # set 0.0 for single-fault reproducible runs

physics:
  max_comm_range: 800.0      # metres — drones beyond this produce no edge
```

---

## Output

Two Parquet files written to `data/`:

| File | Rows | Description |
|---|---|---|
| `drone_snapshots.parquet` | `n_drones × n_steps` | Per-drone state: position, battery, fault label |
| `drone_edges.parquet` | up to `n_drones² × n_steps` | Directed comm edges: ping, latency, GPS dist, weights |

Both files can be used directly as training data for downstream ML fault classifiers.

---

## Example output

```
[graph_engine] Using 4,218 edges from 3 most recent step(s) (window=3, total=75,188)
[graph_engine] Isolated nodes (no edges in window): [7, 25]

════════════════════════════════════════════════════════════
  FAULT DETECTION REPORT
════════════════════════════════════════════════════════════
  Network cohesion (Louvain modularity): 0.0158
  Communities detected : 4  (of which 2 singleton)

  ⚠ Suspected faulty drones (2):
    Drone  25  [score=3]  |  ISOLATED (no edges in window), low PageRank (0.0000, z=-3.65)
    Drone   7  [score=3]  |  ISOLATED (no edges in window), low PageRank (0.0000, z=-3.65)

  Swarm community leaders:
    Community   0  →  Drone   3  (PageRank 0.0301, size=17)
    Community   1  →  Drone  14  (PageRank 0.0281, size=21)
  Isolated (no-edge) drones: [7, 25]
════════════════════════════════════════════════════════════
```

See [`examples/`](examples/) for full run logs and scenario breakdowns.

---

## Run tests

```bash
pytest tests/ -v
```

Tests cover the simulator and do not require a GPU.

---

## License

Copyright (C) 2026 [Klastrovanie Co., Ltd.]

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. See [LICENSE](LICENSE) for the full text.

AGPL v3 — free for research and non-commercial use.  
Commercial use requires a separate agreement.

This code is released to encourage collaboration across AI systems — not competition.  
The goal is shared solutions, not shared resources.

For commercial licensing: leave a message on [Discussions](../../discussions)


# UAV-Swarm-Self-Organization

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-cuGraph%20%2F%20cuDF-green.svg)](https://rapids.ai/)

**Decentralised self-organization framework for UAV swarms**, powered by NVIDIA RAPIDS cuGraph.

**First Public Release:** 2026-06-05  
**Last Updated:** 2026-06-09

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
 
## Why leader election matters
 
Without an elected leader per sub-cluster, the ground control station (GCS) must maintain a direct communication link to every drone in the swarm. For a swarm of N drones this is **O(N) links**, each consuming bandwidth, RF spectrum, and CPU on the GCS side. It also exposes every drone as a potential point of communication failure with the GCS.
 
With leader election, the GCS only needs to talk to **one leader per community**:
 
```
Without leader election               With leader election
─────────────────────────             ─────────────────────────
       GCS                                    GCS
      /│\\..\\                                 │
     / │ \\ ..\\                               │  (1 link)
    /  │  \\  .\\                              ▼
   ▼   ▼   ▼   ▼                              Leader
  D1  D2  D3 ... DN                          / │ \\
                                            ▼  ▼  ▼   (intra-cluster mesh)
                                           D1 D2 D3      
   O(N) GCS links                          O(1) GCS link per cluster
```
 
Concrete benefits:
 
- **Bandwidth reduction**: GCS uplink/downlink scales with number of clusters, not drones. For a 40-drone swarm split into 2 clusters, GCS communication drops from 40 links to 2.
- **Resilience**: if a single drone loses its link to the GCS, only that one drone is affected — the leader still relays for the rest of the cluster.
- **Lower latency for swarm-internal coordination**: intra-cluster messages stay local instead of round-tripping through the GCS.
- **GPS/RF spectrum efficiency**: fewer simultaneous GCS-to-drone channels reduces interference.
- **Automatic failover**: when a leader itself fails, the next PageRank run elects a new one without GCS intervention.
This is the same architectural pattern used in cellular networks (cluster head / sector controller) and Byzantine consensus protocols (leader-based replication), adapted for ad-hoc UAV mesh networks where the leader can change every few seconds based on current topology.
 
---
 
## How leader election works
 
After Louvain partitioning, the drone with the **highest PageRank** in each community is elected as the community leader.
 
PageRank in this context measures how many high-quality communication links a drone has, weighted by the quality of its neighbours' links in turn. A leader drone tends to be:
 
- Centrally positioned in its sub-cluster (low average GPS distance to peers)
- Reachable with low latency and high ping strength from most community members
- Operating with healthy battery (which influences edge weight)
Because the edge weights incorporate ping strength, latency, GPS distance, and battery, the elected leader is automatically the drone best suited to act as a relay — both physically (central position) and operationally (healthy state).
 
Leaders are reported per community every timestep, so when the swarm splits or merges due to a fault event, the leader set updates automatically without any external coordination.

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
 
# What's New in v2.0.0
 
> Released 2026-06-09
 
v2.0.0 adds a **real-time live simulation API** and a **3D WebGL visualizer**, turning the batch pipeline into an interactive mission demo. The core simulation and graph engine are unchanged — all new additions are strictly additive.
- server.py: added
- drone_swarm_3d.html: added
- some other files: modified (see commit logs)
 
---
 
## New: Algorithm selection (Louvain / Leiden + PageRank / HITS)
 
`graph_engine.py` now accepts two parameters:
 
```python
engine = GraphEngine(cfg, community_algo="leiden", centrality_algo="hits")
```
 
| Community | Centrality | Characteristics |
|---|---|---|
| `louvain` | `pagerank` | Default — fast, general purpose |
| `louvain` | `hits` | Fast clusters + relay-aware leaders (Hub score) |
| `leiden` | `pagerank` | Stable clusters + classic ranking |
| `leiden` | `hits` | Best quality — stable clusters + relay-aware leaders |
 
**HITS** returns `hub_score` (best relay/broadcaster) and `authority_score` (most trusted receiver) separately. Leader election uses `hub_score`. A `pagerank` alias column is always present so `fault_detector.py` works unchanged regardless of algorithm.
 
---
 
## New: Fault model — permanent vs temporary
 
Faults are now classified by recoverability:
 
| Type | Recovery | Behaviour |
|---|---|---|
| `drift` | ❌ Permanent | Drifts away from swarm indefinitely |
| `comm_blackout` | ❌ Permanent | Frozen in place, no edges ever |
| `battery_sudden` | ❌ Permanent | Battery drop is irreversible |
| `comm_degraded` | ✅ Temporary (3–7 steps) | Signal quality recovers |
| `gps_noise` | ✅ Temporary (3–7 steps) | Sensor noise clears |
 
Temporary faults reflect real-world RF interference patterns — the same drone may degrade and recover multiple times during a mission.
 
---
 
## New: Fault-aware leader election
 
`fault_detector.get_community_leaders()` now accepts `drone_faults` and `drone_batteries` maps and excludes permanently faulted or low-battery drones from leadership candidacy:
 
```python
community_leaders = get_community_leaders(
    community_groups, centrality_scores,
    drone_faults=fault_map,
    drone_batteries=battery_map,
    battery_threshold=0.70,
)
```
 
A safety net guarantees at least one leader always exists even if all candidates are faulted.
 
---
 
## New: Leader battery drain + automatic stepdown
 
Leaders consume battery at 2× the normal rate (they handle relay traffic). When a leader's battery drops below 70%, it steps down and the next highest-ranked healthy drone in the same community is automatically elected — logged as a `leader_stepdown` event.
 
Leader count is capped at `max(2, min(8, sqrt(n)))` to prevent explosion with large swarms.
 
---
 
## New: Live simulation API (`server.py`)
 
A FastAPI server exposes the simulation as a step-by-step stateful API. The HTML visualizer polls it every tick.
 
```bash
# On the GPU server (EC2 or local)
python server.py
# → http://0.0.0.0:8000
# → http://localhost:8000/docs  (Swagger UI)
```
 
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sim/start` | Initialise a new mission |
| `POST` | `/sim/tick` | Advance one step, run graph analysis, return full state |
| `POST` | `/sim/inject_fault` | Manually inject a fault on a specific drone |
| `POST` | `/sim/reset` | Reset to same config |
| `GET` | `/sim/state` | Current state without advancing |
| `GET` | `/health` | Liveness check (includes `gpu_available`) |
| `GET` | `/algorithms` | Supported algorithm combinations |
 
**No GPU required to start the server.** Without cuGraph, graph analysis falls back to a CPU k-means + iterative PageRank approximation automatically.
 
---
 
## New: 3D Real-Time Visualizer (`drone_swarm_3d.html`)
 
A standalone HTML file (no build step) that connects to the live API and renders the swarm in 3D via Three.js.
 
**Features:**
- 5-waypoint mission over procedural terrain (trees, ridges, valley, river)
- Per-community drone colouring — only leaders show ID labels
- 🟠 Orange = global leader, ⬜ White = community leader, 🔴 Pink/Red = fault
- Smooth LERP interpolation between server ticks (teleport detection included)
- ⚙ Config panel — drones, steps, fault probability, tick speed, algorithm
- ⚡ Inject Fault button — target any drone with any fault type
- Event log panel — fault detected / recovered / leader stepdown / waypoint reached
- Mouse drag to orbit, scroll to zoom
**Project structure additions (v2.0.0):**
 
```
UAV-Swarm-Self-Organization/
├── server.py                 # FastAPI live simulation API  ← NEW
├── drone_swarm_3d.html       # Three.js 3D visualizer       ← NEW
├── port-mapping.sh           # SSH tunnel helper (local PC) ← NEW
└── run-local-server.sh       # Local HTML server (local PC) ← NEW
```
 
---
 
## Running the real-time demo
 
**Step 1 — Start the API server (GPU machine / EC2)**
 
```bash
cd ~/UAV-Swarm-Self-Organization
python server.py
```
 
**Step 2 — SSH tunnel (local PC, if server is remote)**
 
```bash
bash port-mapping.sh
# Forwards localhost:8888 → EC2:8000
```
 
**Step 3 — Serve the HTML locally**
 
```bash
bash run-local-server.sh
# Serves on http://localhost:3000
```
 
**Step 4 — Open in browser**
 
```
http://localhost:3000/drone_swarm_3d.html
```
 
Top bar shows **● Server Connected** when the API is reachable. Use ⚙ Config to set drone count, fault probability, and algorithm before starting.

## Live Demo Video Recording:
[![UAV Swarm Self-Organization Demo](https://img.youtube.com/vi/h7SSGxUz5Nc/0.jpg)](https://youtu.be/h7SSGxUz5Nc)

https://www.youtube.com/watch?v=B0FfCWOXL8M 


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


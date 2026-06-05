# Examples

## drift — 40 drones, 50 steps, window 3

**Command**
```bash
python main.py --steps 50 --drones 40 --window 3
```

**Scenario**
- 40 drones flying from WP0 `[0, 0, 0]` → WP1 `[500, 300, 80]`
- Drone 7: `DRIFT` fault injected at step 4
- Drone 25: random fault triggered during simulation (`random_fault_prob=0.001`)
- Graph built from last 3 steps only (`window=3`)

**Key results**

| Metric | Value |
|---|---|
| Simulation time | 2.34s |
| Total edges | 75,188 |
| Edges in window | 4,218 |
| Louvain modularity | 0.0158 |
| Communities | 4 (2 main + 2 singleton) |
| Suspected drones | 2 (drone 7, drone 25) |
| False positives | 0 |

**Detection logic**

Drone 7 drifts ~35 m/step away from the swarm.
By step 20, `distance_from_swarm_center` exceeds 800 m (comm range limit),
so all edges to/from drone 7 are absent in the window-3 graph.
cuGraph Louvain returns no partition entry for drone 7;
`GraphEngine._restore_isolated_nodes()` re-inserts it as a singleton community
with `pagerank=0.0`, giving `suspicion_score=3` (isolated×2 + low_pagerank×1).

**Full log**

See [`drift_40drones_50steps.log`](drift_40drones_50steps.log).

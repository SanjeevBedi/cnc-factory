# CNC Factory — Implementation Plan

## Project Location
`/Users/sbedi/Nextcloud/automatic_to_autonomous/CNC Factory/`

## Hard Constraints
- `Build_Solid.py` is **NEVER modified** — it is used by other projects.
- Communication with Build_Solid.py is through `.npy` / `.npz` files on disk
  and subprocess calls **only**. No direct imports.
- OCC (pyocc conda env) is isolated to `cnc_solid_bridge.py` subprocess calls only.
- All CNC Factory code lives in this directory.

---

## Seed Creation Tick
Every scheduler tick a new part seed is created with probability `SEED_CREATION_PROB`.

```
Default:          0.00002% = 2e-7 per tick
t_unit:           60 s per tick  (configurable)
Average interval: ~5,000,000 ticks  (~9.5 sim-years at default)
Testing override: --force-seed flag bypasses probability check
```

---

## Full Pipeline

```
[pyocc env — UNCHANGED]              [standard Python env — this project]
─────────────────────────            ────────────────────────────────────────────
Build_Solid.py
  └─ solid_faces_seed_N.npy ───────→ cnc_solid_bridge.py
  └─ connectivity_N.npz               └─ subprocess wrapper + .npy reader

                                      feature_extractor.py       ← Phase 1
                                        └─ Newell normal computation
                                        └─ edge labels: Wall / Cliff / Level
                                        └─ safe region  (Shapely buffer)

                                      feeds_speeds_engine.py     ← Phase 2
                                        └─ RPM  = (1000·V)/(π·D)
                                        └─ Feed = N·Z·fz
                                        └─ Fc   = Kc·Ac  (physics check)
                                        └─ P    = Fc·V/60 (power check)

                                      toolpath_planner.py        ← Phase 3
                                        └─ raster paths inside safe region
                                        └─ entry: ramp / helix / plunge

                                      gcode_generator.py         ← Phase 4
                                        └─ waypoints → validated G-code

                                      scheduler.py               ← Phase 5
                                        └─ seed creation tick (2e-7 prob)
                                        └─ multi-machine job assignment
                                        └─ J = w1·Tm + w2·C + w3/L + w4·S

HTTP ←──────────────────────────────── cnc_llm_client.py  (existing, unchanged)
cnc_sim_api_server.py                   └─ program(), get_state(), reset()
  └─ cnc_simulator.py

                                      cnc_agent.py               ← Phase 6
                                        └─ per-machine LLM error handler
                                        └─ anomaly → context → LLM → safety

                                      factory_agent.py           ← Phase 7
                                        └─ tool crib management
                                        └─ factory LLM orchestration
```

---

## Edge Label Definitions

| Label   | Condition | CNC Meaning |
|---------|-----------|-------------|
| `wall`  | Adjacent vertical face max-z **>** top-face z | Wall rises above — tool **cannot** enter from this side |
| `cliff` | No adjacent face, OR adjacent vertical face max-z **≤** top-face z | Open air / step-down — **safe** entry direction |
| `level` | Adjacent face is horizontal (normal.z > 0.9) | Another flat face at same height — safe lateral move |

---

## Phase Build Order

| Phase | File(s) | Status |
|-------|---------|--------|
| 1 | `config.py`, `cnc_solid_bridge.py`, `feature_extractor.py`, `test_feature_extractor.py` | ✅ Built |
| 2 | `feeds_speeds_engine.py` + `test_feeds_speeds.py` | ✅ Built |
| 3 | `toolpath_planner.py` + `test_toolpath_planner.py` | ✅ Built |
| 4 | `gcode_generator.py` + `test_gcode_generator.py` | ✅ Built |
| 5 | `scheduler.py` + `test_scheduler.py` | ✅ Built |
| 6 | `cnc_agent.py` + `test_cnc_agent.py` | ✅ Built |
| 7 | `factory_agent.py` + `test_factory_agent.py` | ✅ Built |
| — | `factory_main.py` | ✅ Built |

---

## File Map

```
CNC Factory/
├── PLAN.md                    ← this file
├── config.py                  ← all tuneable constants
├── cnc_solid_bridge.py        ← ONLY file that touches Build_Solid.py output
├── feature_extractor.py       ← Phase 1: edge labels + safe regions
├── test_feature_extractor.py  ← Phase 1: tests (synthetic + real seeds)
├── feeds_speeds_engine.py     ← Phase 2
├── toolpath_planner.py        ← Phase 3
├── gcode_generator.py         ← Phase 4
├── scheduler.py               ← Phase 5 (includes seed tick)
├── cnc_agent.py               ← Phase 6
├── factory_agent.py           ← Phase 7
└── factory_main.py            ← Orchestrator
```

---

## Material Reference (Kc in MPa)

| Material           | SFM Range  | Kc (MPa) |
|--------------------|-----------|----------|
| Aluminium 6061     | 800–1200  | 600      |
| Aluminium cast     | 600–900   | 700      |
| Mild Steel 1018    | 300–500   | 1800     |
| Alloy Steel 4140   | 200–400   | 2200     |
| Stainless 304      | 150–300   | 2400     |
| Tool Steel D2      | 100–250   | 3000     |
| Cast Iron          | 400–700   | 1600     |
| Brass              | 500–1000  | 500      |
| Titanium Ti-6Al-4V | 80–200    | 3000     |
| Inconel            | 50–150    | 3500     |

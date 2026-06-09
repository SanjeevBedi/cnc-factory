# CNC Factory — Conversation Log & Session Memory
> Last updated: Session 4 (Toolpath loop fix + Completed Parts + Auto-seed generation)
> Purpose: Recall context, decisions, and progress across chat sessions.

---

## Project Overview

**Name:** Autonomous CNC Factory Simulation  
**Location:** `/Users/sbedi/Nextcloud/automatic_to_autonomous/CNC Factory/`  
**Goal:** A fully visual, agent-based autonomous CNC factory simulation where:
- Seeds generate random 3D solids (via `Build_Solid.py` in pyocc conda env)
- A full Phase 1–7 pipeline processes each solid into a running CNC job
- 4 CNC machine agents run jobs, report errors, and interact with a factory LLM agent
- The factory agent queries OpenAI ChatGPT (3 temperature candidates), scores responses, and dispatches the best command
- All of this is visible in a live tkinter GUI with per-machine panels, agent chat dialogs, and disturbance injection

---

## Hard Constraints (NEVER violate)

| Constraint | Detail |
|-----------|--------|
| `Build_Solid.py` is **never modified** | Used by other projects |
| Communication with Build_Solid.py | Only via `.npy` / `.npz` disk files + subprocess |
| OCC / pythonocc | Isolated to `cnc_solid_bridge.py` subprocess calls only |
| All CNC Factory code | Lives in this directory only |

---

## File Map (current state)

```
CNC Factory/
├── PLAN.md                      ← original architecture plan
├── CONVERSATION_LOG.md          ← THIS FILE — session memory
├── factory_secrets.ini          ← project-local OpenAI key (NOT committed)
├── .gitignore                   ← protects factory_secrets.ini
│
├── config.py                    ← all constants + loads OPENAI_API_KEY from factory_secrets.ini
├── cnc_solid_bridge.py          ← ONLY file touching Build_Solid.py output
├── feature_extractor.py         ← Phase 1: edge labels + safe regions
├── feeds_speeds_engine.py       ← Phase 2: RPM, feed, power
├── toolpath_planner.py          ← Phase 3: raster paths
├── gcode_generator.py           ← Phase 4: G-code output
├── scheduler.py                 ← Phase 5: multi-machine job queue
├── cnc_agent.py                 ← Phase 6: per-machine LLM error handler
├── factory_agent.py             ← Phase 7: factory orchestrator + OpenAI LLM
│
├── disturbance_engine.py        ← full CNC disturbance taxonomy (25+ types)
├── agent_dialog.py              ← AgentConversationWindow (streaming chat UI)
├── pipeline_monitor.py          ← NEW: per-seed 8-stage pipeline visual window
├── factory_gui.py               ← main visual dashboard (tkinter, 4 machines)
├── factory_main.py              ← CLI entry point + integration test
│
└── test_*.py                    ← unit tests for each phase
```

---

## Pipeline Flow (Phases 1–7)

```
Seed number selected
      ↓
[pyocc env] Build_Solid.py  →  solid_faces_seed_N.npy
      ↓
cnc_solid_bridge.py         →  load_face_polygons(seed)
      ↓
feature_extractor.py        →  top faces, edge labels (wall/cliff/level), safe regions
      ↓
feeds_speeds_engine.py      →  RPM, feed rate, cutting force, power check
      ↓
toolpath_planner.py         →  raster path inside safe regions, entry strategy
      ↓
gcode_generator.py          →  validated G-code lines
      ↓
scheduler.py                →  job queued, assigned to idle machine
      ↓
cnc_agent.py                →  per-machine execution, error detection
      ↓
factory_agent.py            →  OpenAI prompt → 3 candidates → best → dispatch
```

---

## Key Paths (hardcoded in config.py)

```python
SOLID_OUTPUT_DIR  = "/Users/sbedi/Nextcloud/Python/Solid/random_solids/Output"
BUILD_SOLID_SCRIPT= "/Users/sbedi/Nextcloud/Python/Solid/random_solids/Build_Solid.py"
PYOCC_CONDA_ENV   = "pyocc"
SIM_SERVER        = "/Users/sbedi/Nextcloud/Python/Solid/random_solids/Simulator/cnc_sim_api_server.py"
SIM_PYTHON        = "/Users/sbedi/Nextcloud/Python/Solid/random_solids/.conda/bin/python"
SIM_PORTS         = {"M01":8001, "M02":8002, "M03":8003, "M04":8004}
```

---

## OpenAI API Key Setup (Session 2)

**Problem:** Two projects on the same machine, each with a different API key.  
**Solution:** Project-local `factory_secrets.ini` file (not `.env` — IDE blocks `.env` files).

**Priority order:**
1. `factory_secrets.ini` → `OPENAI_API_KEY=sk-...` ← **edit this file**
2. Shell env var `OPENAI_API_KEY` (fallback for CI)
3. Empty string → LLM disabled, rule-based fallback used

**How `config.py` loads it:**
```python
from dotenv import dotenv_values as _dotenv_values
_secrets = _dotenv_values("factory_secrets.ini")
OPENAI_API_KEY: str = _secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
```

**Status:** ✅ Working — `config.OPENAI_API_KEY` loads the real key automatically.  
**LLM model:** `gpt-4o-mini` (set in `config.OPENAI_MODEL`)

---

## Pipeline Monitor (Session 2 — NEW)

**File:** `pipeline_monitor.py`  
**Class:** `PipelineMonitorWindow(tk.Toplevel)`

**What it shows:**
```
🌱 Seed ▶ 🧱 Solid ▶ 📐 Features ▶ ⚙ F&S ▶ 📍 Toolpath ▶ 📄 G-Code ▶ 📅 Scheduler ▶ 🏭 Machine
```
- Each box: grey (pending) → yellow (active) → green (done) → red (error)
- Per-stage elapsed time shown inside box
- Scrolling detail log with exact numbers
- Result banner at bottom showing assigned machine

**Integration:**
- `factory_agent.submit_new_seed(seed, progress_cb=win.progress)` fires events
- `factory_gui._load_seeds_thread()` opens one window per seed in background thread
- Events posted via `GuiEvent("open_pipeline", ...)` to main thread queue

---

## factory_gui.py — Key Features

| Feature | How to use |
|---------|-----------|
| **Load Seeds** | Click `🌱 Load Seeds` in toolbar → enter seed numbers or blank for auto-4 |
| **Pipeline window** | Opens automatically per seed when loading |
| **Auto-start** | Simulation starts automatically after seeds are loaded |
| **2D toolpath canvas** | Shows per-machine bird's-eye G-code path animating in real time |
| **G-code scroll** | Live scrolling G-code with current line highlighted |
| **Tool crib table** | Per-machine tool life % with colour warnings |
| **Inject Disturbance** | `⚠ Inject Disturbance` button → tabbed dialog with 25+ disturbance types |
| **Agent Chat** | `💬 Agent Chat` button → streaming 7-phase conversation window |
| **3D Simulator** | `3D↗` button in machine header → launches cnc_sim_api_server.py subprocess |
| **OpenAI key** | `🔑 OpenAI key` button → password dialog (or auto-loaded from factory_secrets.ini) |

**CLI usage:**
```bash
python factory_gui.py                          # normal start
python factory_gui.py --auto                   # auto-start simulation
python factory_gui.py --demo-break M02 5       # inject tool break on M02 at tick 5
python factory_gui.py --openai-key sk-...      # override key from CLI
```

---

## Agent Chat — 7-Phase Conversation Flow

When a disturbance is injected on a machine, the `AgentConversationWindow` shows:

| Phase | What happens |
|-------|-------------|
| `MACHINE_REPORT` | Machine dumps its full state: job, section, tool life, queue |
| `FACTORY_ANALYSIS` | Factory builds engineering context (Kc ratio, feed/RPM recalc, power check) |
| `LLM_QUERY` | 3× OpenAI calls with streaming at temps 0.2, 0.5, 0.8 |
| `SCORING` | Each candidate scored against active policy (min_time / best_finish / etc.) |
| `NEGOTIATION` | If action=transfer, factory queries target machine for availability |
| `DECISION` | Best candidate dispatched to machine agent |
| `FOLLOWUP` | OpenAI asked for 3–5 concrete implementation steps (plain text, streamed) |
| `IMPLEMENTATION` | Confirmation shown; machine state updated |

---

## Disturbance Engine — Categories

| Category | Examples |
|----------|---------|
| **TOOL** | Breakage (safety-critical), wear >90%, wrong type, chatter, deflection |
| **MATERIAL** | Harder/softer than spec, wrong grade, oversize/undersize stock, inclusion |
| **PROCESS** | Spindle overload, coolant failure, poor finish, dimension error, power exceedance |
| **MACHINE** | Spindle bearing fault, axis fault, fixture loose (safety-critical), power fault |
| **SCHEDULE** | Priority change, delivery delay, quality hold |

**Tool breakage scenario (key demo):**
1. Click `⚠ Inject Disturbance` on any machine
2. Tab: **Tool** → select **Tool breakage**
3. Set `no_stock = True`
4. Click **⚠ Inject**
5. Factory clears inventory for that tool → sends prompt to OpenAI
6. 3 candidates streamed live → scored → best dispatched
7. Machine state updates visually

---

## Scheduling Policies

| Policy | Behaviour |
|--------|----------|
| `min_time` | Assign job to machine that finishes soonest |
| `min_cost` | Prefer cheapest machine × estimated time |
| `best_finish` | Weight toward lower feed rate for better surface quality |
| `max_tool_life` | Prefer machine with most remaining tool life |
| `multi_objective` | J = w1·Tm + w2·C + w3/L + w4·S (configurable weights in config.py) |

---

## Things Still To Do / Ideas Raised

| Item | Status | Notes |
|------|--------|-------|
| Visual comparison: seed solid vs machined part | ❌ Not built | Mentioned in Session 1 — side-by-side 3D viewer using matplotlib |
| 3D simulator connection for all 4 machines | ⚠ Partial | `[3D↗]` button exists, needs 4× cnc_sim_api_server.py instances on ports 8001-8004 |
| `generate_if_missing=True` pipeline path | ✅ Built | `submit_new_seed(..., generate_if_missing=True)` calls Build_Solid.py |
| Pipeline monitor "machine assigned" update | ⚠ Partial | Shows "queued" — actual machine assignment happens on next tick |
| Run all unit tests | ⚠ Not run this session | `python -m pytest test_*.py -v` |

---

## Session 4 — Toolpath loop fix + Completed Parts + Auto-seed

### Bugs/features
| # | Issue | Root cause | Fix |
|---|-------|-----------|-----|
| 1 | Toolpath repeats after job finishes | When cursor reached end of lines, cursor was reset to 0 but lines were NOT cleared — animation looped | Replaced per-machine cursor dict with `_anim_queues[mid]: deque` + `_anim_seen: set`. Jobs are popped from deque when animation finishes. Canvas cleared via `gcode_clear` event. |
| 2 | No completed parts record | Nothing recorded when job finished | Added `_record_completed_job()` → posts `job_complete` event. Factory panel shows treeview with Seed / Machine / Lines / Mach.s / Total.s / Load.s / Unload.s |
| 3 | Canvas reloaded every step | `load_gcode()` called every `_do_sim_step` — O(n_lines) per step | `MachinePanel` tracks `_canvas_lines_id = id(lines)`; `load_gcode()` only called when id changes (new job) |
| 4 | No auto-seed generation | Not implemented | `_auto_seed_tick()` called every step. Base prob = 0.0002% (`config.SEED_CREATION_PROB = 2e-6`). For each machine with anim queue > 10: divide prob by 10. Cycles through all available seeds, then wraps. |

### New fields added to `FactoryGUI.__init__`
```
_anim_queues: dict[str, deque]   # per-machine animation job queue
_anim_seen:   set[str]           # job IDs already queued (no duplicates)
_seed_gen_prob: float            # base = config.SEED_CREATION_PROB = 2e-6
_available_seeds: list[int]      # populated on first Load Seeds click
_loaded_seeds: set[int]          # seeds submitted this session
_completed_parts: list[dict]     # full history
```

### Auto-seed probability formula
```
overloaded = count(machines where anim_queue > 10)
eff_prob   = base_prob / 10^overloaded
```
- 0 overloaded: 2e-6 (0.0002%)
- 1 overloaded: 2e-7
- 2 overloaded: 2e-8
- All 4 overloaded: 2e-10

### Files changed in Session 4
- `config.py` — `SEED_CREATION_PROB` updated to 2e-6 (0.0002%)
- `factory_gui.py` — new `_anim_queues`, `_do_sim_step` rewrite, `_record_completed_job()`, `_auto_seed_tick()`, `FactoryPanel` completed-parts treeview, `MachinePanel.update_gcode()` canvas-id caching, `gcode_clear` event handler, `job_complete` event handler

---

## Session 3 — Bug Fixes (GUI run feedback)

### Bugs reported after first real run
| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | Seed 1000 appeared not to load | Pipeline monitor window was being opened from main thread before background thread started — timing race | Added `time.sleep(0.08)` grace period; seed 1000 loads 60 lines fine |
| 2 | Button text invisible on coloured background | `YELLOW (#d29922)` and `ACCENT (#58a6ff)` with `fg="white"` — poor contrast | Replaced all toolbar buttons with dark backgrounds `(#1a7a1a, #7a6a00, #7a1a1a, #1a3a7a)` + `fg="white"` for WCAG AA compliance |
| 3 | `🔑 OpenAI key` button unnecessary | Key auto-loads from `factory_secrets.ini` | Removed button; replaced with small `● OpenAI` / `○ LLM off` status indicator |
| 4 | Simulation starts but nothing animates | `CncAgent.tick()` calls `execute_job()` which runs the **entire job in one tick** synchronously — GUI never sees intermediate lines | Rewrote `factory_agent.tick()`: scheduler assigns jobs to `Machine` objects, then mirrors them into `agent.part_queue`. GUI's `_do_sim_step()` advances one G-code line per machine per step independently of the scheduler |
| 5 | Machine panels show job UUID not seed number | `get_state()` returned `job_id` (UUID), not `seed` | Added `current_seed` field to `get_state()`; panel now shows `Seed 42` |

### Files changed in Session 3
- `factory_gui.py` — button contrast, removed key button, LLM status indicator, new `_do_sim_step()` logic, `current_seed` display
- `factory_agent.py` — rewrote `tick()` to mirror scheduler → agent queues without calling `execute_job()`
- `cnc_agent.py` — added `current_seed` to `get_state()`

---

## Session History

### Session 1 — Architecture & All 7 Phases
- Built all 7 phases from scratch (config → feature extractor → feeds/speeds → toolpath → gcode → scheduler → cnc_agent → factory_agent)
- Built `factory_gui.py` with 4 machine panels, factory panel, 2D canvas, chat windows
- Built `disturbance_engine.py` with full 25-type taxonomy
- Built `agent_dialog.py` with streaming 7-phase conversation window
- All phases passing integration test (`python factory_main.py --test`)

### Session 2 — API Key + Pipeline Monitor
- **Problem:** Two API keys for different projects — needed project-local key storage
- **Solution:** `factory_secrets.ini` + `python-dotenv` + `config.OPENAI_API_KEY`
- **Built:** `pipeline_monitor.py` — 8-stage visual pipeline window per seed
- **Updated:** `factory_agent.submit_new_seed()` — added `progress_cb` + `generate_if_missing`
- **Updated:** `factory_gui.py` — `🌱 Load Seeds` button, pipeline window integration, auto-start
- **Verified:** Full pipeline runs on real seed, 111 G-code lines generated, LLM=True

---

## How to Resume Next Session

Tell the assistant:
> "Read CONVERSATION_LOG.md and resume from there."

Then describe what you want to work on next. Suggested next steps:
1. **Run the GUI** — `python factory_gui.py` and test the full demo flow
2. **Part comparison window** — seed solid vs machined result, side-by-side 3D
3. **4× simulator instances** — wire up cnc_sim_api_server.py on ports 8001–8004
4. **Pipeline monitor "assigned" update** — post machine assignment back to the monitor window after the scheduler tick assigns the job

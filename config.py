"""
config.py — CNC Factory global configuration.
All tuneable constants live here. No business logic.

API key loading
───────────────
The OpenAI key is read from  factory_secrets.ini  in this directory.
That file is project-local so it never collides with keys used by
other projects on the same machine.  It is listed in .gitignore and
should never be committed to version control.

Priority order (highest → lowest):
  1. factory_secrets.ini  (project-local — edit THIS file)
  2. Shell env var OPENAI_API_KEY  (fallback for CI / servers)
  3. Empty string  → LLM disabled, rule-based fallback used
"""

import os

try:
    from dotenv import dotenv_values as _dotenv_values
    _secrets = _dotenv_values(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "factory_secrets.ini")
    )
except Exception:
    _secrets = {}

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

SOLID_OUTPUT_DIR: str = os.path.join(
    "/Users/sbedi/Nextcloud/Python/Solid/random_solids", "Output"
)
BUILD_SOLID_SCRIPT: str = os.path.join(
    "/Users/sbedi/Nextcloud/Python/Solid/random_solids", "Build_Solid.py"
)
CNC_FACTORY_DIR: str = _HERE

# Conda environment that owns OpenCASCADE / pythonocc
PYOCC_CONDA_ENV: str = "pyocc"

# ══════════════════════════════════════════════════════════════════════════════
# TIMING MODEL
# ══════════════════════════════════════════════════════════════════════════════
#
# All timing in the factory uses exactly TWO time quanta:
#
#   T_TICK_S  ─ Machine update quantum (the smallest time unit).
#               One scheduler + animation step = T_TICK_S simulated seconds.
#               Valid range: 0.25 s … 10 s.  Default: 5 s.
#
#   T_G_S     ─ Graphics / dialog update interval.
#               T_G_S = T_TICK_S × T_G_MULT.
#               GUI panels and dialog boxes are refreshed every T_G_S sim-s.
#
# ─────────────────────────────────────────────────────────────────────────────
# Toolpath waypoints carry t_start and t_end (decimal seconds of sim-time).
# At each tick (current_sim_time = tick × T_TICK_S) the executor:
#   • Skips waypoints already tagged (done).
#   • Executes (tags) all waypoints whose t_end ≤ current_sim_time.
#   • If part is in unclamp stage: executes unclamp, then waits.
#   • If part is in clamp stage: changes dialog name/markers, waits one tick.
#
# ─────────────────────────────────────────────────────────────────────────────
# SHIFT CAPACITY & SEED PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────
#   shift_s            = SIM_SHIFT_HOURS × 3600
#   ticks_per_shift    = shift_s / T_TICK_S
#   avg_cycle_s        = T_AVG_MACH_S + PART_SETUP_TIME_S + PART_REMOVAL_TIME_S
#                        + PART_BUFFER_TIME_S
#   max_parts_per_mach = floor(shift_s / avg_cycle_s)
#   total_capacity     = max_parts_per_mach × SIM_N_MACHINES
#
# With SIM_IDLE_PCT percent idle time:
#   target_parts = total_capacity × (1 - SIM_IDLE_PCT/100)
#   P_seed       = target_parts / ticks_per_shift   (probability per tick)
#
# Idle targets (change SIM_IDLE_PCT):
#   25 % idle  →  base-case / stress test    (start here)
#   10 % idle  →  normal production rate
#    5 % idle  →  high-demand / max-load
# ══════════════════════════════════════════════════════════════════════════════

# ── Machine update quantum ────────────────────────────────────────────────────
T_TICK_S:    float = 5.0     # simulated seconds per tick  (range 0.25 – 10 s)
T_G_MULT:    int   = 10      # GUI/dialog update every T_G_MULT ticks
T_G_S:       float = T_TICK_S * T_G_MULT   # = 50 s  (computed; do not edit)

# ── Shift parameters ─────────────────────────────────────────────────────────
SIM_SHIFT_HOURS:   float = 8.0    # shift length (hours)
SIM_N_MACHINES:    int   = 4      # number of CNC machines

# ── Derived shift counters (do not edit) ─────────────────────────────────────
_SHIFT_S:           float = SIM_SHIFT_HOURS * 3600.0          # 28 800 s
_TICKS_PER_SHIFT:   int   = int(_SHIFT_S / T_TICK_S)          # 5 760 ticks

# ── Average machining time ───────────────────────────────────────────────────
# T_AVG_MACH_S is the mean pure-cutting time (no setup/removal).
# It is re-computed by timing_model.compute_t_avg() on first GUI start;
# this constant is the fallback / override used before that computation.
T_AVG_MACH_S:      float = 900.0   # ≈ 15 min  (fallback; recomputed at runtime)

# ── Idle-time target ─────────────────────────────────────────────────────────
# SIM_IDLE_PCT controls how busy the machines are.
#
# This directly sets the seed-arrival probability:
#   P = total_capacity × (1 - SIM_IDLE_PCT/100) / ticks_per_shift
#
# Debugging ladder (machines clearly idle between jobs → busy factory):
#   90 % idle  →  trickle mode     ← USE THIS to verify idle/restart cycle
#   75 % idle  →  light production
#   50 % idle  →  moderate load
#   25 % idle  →  base-case / stress test
#   10 % idle  →  normal production rate
#    5 % idle  →  high-demand / max-load
SIM_IDLE_PCT:      float = 10   # % of shift time machines are idle

# ── Safety cap on scheduler look-ahead ───────────────────────────────────────
# SIM_MAX_AHEAD_PER_MACHINE is a PER-MACHINE limit.
# The effective global cap scales with the number of enabled machines:
#
#   back-pressure fires when:  total_q  >= n_enabled × SIM_MAX_AHEAD_PER_MACHINE
#   overflow guard fires when: total_q  >  n_enabled × SIM_MAX_AHEAD_PER_MACHINE
#
# With 4 machines and limit=2  →  cap = 8  (2 jobs ahead per machine)
# With 1 machine  and limit=2  →  cap = 2  (2 jobs ahead for that machine)
#
# Raising this lets the CAM pipeline run further ahead so machines never
# starve; lowering it keeps memory and queue latency small.
SIM_MAX_AHEAD_PER_MACHINE: int = 1

# ── Reproducibility ──────────────────────────────────────────────────────────
# Fixed RNG seed used for ALL random decisions in the simulation:
#   - auto-seed arrivals (Bernoulli trial each tick)
#   - seed selection from the library
#   - material assignment
# Set to None to get a different random run every time.
SIM_RNG_SEED: int = 42

# ── Wall-clock speed ─────────────────────────────────────────────────────────
# GUI speed slider multiplies the wall sleep:  sleep = T_TICK_S / SIM_COMPRESSION / speed
# SIM_COMPRESSION = T_TICK_S → wall_sleep_base = 1.0 s/tick  (1 sim-tick = 1 wall-second at 1×).
# Raise SIM_COMPRESSION to run faster (e.g. _SHIFT_S / 180.0 ≈ 160× for 3-min shift).
SIM_TARGET_WALL_S:  float = 180.0                                   # 3 min wall-clock per shift at 1×
SIM_COMPRESSION:    float = T_TICK_S                                # 1 wall-second per tick at 1×

# ── GUI repaint cap ───────────────────────────────────────────────────────────
SIM_GUI_MAX_FPS:   int = 30

# ── Feature Extractor ─────────────────────────────────────────────────────────
# A face with normal.z > this is a TOP face (machined from above)
NORMAL_UP_THRESHOLD: float = 0.9

# A face with |normal.z| < this is VERTICAL (wall-like)
NORMAL_VERTICAL_THRESHOLD: float = 0.1

# ── Machine workspace & geometry units ──────────────────────────────────────
# Solid coordinates are generated in cm by Build_Solid.py.
# The rest of the pipeline (toolpath, feeds-and-speeds, G-code) works in mm.
# cnc_solid_bridge.load_face_polygons() applies GEOM_CM_TO_MM, then enforces
# the minimum footprint rule, before handing geometry downstream.
#
# Workspace: representative VMC travel (Haas VF-2 class).
MACHINE_WORKSPACE_X_MM: float = 500.0   # X travel, mm
MACHINE_WORKSPACE_Y_MM: float = 400.0   # Y travel, mm

GEOM_CM_TO_MM: float = 10.0             # cm → mm unit conversion factor

# A part whose XY footprint < MIN_FOOTPRINT_FRACTION of workspace area
# is scaled up uniformly until it meets the threshold.
MIN_FOOTPRINT_FRACTION: float = 0.10    # 10 % of workspace XY area

# Vertex matching tolerance (mm) when comparing edge endpoints across faces
EDGE_MATCH_TOLERANCE: float = 1e-2

# Z-tolerance (mm) for deciding whether a face is truly flat
FLAT_FACE_Z_TOLERANCE: float = 1e-3

# Minimum z-elevation (mm) above the current top-face height for a
# neighbouring vertical face to be classified as WALL instead of CLIFF
WALL_Z_ELEVATION_TOL: float = 1e-2

# ── Tool Defaults ─────────────────────────────────────────────────────────────
DEFAULT_TOOL_DIAMETER_MM: float = 12.0
DEFAULT_TOOL_RADIUS_MM:   float = DEFAULT_TOOL_DIAMETER_MM / 2.0
DEFAULT_TOOL_FLUTES:      int   = 4
DEFAULT_TOOL_TYPE:        str   = "flat"   # flat | bull | ball

# ── Feeds & Speeds ────────────────────────────────────────────────────────────
# material_name → (sfm_range: tuple[int,int], Kc_MPa: int)
MATERIAL_TABLE: dict = {
    "aluminium_6061":   ((800,  1200), 600),
    "aluminium_cast":   ((600,   900), 700),
    "mild_steel_1018":  ((300,   500), 1800),
    "alloy_steel_4140": ((200,   400), 2200),
    "stainless_304":    ((150,   300), 2400),
    "tool_steel_d2":    ((100,   250), 3000),
    "cast_iron":        ((400,   700), 1600),
    "brass":            ((500,  1000), 500),
    "titanium_ti64":    ((80,    200), 3000),
    "inconel":          ((50,    150), 3500),
}

DEFAULT_MATERIAL: str = "aluminium_6061"

# Chip load per tooth (mm/tooth) by tool-diameter category
CHIP_LOAD_TABLE: dict = {
    "small":  (0.025, 0.075),   # D < 12 mm
    "medium": (0.050, 0.150),   # 12 <= D < 25 mm
    "large":  (0.100, 0.300),   # D >= 25 mm
}

# Machine limits (conservative defaults)
MACHINE_POWER_LIMIT_KW:  float = 7.5
MACHINE_TORQUE_LIMIT_NM: float = 50.0

# ── Scheduler ─────────────────────────────────────────────────────────────────
TOOL_CRIB_INSPECT_INTERVAL: int  = 500    # ticks between tool-crib audits
TOOL_LIFE_WARN_PCT:         float = 20.0  # schedule replacement below this %
TOOL_LIFE_STOP_PCT:         float = 10.0  # hard stop below this %

# Multi-objective cost weights  J = w1·Tm + w2·C + w3/L + w4·S
SCHED_W_TIME:   float = 0.4
SCHED_W_COST:   float = 0.2
SCHED_W_LIFE:   float = 0.2
SCHED_W_FINISH: float = 0.2

# ── CNC Simulator API ─────────────────────────────────────────────────────────
SIM_API_HOST: str = "127.0.0.1"
SIM_API_PORT: int = 8000
SIM_API_BASE: str = f"http://{SIM_API_HOST}:{SIM_API_PORT}"

# ── CNC Agent — error handling ────────────────────────────────────────────────
# Ticks to wait for factory response before sending a reminder
ERROR_WAIT_TICKS: int = 10

# Maximum reminders before the job is moved to the factory rework queue
MAX_REMINDERS: int = 4

# ── CNC Agent — part timing (all in simulated seconds) ──────────────────────────────────
TOOL_CHANGE_TIME_S:  float = 150.0   # seconds to swap a tool (2.5 min)
# WARN_PCT: replace at next part-load boundary (half the change time)
# STOP_PCT: block new job start; finish current op then change tool

# ── Tool-wear acceleration (testing only) ───────────────────────────────────
# Each completed job deducts  (mach_s × TOOL_LIFE_DEPLETION_MULT)  from the
# tool’s remaining life instead of mach_s alone.  At 1.0 a 30-hr tool lasts
# ~115 jobs; at 20.0 it hits WARN_PCT after ~6 jobs — visible in the first
# minute of a sim run.
# Set back to 1.0 for realistic production behaviour.
TOOL_LIFE_DEPLETION_MULT: float = 20.0   # 1.0 = real-time; >1 = accelerated wear
PART_SETUP_TIME_S:   float = 120.0   # seconds to clamp and fixture a part  (2 min)
PART_REMOVAL_TIME_S: float = 120.0   # seconds to unclamp and remove a part (2 min)
PART_BUFFER_TIME_S:  float =  60.0   # contingency per cycle
MANUAL_OVERRIDE_PCT: float = 100.0   # feed-rate override (100 % = nominal)

# Alias so legacy code that imports T_UNIT_SECONDS still works
T_UNIT_SECONDS: float = T_TICK_S

# ── Material probability weights for random part assignment ───────────────────
# Each part drawn from the queue is assigned a random material.
# Weights are proportional; they do NOT have to sum to 1.
# Heavier materials have lower weights so the mix stays realistic.
MATERIAL_WEIGHTS: dict = {
    "aluminium_6061":   0.35,
    "aluminium_cast":   0.15,
    "mild_steel_1018":  0.20,
    "alloy_steel_4140": 0.10,
    "stainless_304":    0.08,
    "tool_steel_d2":    0.02,
    "cast_iron":        0.05,
    "brass":            0.03,
    "titanium_ti64":    0.015,
    "inconel":          0.005,
}

# ── Cost / amortisation ───────────────────────────────────────────────────────
MACHINE_CAPITAL_COST_USD: float  = 150_000.0  # purchase price per machine
MACHINE_AMORT_YEARS:      float  = 10.0        # amortisation period (years)
MACHINE_INTEREST_RATE:    float  = 0.0          # annual interest rate (0 = straight-line)
ANNUAL_WORKING_HOURS:     float  = 2_000.0     # working hours per year

# ── Factory Agent / OpenAI LLM ────────────────────────────────────────────────
# Model used for factory-level error diagnosis.
# The factory uses OpenAI (not the local Ollama-based cnc_llm).
OPENAI_MODEL:            str = "gpt-4o-mini"
OPENAI_MAX_TOKENS:       int = 400
FACTORY_N_LLM_CANDIDATES:int = 3     # number of candidate responses to generate
# Temperature variation across the three candidates
FACTORY_LLM_TEMPS:       tuple = (0.2, 0.5, 0.8)

# Initial factory tool-inventory (copies of each tool spec stocked at the factory)
FACTORY_TOOL_STOCK_PER_SPEC: int = 3

# ── OpenAI API key ───────────────────────────────────────────────────────────
# Loaded from factory_secrets.ini (project-local).  Falls back to the shell
# environment variable, then to an empty string (LLM disabled).
#
# To set your key:  open factory_secrets.ini and replace sk-your-key-here
#
# ── TEMPORARILY DISABLED — forces rule-based fallback ───────────────────────
# To re-enable: comment out the override line and uncomment the block below.
#OPENAI_API_KEY: str = ""   # DISABLED
OPENAI_API_KEY: str = (
    _secrets.get("OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "")
)

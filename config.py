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

# ── Seed / Timer ──────────────────────────────────────────────────────────────
# Probability that a new part is created on each GUI simulation step
# when all machine animation queues are non-empty (background rate).
# 0.002% = 0.002 / 100 = 2e-5
# When machines are idle the GUI switches to idle-demand mode (see _auto_seed_tick).
# Reduced by /10 for each machine whose queue depth exceeds 10.
SEED_CREATION_PROB: float = 0.002 / 100   # 2e-5  (×10 vs previous 2e-6)

# Duration of one scheduler tick (seconds)
T_UNIT_SECONDS: int = 1   # 1 simulated second per scheduler tick

# ── Simulation loop — three counters ──────────────────────────────────────────
# Each loop iteration advances SIM_TICK_S simulated seconds.
# Wall-clock sleep per iteration = SIM_TICK_S / SIM_COMPRESSION  (at 1x speed).
# Speed slider scales the sleep: wall_sleep = SIM_TICK_S / (SIM_COMPRESSION x speed)
#
# Three counters, all driven by the same tick:
#
#  1. SCHEDULER  -- fa.tick() called every iteration (T_UNIT_SECONDS = 1)
#     remaining_s decrements by 1 per tick; job completes after ~T_avg ticks.
#
#  2. GUI counter -- accumulates simulated seconds; resets after SIM_GUI_INTERVAL_S.
#     Dialog boxes refreshed each reset.
#
#  3. SEED counter -- accumulates simulated seconds; resets after SIM_SEED_INTERVAL_S.
#     SIM_SEED_INTERVAL_S = SHIFT_S / parts_per_shift = 28800 / 355 = 81.1 s
#     ensures exactly 355 seeds are offered per shift.
#
# Compression ratio: SHIFT_S / SIM_TARGET_WALL_S = 28800 / 180 = 160x
SIM_TICK_S:           float = 1.0     # simulated seconds advanced per loop iteration
SIM_TARGET_WALL_S:    float = 180.0   # wall-clock seconds for one full shift at 1x
SIM_COMPRESSION:      float = 160.0   # = SHIFT_S / SIM_TARGET_WALL_S
SIM_GUI_INTERVAL_S:   float = 5.0     # refresh GUI every 5 simulated seconds
# SIM_SEED_INTERVAL_S = CYCLE_S / N_MACHINES
# where CYCLE_S = T_avg + PART_SETUP_TIME_S + PART_REMOVAL_TIME_S + PART_BUFFER_TIME_S
#               = 324.5 + 60 + 60 + 60 = 504.5 s
# parts/shift   = N x SHIFT_S / CYCLE_S = 4 x 28800 / 504.5 = 228
# interval      = CYCLE_S / N           = 504.5 / 4          = 126.1 sim-s
SIM_SEED_INTERVAL_S:  float = 126.1   # sim-s between seed offers  (= CYCLE_S / N)
SIM_GUI_MAX_FPS:      int   = 30      # hard cap on GUI repaint rate (wall clock)

# ── Seed generation ───────────────────────────────────────────────────
# Every SIM_SEED_INTERVAL_S simulated seconds the loop offers one seed.
# SIM_SEED_INTERVAL_S = SHIFT_S / parts_per_shift = 28800 / 355 = 81.1 s
# Safety cap: skip if scheduler queue >= SIM_MAX_QUEUE_AHEAD jobs.
SIM_T_AVG_S:          float = 324.5   # measured mean machining time (s), cm→mm
SIM_SHIFT_HOURS:      float = 8.0     # shift length (hours)
SIM_N_MACHINES:       int   = 4       # number of CNC machines
SIM_MAX_QUEUE_AHEAD:  int   = 8       # safety cap on scheduler queue depth

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

# ── CNC Agent — part timing ───────────────────────────────────────────────────
TOOL_CHANGE_TIME_S:  float = 120.0   # seconds to swap a tool
PART_SETUP_TIME_S:   float = 60.0   # seconds to clamp and fixture a part (1 min)
PART_REMOVAL_TIME_S: float = 60.0   # seconds to unclamp and remove a part  (1 min)
PART_BUFFER_TIME_S:  float = 60.0   # contingency per cycle: tool change, inspection etc.
MANUAL_OVERRIDE_PCT: float = 100.0   # feed-rate override (100 % = nominal)

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
OPENAI_API_KEY: str = (
    _secrets.get("OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "")
)

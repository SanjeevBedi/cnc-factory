"""
cnc_agent.py — Phase 6 of the CNC Factory pipeline.

Responsibilities (per the factory specification)
-------------------------------------------------
1.  Model one physical CNC machine as a software agent.
        - Maintains its own Tool Crib  (T1 … T10, per-machine spec).
        - Maintains its own Part Queue (FIFO).
        - Tracks section-level progress through the G-code program.

2.  Select the best tool for each feature (face, pocket, hole) based on
    the active factory policy.
        - Pocketing : largest tool whose diameter < pocket_width / 4.
        - Step-over  : ≤ tool_radius / 3  (reduced 10 % for max_tool_life).
        - Depth of cut: governed by tool-radius × shank-length table from spec.

3.  Execute G-code programs.
        - Dry-run (default for testing) : parse lines, track sections, tally
          air cuts vs material cuts without hitting the simulator API.
        - Connected mode: delegate to CncSimClient (cnc_llm_client.py).

4.  Compute machining cost.
        Toolpath time = Σ(move_length / feed) × override + tool_change + setup + removal
        Machine cost  = hourly_rate × time   (rate from straight-line amortisation)
        Tool cost     = consumable_cost × (cutting_time / max_life_hrs)

5.  Track tool wear and enforce life thresholds.
        - remaining_life_hrs decremented after each use.
        - Tool flagged for replacement when remaining_life_hrs < 10 % of max.
        - Actual change happens at next part changeover; change adds TOOL_CHANGE_TIME_S.

6.  Implement the factory error protocol.
        inject_error(description) → status = "awaiting_factory"
        Each tick()  increments error_wait_count.
        At multiples of ERROR_WAIT_TICKS: send a reminder (up to MAX_REMINDERS).
        After MAX_REMINDERS reminders: move part to rework queue, resume machining.

7.  Expose get_state() → JSON-serialisable dict for the Factory Agent (Phase 7).

Simulator note
--------------
Four independent CncAgent instances represent four machines.
Each can be pointed at a different simulator port.  Whether four full
simulator processes can run simultaneously is a separate question;
dry_run=True lets the agent operate without a live simulator.

Sources
-------
Autonomous_Factory_Sim.docx (tool tables, error protocol, cost model, tool selection)
cnc_llm_client.py            (CncSimClient, prepare_program)
CNC_SIM_COMMANDS.md          (G-code command set)
config.py                    (all tuneable constants)
"""

from __future__ import annotations

import re
import sys
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import config
from scheduler import Job

# optional simulator client — not a hard dependency
try:
    sys.path.insert(0, "/Users/sbedi/Nextcloud/Python/Solid/LLM")
    from cnc_llm_client import CncSimClient, prepare_program
    _SIM_AVAILABLE = True
except ImportError:
    _SIM_AVAILABLE = False
    CncSimClient = None


# ── Tool record ───────────────────────────────────────────────────────────────

@dataclass
class ToolRecord:
    """One physical tool in the machine's tool crib."""
    tool_id:             str      # "T1" … "T10"
    n_inserts:           int
    diameter_mm:         float
    shank_length_mm:     float
    max_life_hrs:        float
    remaining_life_hrs:  float
    holder_cost_usd:     float
    consumable_cost_usd: float
    requires_coolant:    bool = True

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    @property
    def remaining_life_frac(self) -> float:
        return self.remaining_life_hrs / self.max_life_hrs if self.max_life_hrs > 0 else 0.0

    @property
    def remaining_life_pct(self) -> float:
        return self.remaining_life_frac * 100.0

    @property
    def needs_replacement(self) -> bool:
        return self.remaining_life_pct < config.TOOL_LIFE_STOP_PCT

    def deduct_life(self, cutting_time_s: float) -> None:
        """Reduce remaining life by the cutting time used."""
        hours_used = cutting_time_s / 3_600.0
        self.remaining_life_hrs = max(0.0, self.remaining_life_hrs - hours_used)

    def reset_life(self) -> None:
        """Replace consumable — reset life to max."""
        self.remaining_life_hrs = self.max_life_hrs

    def consumable_cost_for_time(self, cutting_time_s: float) -> float:
        """Pro-rated consumable cost for a given cutting time."""
        if self.max_life_hrs <= 0:
            return 0.0
        frac = (cutting_time_s / 3_600.0) / self.max_life_hrs
        return self.consumable_cost_usd * frac


# ── Tool crib ─────────────────────────────────────────────────────────────────

class ToolCrib:
    """Manages all tools loaded into one machine."""

    def __init__(self, machine_id: str, tools: list[ToolRecord]) -> None:
        self.machine_id = machine_id
        self._tools: dict[str, ToolRecord] = {t.tool_id: t for t in tools}

    @property
    def tools(self) -> list[ToolRecord]:
        return list(self._tools.values())

    def get(self, tool_id: str) -> Optional[ToolRecord]:
        return self._tools.get(tool_id)

    def add_tool(self, tool: ToolRecord) -> None:
        """Factory agent loads a new tool at changeover."""
        self._tools[tool.tool_id] = tool

    def remove_tool(self, tool_id: str) -> Optional[ToolRecord]:
        return self._tools.pop(tool_id, None)

    def select_for_face(self, face_width_mm: float) -> Optional[ToolRecord]:
        """
        Largest available tool whose diameter < face_width_mm / 4.
        Used for face milling.
        """
        candidates = [
            t for t in self._tools.values()
            if not t.needs_replacement and t.diameter_mm < face_width_mm / 4.0
        ]
        if not candidates:
            # Relax constraint: just largest usable tool
            candidates = [t for t in self._tools.values() if not t.needs_replacement]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.diameter_mm)

    def select_for_pocket(self, pocket_width_mm: float) -> Optional[ToolRecord]:
        """
        Pocketing: largest tool with diameter < pocket_width / 4.
        """
        return self.select_for_face(pocket_width_mm)

    def select_for_hole(self, hole_diameter_mm: float) -> Optional[ToolRecord]:
        """
        Drilling: tool whose diameter ≤ hole_diameter.
        Prefer largest that fits.
        """
        candidates = [
            t for t in self._tools.values()
            if not t.needs_replacement and t.diameter_mm <= hole_diameter_mm
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.diameter_mm)

    def tools_needing_replacement(self) -> list[ToolRecord]:
        return [t for t in self._tools.values()
                if t.remaining_life_pct < config.TOOL_LIFE_WARN_PCT]

    def update_tool_from_factory(
        self, tool_id: str, new_tool: ToolRecord
    ) -> tuple[Optional[ToolRecord], ToolRecord]:
        """
        Factory agent authorised swap: remove old, install new.
        Returns (removed, installed).
        """
        removed = self._tools.pop(tool_id, None)
        self._tools[new_tool.tool_id] = new_tool
        return removed, new_tool

    def state_list(self) -> list[dict]:
        return [
            {
                "tool_id":            t.tool_id,
                "diameter_mm":        t.diameter_mm,
                "shank_length_mm":    t.shank_length_mm,
                "remaining_life_hrs": round(t.remaining_life_hrs, 3),
                "remaining_life_pct": round(t.remaining_life_pct, 1),
                "needs_replacement":  t.needs_replacement,
            }
            for t in self._tools.values()
        ]


# ── Error event ───────────────────────────────────────────────────────────────

@dataclass
class ErrorEvent:
    """Recorded when an error condition occurs on the machine."""
    event_id:         str
    tick:             int
    job_id:           str
    section:          str
    gcode_line:       str
    description:      str
    reminders_sent:   int  = 0
    resolved:         bool = False
    factory_response: str  = ""
    wait_count:       int  = 0   # ticks since error started


# ── Execution result ──────────────────────────────────────────────────────────

@dataclass
class AgentExecResult:
    """Summary of one job execution."""
    job_id:               str
    machine_id:           str
    lines_executed:       int
    sections_completed:   list = field(default_factory=list)
    tool_changes:         list = field(default_factory=list)   # tool_ids swapped
    volume_removed_mm3:   float = 0.0
    air_cut_lines:        int   = 0
    material_cut_lines:   int   = 0
    machining_time_s:     float = 0.0
    machine_cost_usd:     float = 0.0
    tool_cost_usd:        float = 0.0
    total_cost_usd:       float = 0.0
    errors:               list  = field(default_factory=list)
    completed:            bool  = False


# ── CNC Agent ─────────────────────────────────────────────────────────────────

class CncAgent:
    """
    One virtual CNC machine modelled as an autonomous agent.

    Parameters
    ----------
    machine_id   : label, e.g. "M01"
    tool_crib    : ToolCrib pre-loaded with this machine's tools
    rework_queue : shared factory rework queue (passed in, not owned)
    sim_base_url : base URL of the CNC simulator API.  None = dry_run.
    dry_run      : if True, execute G-code without calling the simulator API
    """

    def __init__(
        self,
        machine_id:   str,
        tool_crib:    ToolCrib,
        rework_queue: Optional[deque] = None,
        sim_base_url: Optional[str]   = None,
        dry_run:      bool             = True,
    ) -> None:
        self.machine_id    = machine_id
        self.tool_crib     = tool_crib
        self.rework_queue  = rework_queue if rework_queue is not None else deque()

        # Part queue — FIFO
        self.part_queue:   deque[Job] = deque()
        self.current_job:  Optional[Job] = None

        # Status
        self.status: str = "idle"
        # idle | running | paused | awaiting_factory | error | stopped

        # Error protocol
        self.active_error:   Optional[ErrorEvent] = None
        self._error_history: list[ErrorEvent] = []
        self.current_tick:   int = 0

        # Progress
        self.current_section:     str        = ""
        self.completed_sections:  list[str]  = []
        self.gcode_line_idx:      int        = 0

        # Simulator
        self.dry_run    = dry_run or (not _SIM_AVAILABLE) or (sim_base_url is None)
        self.sim_client = None
        if not self.dry_run and _SIM_AVAILABLE and sim_base_url:
            self.sim_client = CncSimClient(base_url=sim_base_url)

        # Cost — straight-line machine amortisation
        yearly = config.MACHINE_CAPITAL_COST_USD / config.MACHINE_AMORT_YEARS
        self.machine_hourly_rate_usd: float = yearly / config.ANNUAL_WORKING_HOURS

        # Pending tool change (set when a tool is worn out mid-job)
        self._pending_tool_change: bool = False

    # ── Queue management ──────────────────────────────────────────────────────

    def enqueue(self, job: Job) -> None:
        """Add a job to the back of this machine's part queue."""
        self.part_queue.append(job)

    def _dequeue_next(self) -> Optional[Job]:
        if not self.part_queue:
            return None
        return self.part_queue.popleft()

    # ── Tool selection ────────────────────────────────────────────────────────

    def select_tool(
        self,
        feature_type:    str,     # 'face' | 'pocket' | 'hole'
        feature_width_mm: float,
        policy:          str = "min_time",
    ) -> Optional[ToolRecord]:
        """
        Choose the best tool for a feature under the active policy.

        Policy adjustments
        ------------------
        max_tool_life: prefer tool with most remaining life among candidates.
        min_cost:      prefer tool with lowest consumable cost rate.
        best_finish:   prefer smallest tool (finer step-over).
        min_time:      prefer largest tool (fewest passes).
        """
        if feature_type == "pocket":
            candidates = [
                t for t in self.tool_crib.tools
                if not t.needs_replacement
                and t.diameter_mm < feature_width_mm / 4.0
            ]
        elif feature_type == "hole":
            candidates = [
                t for t in self.tool_crib.tools
                if not t.needs_replacement
                and t.diameter_mm <= feature_width_mm
            ]
        else:  # face
            candidates = [
                t for t in self.tool_crib.tools
                if not t.needs_replacement
                and t.diameter_mm < feature_width_mm / 4.0
            ]

        if not candidates:
            candidates = [t for t in self.tool_crib.tools if not t.needs_replacement]
        if not candidates:
            return None

        if policy == "max_tool_life":
            return max(candidates, key=lambda t: t.remaining_life_hrs)
        if policy == "min_cost":
            return min(candidates, key=lambda t: t.consumable_cost_usd)
        if policy == "best_finish":
            return min(candidates, key=lambda t: t.diameter_mm)
        # min_time / default: largest diameter → fewest passes
        return max(candidates, key=lambda t: t.diameter_mm)

    # ── Depth-of-cut model ────────────────────────────────────────────────────

    @staticmethod
    def depth_of_cut_mm(tool: ToolRecord, policy: str = "min_time") -> float:
        """
        Return recommended axial depth of cut (mm) based on tool geometry.

        Rules (from spec):
            R > 25 mm   → 3 mm if shank < 75;   2 mm if 75–150;   1 mm if 150–250
            15 ≤ R ≤ 25 → 3 mm if shank ≤ 50;   2 mm if 50–100;   1 mm if 100–175
            10 < R < 15 → 3 mm if shank < 25;   2 mm if 25–50;    1 mm if 50–100
            R ≤ 10 mm   → 1 mm (conservative default)

        max_tool_life policy: reduce by 10 %.
        """
        R = tool.radius_mm
        S = tool.shank_length_mm

        if R > 25:
            doc = 3.0 if S < 75 else (2.0 if S <= 150 else 1.0)
        elif R >= 15:
            doc = 3.0 if S <= 50 else (2.0 if S <= 100 else 1.0)
        elif R > 10:
            doc = 3.0 if S < 25 else (2.0 if S <= 50 else 1.0)
        else:
            doc = 1.0

        if policy == "max_tool_life":
            doc *= 0.90
        return round(doc, 3)

    @staticmethod
    def step_over_mm(tool: ToolRecord, policy: str = "min_time") -> float:
        """Step-over = tool_radius / 3; reduced 10 % for max_tool_life."""
        so = tool.radius_mm / 3.0
        if policy == "max_tool_life":
            so *= 0.90
        return round(so, 3)

    # ── Cost model ────────────────────────────────────────────────────────────

    def compute_machining_time_s(self, job: Job) -> float:
        """
        Toolpath time (seconds):
            Σ(line length / feed) × override  +  tool_change_time  +  setup  +  removal
        Uses job.estimated_time_s as the cutting component (computed in Phase 3).
        """
        cutting  = job.estimated_time_s * (config.MANUAL_OVERRIDE_PCT / 100.0)
        overhead = (config.PART_SETUP_TIME_S +
                    config.PART_REMOVAL_TIME_S +
                    (config.TOOL_CHANGE_TIME_S if self._pending_tool_change else 0.0))
        return cutting + overhead

    def compute_cost(self, job: Job, tool: Optional[ToolRecord] = None) -> dict:
        """
        Return cost breakdown dict.
        machine_cost = hourly_rate × total_time
        tool_cost    = consumable rate × cutting_time
        """
        total_time_s  = self.compute_machining_time_s(job)
        machine_cost  = self.machine_hourly_rate_usd * (total_time_s / 3_600.0)
        tool_cost     = 0.0
        if tool is not None:
            tool_cost = tool.consumable_cost_for_time(job.estimated_time_s)
        return {
            "total_time_s":   round(total_time_s, 1),
            "machine_cost":   round(machine_cost, 4),
            "tool_cost":      round(tool_cost, 4),
            "total_cost":     round(machine_cost + tool_cost, 4),
        }

    # ── Error protocol ────────────────────────────────────────────────────────

    def inject_error(
        self,
        description: str,
        gcode_line:  str = "",
    ) -> ErrorEvent:
        """
        Signal an error condition. Sets status to "awaiting_factory".
        The caller (or factory agent) must eventually call handle_factory_response().
        """
        evt = ErrorEvent(
            event_id   = str(uuid.uuid4()),
            tick       = self.current_tick,
            job_id     = self.current_job.job_id if self.current_job else "",
            section    = self.current_section,
            gcode_line = gcode_line,
            description= description,
        )
        self.active_error = evt
        self._error_history.append(evt)
        self.status = "awaiting_factory"
        return evt

    def handle_factory_response(self, response: dict) -> None:
        """
        Factory agent sends a response to the active error.
        Possible actions: 'continue' | 'reduce_feed' | 'abort' | 'rework'
        """
        if self.active_error is None:
            return
        self.active_error.factory_response = response.get("action", "continue")
        self.active_error.resolved         = True
        action = response.get("action", "continue")

        if action == "abort":
            self.status = "stopped"
        elif action == "rework":
            self._move_current_to_rework()
            self.active_error = None
            self.status = "idle"
        elif action == "reduce_feed":
            # Apply feed reduction (stored; used next time compute_machining_time_s called)
            pct = float(response.get("feed_override_pct", 80.0))
            # Re-scale estimate (simplified: flag for next job selection)
            if self.current_job is not None:
                self.current_job.estimated_time_s *= (100.0 / pct)
            self.active_error = None
            self.status = "running"
        else:  # continue
            self.active_error = None
            self.status = "running"

    def _tick_error_protocol(self) -> list[str]:
        """
        Called each tick when status == "awaiting_factory".
        Returns list of action strings taken this tick.
        """
        events = []
        if self.active_error is None:
            self.status = "idle"
            return events

        self.active_error.wait_count += 1

        if self.active_error.wait_count % config.ERROR_WAIT_TICKS == 0:
            self.active_error.reminders_sent += 1
            events.append(
                f"reminder_{self.active_error.reminders_sent}_sent"
            )
            if self.active_error.reminders_sent >= config.MAX_REMINDERS:
                # No factory response after MAX_REMINDERS — move to rework
                self._move_current_to_rework()
                events.append("moved_to_rework")
                self.active_error.resolved = True
                self.active_error = None
                self.status = "idle"
        return events

    def _move_current_to_rework(self) -> None:
        if self.current_job is not None:
            self.current_job.status = "rework"
            self.rework_queue.append(self.current_job)
            self.current_job = None

    # ── G-code execution ──────────────────────────────────────────────────────

    _COMMENT_RE  = re.compile(r"^\((.+)\)$")
    _SECTION_RE  = re.compile(r"^\((Face|Pocket|Hole|approach|entry|retract|link)")
    _G00_RE      = re.compile(r"^G00\b")
    _G01_RE      = re.compile(r"^G01\b")
    _M3_RE       = re.compile(r"\bM3\b")
    _M5_RE       = re.compile(r"^M5$")
    _M30_RE      = re.compile(r"^M30$")

    def execute_job(self, job: Job, policy: str = "min_time") -> AgentExecResult:
        """
        Execute all G-code lines for a job.

        In dry_run mode: parse lines, track sections, tally air/material cuts
        without calling the simulator API.
        In connected mode: delegate each line to CncSimClient.step().
        """
        self.current_job    = job
        self.gcode_line_idx = 0
        self.current_section= "Initialisation"
        self.completed_sections = []
        self.status = "running"

        # Tool selection (use first available if no feature hint in job)
        tool = self.select_tool("face", 200.0, policy)

        result = AgentExecResult(
            job_id     = job.job_id,
            machine_id = self.machine_id,
            lines_executed = 0,
        )

        # Handle any pending tool change before starting
        if self._pending_tool_change and tool is not None:
            result.tool_changes.append(tool.tool_id)
            result.machining_time_s += config.TOOL_CHANGE_TIME_S
            self._pending_tool_change = False

        spindle_on = False

        for idx, line in enumerate(job.gcode_lines):
            self.gcode_line_idx = idx
            clean = line.strip()
            if not clean:
                continue

            # Section tracking
            m = self._SECTION_RE.match(clean)
            if m:
                if self.current_section:
                    self.completed_sections.append(self.current_section)
                self.current_section = clean.lstrip("(").rstrip(")")
                result.sections_completed = list(self.completed_sections)

            # Spindle state
            if self._M3_RE.search(clean):
                spindle_on = True
            if self._M5_RE.match(clean):
                spindle_on = False

            # Motion lines
            is_rapid = bool(self._G00_RE.match(clean))
            is_feed  = bool(self._G01_RE.match(clean))

            if is_rapid:
                result.air_cut_lines += 1
            elif is_feed:
                if spindle_on:
                    result.material_cut_lines += 1
                else:
                    result.air_cut_lines += 1

            if not self.dry_run and self.sim_client is not None:
                try:
                    step = self.sim_client.step(clean, apply_cut=True)
                    if "MATERIAL_CUT" in step.message:
                        vol = step.state.get("volume", {}).get("last_cut", 0.0)
                        result.volume_removed_mm3 += vol
                except Exception as exc:
                    self.inject_error(str(exc), gcode_line=clean)
                    break

            result.lines_executed += 1

            if self._M30_RE.match(clean):
                break

        # Finalise section
        if self.current_section:
            self.completed_sections.append(self.current_section)
        result.sections_completed = list(self.completed_sections)

        # Machining time and cost
        result.machining_time_s = self.compute_machining_time_s(job)
        cost_info = self.compute_cost(job, tool)
        result.machine_cost_usd = cost_info["machine_cost"]
        result.tool_cost_usd    = cost_info["tool_cost"]
        result.total_cost_usd   = cost_info["total_cost"]

        # Deduct tool life
        if tool is not None:
            tool.deduct_life(job.estimated_time_s)
            if tool.needs_replacement:
                self._pending_tool_change = True

        result.completed = True
        job.status = "done"
        self.current_job = None
        self.status = "idle"
        return result

    # ── Scheduler-integration tick ────────────────────────────────────────────

    def tick(self, tick_num: int) -> dict:
        """
        Called by the factory / scheduler each clock tick.
        - If awaiting_factory: runs error protocol.
        - If idle and queue non-empty: starts next job.
        Returns a state snapshot (JSON-serialisable).
        """
        self.current_tick = tick_num
        events = []

        if self.status == "awaiting_factory":
            events.extend(self._tick_error_protocol())

        if self.status == "idle" and self.part_queue:
            job = self._dequeue_next()
            result = self.execute_job(job)
            events.append(f"completed_job_{job.job_id[:8]}")

        state = self.get_state()
        state["tick"]   = tick_num
        state["events"] = events
        return state

    # ── State reporting ───────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """JSON-serialisable snapshot for the Factory Agent."""
        # current_job: prefer actively-running job, fall back to head of queue
        active_job = self.current_job or (self.part_queue[0] if self.part_queue else None)
        return {
            "machine_id":   self.machine_id,
            "status":       self.status,
            "current_job":  active_job.job_id if active_job else None,
            "current_seed": active_job.seed   if active_job else None,
            "queue_depth":  len(self.part_queue),
            "rework_depth": len(self.rework_queue),
            "tool_crib":    self.tool_crib.state_list(),
            "active_error": {
                "event_id":       self.active_error.event_id,
                "description":    self.active_error.description,
                "reminders_sent": self.active_error.reminders_sent,
                "wait_count":     self.active_error.wait_count,
            } if self.active_error else None,
            "current_section":    self.current_section,
            "completed_sections": list(self.completed_sections),
            "pending_tool_change": self._pending_tool_change,
        }


# ── Default tool cribs (from spec) ────────────────────────────────────────────

def _std_tools_t5_t10() -> list[dict]:
    """T5–T10 shared across all four machines."""
    return [
        {"tool_id":"T5",  "n_inserts":2, "diameter_mm":3.0,  "shank_length_mm":50.0,
         "max_life_hrs":30.0, "remaining_life_hrs":25.0,
         "holder_cost_usd":200.0, "consumable_cost_usd":80.0},
        {"tool_id":"T6",  "n_inserts":2, "diameter_mm":5.0,  "shank_length_mm":75.0,
         "max_life_hrs":30.0, "remaining_life_hrs":20.0,
         "holder_cost_usd":500.0, "consumable_cost_usd":20.0},
        {"tool_id":"T7",  "n_inserts":2, "diameter_mm":10.0, "shank_length_mm":75.0,
         "max_life_hrs":30.0, "remaining_life_hrs":20.0,
         "holder_cost_usd":500.0, "consumable_cost_usd":20.0},
        {"tool_id":"T8",  "n_inserts":2, "diameter_mm":15.0, "shank_length_mm":75.0,
         "max_life_hrs":30.0, "remaining_life_hrs":20.0,
         "holder_cost_usd":500.0, "consumable_cost_usd":20.0},
        {"tool_id":"T9",  "n_inserts":2, "diameter_mm":20.0, "shank_length_mm":75.0,
         "max_life_hrs":30.0, "remaining_life_hrs":20.0,
         "holder_cost_usd":500.0, "consumable_cost_usd":20.0},
        {"tool_id":"T10", "n_inserts":2, "diameter_mm":25.0, "shank_length_mm":75.0,
         "max_life_hrs":30.0, "remaining_life_hrs":20.0,
         "holder_cost_usd":500.0, "consumable_cost_usd":20.0},
    ]


_MACHINE_TOOL_DATA: dict[str, list[dict]] = {
    "M01": [
        {"tool_id":"T1","n_inserts":8,"diameter_mm":100.0,"shank_length_mm":75.0,
         "max_life_hrs":30.0,"remaining_life_hrs":25.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
        {"tool_id":"T2","n_inserts":6,"diameter_mm":50.0, "shank_length_mm":75.0,
         "max_life_hrs":30.0,"remaining_life_hrs":20.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
        {"tool_id":"T3","n_inserts":4,"diameter_mm":25.0, "shank_length_mm":75.0,
         "max_life_hrs":30.0,"remaining_life_hrs":25.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
        {"tool_id":"T4","n_inserts":4,"diameter_mm":10.0, "shank_length_mm":100.0,
         "max_life_hrs":30.0,"remaining_life_hrs":10.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
    ] + _std_tools_t5_t10(),
    "M02": [
        {"tool_id":"T1","n_inserts":8,"diameter_mm":75.0, "shank_length_mm":100.0,
         "max_life_hrs":30.0,"remaining_life_hrs":25.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
        {"tool_id":"T2","n_inserts":4,"diameter_mm":25.0, "shank_length_mm":75.0,
         "max_life_hrs":30.0,"remaining_life_hrs":20.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
        {"tool_id":"T3","n_inserts":4,"diameter_mm":20.0, "shank_length_mm":150.0,
         "max_life_hrs":30.0,"remaining_life_hrs":25.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
        {"tool_id":"T4","n_inserts":2,"diameter_mm":8.0,  "shank_length_mm":100.0,
         "max_life_hrs":30.0,"remaining_life_hrs":10.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
    ] + _std_tools_t5_t10(),
    "M03": [
        {"tool_id":"T1","n_inserts":8,"diameter_mm":50.0, "shank_length_mm":125.0,
         "max_life_hrs":30.0,"remaining_life_hrs":25.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
        {"tool_id":"T2","n_inserts":6,"diameter_mm":30.0, "shank_length_mm":125.0,
         "max_life_hrs":30.0,"remaining_life_hrs":20.0,"holder_cost_usd":700.0,"consumable_cost_usd":20.0},
        {"tool_id":"T3","n_inserts":4,"diameter_mm":15.0, "shank_length_mm":150.0,
         "max_life_hrs":30.0,"remaining_life_hrs":25.0,"holder_cost_usd":400.0,"consumable_cost_usd":20.0},
        {"tool_id":"T4","n_inserts":2,"diameter_mm":6.0,  "shank_length_mm":100.0,
         "max_life_hrs":30.0,"remaining_life_hrs":10.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
    ] + _std_tools_t5_t10(),
    "M04": [
        {"tool_id":"T1","n_inserts":6,"diameter_mm":65.0, "shank_length_mm":125.0,
         "max_life_hrs":30.0,"remaining_life_hrs":25.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
        {"tool_id":"T2","n_inserts":6,"diameter_mm":40.0, "shank_length_mm":150.0,
         "max_life_hrs":30.0,"remaining_life_hrs":20.0,"holder_cost_usd":900.0,"consumable_cost_usd":20.0},
        {"tool_id":"T3","n_inserts":4,"diameter_mm":10.0, "shank_length_mm":75.0,
         "max_life_hrs":30.0,"remaining_life_hrs":25.0,"holder_cost_usd":400.0,"consumable_cost_usd":20.0},
        {"tool_id":"T4","n_inserts":2,"diameter_mm":5.0,  "shank_length_mm":50.0,
         "max_life_hrs":30.0,"remaining_life_hrs":10.0,"holder_cost_usd":500.0,"consumable_cost_usd":20.0},
    ] + _std_tools_t5_t10(),
}


def build_default_crib(machine_id: str) -> ToolCrib:
    """
    Create the factory-default ToolCrib for a machine (M01 … M04).
    Unknown machine_id returns an empty crib.
    """
    raw = _MACHINE_TOOL_DATA.get(machine_id, [])
    tools = [
        ToolRecord(
            tool_id             = d["tool_id"],
            n_inserts           = d["n_inserts"],
            diameter_mm         = d["diameter_mm"],
            shank_length_mm     = d["shank_length_mm"],
            max_life_hrs        = d["max_life_hrs"],
            remaining_life_hrs  = d["remaining_life_hrs"],
            holder_cost_usd     = d["holder_cost_usd"],
            consumable_cost_usd = d["consumable_cost_usd"],
            requires_coolant    = True,
        )
        for d in raw
    ]
    return ToolCrib(machine_id=machine_id, tools=tools)


def build_factory(n_machines: int = 4, dry_run: bool = True) -> list[CncAgent]:
    """
    Instantiate n_machines CncAgent objects sharing one rework queue.
    Returns list of agents M01, M02, … M0n.
    """
    rework: deque = deque()
    agents = []
    for i in range(1, n_machines + 1):
        mid   = f"M{i:02d}"
        crib  = build_default_crib(mid)
        agent = CncAgent(
            machine_id   = mid,
            tool_crib    = crib,
            rework_queue = rework,
            dry_run      = dry_run,
        )
        agents.append(agent)
    return agents

"""
scheduler.py — Phase 5 of the CNC Factory pipeline.

Responsibilities
----------------
1.  Maintain a fleet of virtual CNC machines, each with:
        - status  (idle | running | paused | stopped | awaiting_tool_change)
        - tool life fraction  (1.0 = new … 0.0 = exhausted)
        - current job and time remaining

2.  Manage a job queue (FIFO by default, ordered by policy score).

3.  Advance a discrete global clock:
        t_real = tick × T_UNIT_SECONDS           (enhanced_plan.tex §5)
    Each tick:
        a. Progress running jobs (subtract T_UNIT_SECONDS from remaining time).
        b. Deplete tool life proportionally to cutting time.
        c. Check life thresholds: warn < 20%, hard-stop < 10%.
        d. Complete finished jobs and mark machines idle.
        e. Assign queued jobs to idle machines using the active policy.
        f. Every TOOL_CRIB_INSPECT_INTERVAL ticks: run a full tool-crib audit.

4.  Scheduling policies (unified_plan.tex §6):
        min_time          — assign job to the machine finishing soonest
        max_tool_life     — prefer machine with most remaining life
        min_cost          — prefer cheapest machine × estimated time
        best_finish       — lower feed rate weight → smoother result
        multi_objective   — J = w1·Tm + w2·C + w3·(1/L) + w4·S   (enhanced_plan.tex §3)

5.  Provide a serialisable state dict for consumption by Phase 6 CNC Agents.

Sources
-------
enhanced_plan.tex  §3 (cost function), §5 (timer), §6 (tool crib)
unified_plan.tex   §6 (scheduler policies)
config.py          TOOL_CRIB_INSPECT_INTERVAL, T_UNIT_SECONDS, SCHED_W_*
"""

from __future__ import annotations

import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import config


# ── Constants ─────────────────────────────────────────────────────────────────
VALID_POLICIES = frozenset(
    {"min_time", "max_tool_life", "min_cost", "best_finish", "multi_objective"}
)

# Default total cutting life for a tool (seconds).
# Real value depends on material, coating, etc.
DEFAULT_TOOL_LIFE_S: float = 3_600.0   # 1 hour of cutting


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Job:
    """A single machining job derived from one GCodeResult."""
    job_id:             str
    seed:               object
    estimated_time_s:   float        # from GCodeResult / ToolpathResult
    material:           str          # for finish-quality weighting
    feed_rate_mmpm:     float        # used by best_finish policy
    cost_per_s:         float        # machine cost rate ($/s)
    priority:           float = 1.0  # higher = more urgent
    gcode_lines:        list  = field(default_factory=list)  # full program
    # Stamped toolpath — set by factory_agent.build_job_from_seed() after
    # timing_model.stamp_toolpath() runs.  Each Waypoint inside carries
    # t_start / t_end (decimal sim-seconds from job start) and a _done flag.
    # The GUI tick-executor calls timing_model.waypoints_due() each tick to
    # find which waypoints have elapsed and advances the G-code cursor to match.
    # None until the CAM pipeline completes.
    toolpath_result:    object = None   # ToolpathResult | None

    # Tool actually selected from the machine’s crib for this job.
    # Set by factory_agent.build_job_from_seed(); used by the GUI to
    # deplete the correct ToolRecord when the job completes.
    tool_id_used:       str   = ""     # e.g. "T2"
    tool_diameter_used: float = 0.0    # mm — for log / KPI display

    # Runtime fields (set by Scheduler)
    status:             str   = "queued"   # queued | running | done | failed
    assigned_machine_id: Optional[str] = None
    created_at_tick:    int   = 0
    started_at_tick:    Optional[int] = None
    finished_at_tick:   Optional[int] = None

    @property
    def wall_time_s(self) -> Optional[float]:
        """Actual elapsed ticks × T_UNIT if job is done, else None."""
        if self.started_at_tick is None or self.finished_at_tick is None:
            return None
        return (self.finished_at_tick - self.started_at_tick) * config.T_UNIT_SECONDS


@dataclass
class Machine:
    """One virtual CNC machine."""
    machine_id:          str
    cost_per_hour:       float = 100.0    # $/hour (used by min_cost policy)
    total_tool_life_s:   float = DEFAULT_TOOL_LIFE_S

    # Runtime state
    status:              str   = "idle"
    tool_life_fraction:  float = 1.0      # 1.0 = new, 0.0 = dead
    time_accumulated_s:  float = 0.0      # total cutting time this tool has seen
    current_job:         Optional[Job]  = None
    remaining_s:         float = 0.0     # seconds left on current job

    @property
    def cost_per_s(self) -> float:
        return self.cost_per_hour / 3_600.0

    @property
    def tool_life_pct(self) -> float:
        return self.tool_life_fraction * 100.0

    def is_idle(self) -> bool:
        return self.status == "idle"

    def is_running(self) -> bool:
        return self.status == "running"


@dataclass
class ToolCribEvent:
    tick:         int
    machine_id:   str
    event_type:   str   # 'warn' | 'stop' | 'replace'
    life_pct:     float
    message:      str


@dataclass
class TickResult:
    """Summary of one scheduler tick."""
    tick:               int
    t_real_s:           float
    jobs_started:       list   = field(default_factory=list)  # job_ids
    jobs_finished:      list   = field(default_factory=list)  # job_ids
    machines_stopped:   list   = field(default_factory=list)  # machine_ids (tool exhausted)
    tool_crib_events:   list   = field(default_factory=list)  # ToolCribEvent
    new_jobs_created:   int    = 0
    warnings:           list   = field(default_factory=list)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class Scheduler:
    """
    Discrete-time factory scheduler.

    Parameters
    ----------
    n_machines        : number of virtual CNC machines
    policy            : scheduling policy name
    total_tool_life_s : tool life before replacement (seconds of cutting)
    cost_per_hour     : machine hourly rate ($/hr) — used by min_cost policy
    seed_creation_prob: probability per tick of a new part arriving
                        (0 = no auto-creation; submit jobs manually)
    new_job_factory   : callable(seed) → Job | None
                        called when a new seed is created; if None, no auto-jobs
    rng_seed          : optional random seed for reproducibility
    """

    def __init__(
        self,
        n_machines:         int            = 3,
        policy:             str            = "multi_objective",
        total_tool_life_s:  float          = DEFAULT_TOOL_LIFE_S,
        cost_per_hour:      float          = 100.0,
        seed_creation_prob: float          = 0.0,
        new_job_factory:    Optional[Callable] = None,
        rng_seed:           Optional[int]  = None,
    ) -> None:
        if policy not in VALID_POLICIES:
            raise ValueError(f"Unknown policy {policy!r}. "
                             f"Choose from {sorted(VALID_POLICIES)}")

        self.policy             = policy
        self.seed_creation_prob = seed_creation_prob
        self.new_job_factory    = new_job_factory
        self._rng               = random.Random(rng_seed)

        # Global clock
        self.tick:    int   = 0
        self.t_real_s: float = 0.0

        # Machines
        self.machines: list[Machine] = [
            Machine(
                machine_id      = f"M{i+1:02d}",
                cost_per_hour   = cost_per_hour,
                total_tool_life_s = total_tool_life_s,
            )
            for i in range(n_machines)
        ]

        # Queues and logs
        self.job_queue:       deque[Job] = deque()
        self.completed_jobs:  list[Job]  = []
        self.failed_jobs:     list[Job]  = []
        self.tool_crib_log:   list[ToolCribEvent] = []
        self._next_seed:      int = 1000   # auto-increment for generated seeds

    # ── Public API ─────────────────────────────────────────────────────────────

    def submit(self, job: Job) -> None:
        """Add a job to the back of the queue."""
        job.status           = "queued"
        job.created_at_tick  = self.tick
        self.job_queue.append(job)

    def tick_once(self) -> TickResult:
        """
        Advance the simulation by one time step (T_UNIT_SECONDS seconds).
        Returns a TickResult summarising all events this tick.
        """
        self.tick    += 1
        self.t_real_s = self.tick * config.T_TICK_S

        result = TickResult(tick=self.tick, t_real_s=self.t_real_s)

        # 1. Progress running machines
        for m in self.machines:
            if not m.is_running():
                continue

            # Deplete tool life
            dt = config.T_TICK_S
            m.remaining_s       -= dt
            m.time_accumulated_s += dt
            m.tool_life_fraction  = max(
                0.0,
                1.0 - m.time_accumulated_s / m.total_tool_life_s
            )

            # ── Tool life thresholds ──────────────────────────────────────
            if m.tool_life_fraction * 100.0 < config.TOOL_LIFE_STOP_PCT:
                # Hard stop
                self._stop_machine(m, result)
                continue

            if m.tool_life_fraction * 100.0 < config.TOOL_LIFE_WARN_PCT:
                evt = ToolCribEvent(
                    tick       = self.tick,
                    machine_id = m.machine_id,
                    event_type = "warn",
                    life_pct   = m.tool_life_pct,
                    message    = (
                        f"{m.machine_id} tool life {m.tool_life_pct:.1f}% — "
                        f"schedule replacement"
                    ),
                )
                result.tool_crib_events.append(evt)
                self.tool_crib_log.append(evt)

            # ── Job completion check ──────────────────────────────────────
            if m.remaining_s <= 0.0:
                self._complete_job(m, result)

        # 2. Tool-crib audit (every TOOL_CRIB_INSPECT_INTERVAL ticks)
        if self.tick % config.TOOL_CRIB_INSPECT_INTERVAL == 0:
            self._tool_crib_audit(result)

        # 3. Probabilistic new-job creation
        if (self.seed_creation_prob > 0.0
                and self._rng.random() < self.seed_creation_prob
                and self.new_job_factory is not None):
            seed = self._next_seed
            self._next_seed += 1
            new_job = self.new_job_factory(seed)
            if new_job is not None:
                self.submit(new_job)
                result.new_jobs_created += 1

        # 4. Assign queued jobs to idle machines
        self._assign_jobs(result)

        return result

    def run(self, n_ticks: int) -> list[TickResult]:
        """Run the scheduler for n_ticks steps. Return all TickResults."""
        return [self.tick_once() for _ in range(n_ticks)]

    def replace_tool(self, machine_id: str) -> bool:
        """
        Replace the tool on a machine (reset life to 1.0, status → idle).
        Returns True if successful, False if machine_id not found.
        """
        for m in self.machines:
            if m.machine_id == machine_id:
                m.tool_life_fraction  = 1.0
                m.time_accumulated_s  = 0.0
                if m.status == "awaiting_tool_change":
                    m.status = "idle"
                evt = ToolCribEvent(
                    tick       = self.tick,
                    machine_id = machine_id,
                    event_type = "replace",
                    life_pct   = 100.0,
                    message    = f"{machine_id} tool replaced — life reset to 100%",
                )
                self.tool_crib_log.append(evt)
                return True
        return False

    # ── Internals ──────────────────────────────────────────────────────────────

    def _complete_job(self, m: Machine, result: TickResult) -> None:
        job = m.current_job
        if job is None:
            return
        job.status           = "done"
        job.finished_at_tick = self.tick
        self.completed_jobs.append(job)
        result.jobs_finished.append(job.job_id)
        m.current_job = None
        m.remaining_s = 0.0
        m.status      = "idle"

    def _stop_machine(self, m: Machine, result: TickResult) -> None:
        """Hard-stop a machine due to critical tool life."""
        m.status = "awaiting_tool_change"
        evt = ToolCribEvent(
            tick       = self.tick,
            machine_id = m.machine_id,
            event_type = "stop",
            life_pct   = m.tool_life_pct,
            message    = (
                f"{m.machine_id} STOPPED — tool life {m.tool_life_pct:.1f}% "
                f"< {config.TOOL_LIFE_STOP_PCT}% threshold"
            ),
        )
        result.tool_crib_events.append(evt)
        result.machines_stopped.append(m.machine_id)
        self.tool_crib_log.append(evt)

        # Return current job to front of queue
        if m.current_job is not None:
            m.current_job.status = "queued"
            m.current_job.assigned_machine_id = None
            m.current_job.started_at_tick     = None
            self.job_queue.appendleft(m.current_job)
            m.current_job = None

    def _tool_crib_audit(self, result: TickResult) -> None:
        """Periodic full-fleet audit (every TOOL_CRIB_INSPECT_INTERVAL ticks).
        Always emits an [AUDIT] event regardless of per-tick warns already raised.
        """
        for m in self.machines:
            if m.tool_life_pct < config.TOOL_LIFE_WARN_PCT:
                evt = ToolCribEvent(
                    tick       = self.tick,
                    machine_id = m.machine_id,
                    event_type = "warn",
                    life_pct   = m.tool_life_pct,
                    message    = (
                        f"[AUDIT] {m.machine_id} tool life "
                        f"{m.tool_life_pct:.1f}% — replace at next changeover"
                    ),
                )
                result.tool_crib_events.append(evt)
                self.tool_crib_log.append(evt)

    def _assign_jobs(self, result: TickResult) -> None:
        """Assign queued jobs to idle machines using the active policy."""
        idle = [m for m in self.machines if m.is_idle()]
        if not idle or not self.job_queue:
            return

        for m in idle:
            if not self.job_queue:
                break
            job = self._pick_job(m)
            if job is None:
                continue
            self._start_job(m, job, result)

    def _pick_job(self, machine: Machine) -> Optional[Job]:
        """
        Choose the best job for a given machine from the queue.
        Pops and returns the chosen job, or None if queue is empty.
        """
        if not self.job_queue:
            return None

        if self.policy in ("min_time", "best_finish", "min_cost"):
            # These policies don't consider which machine is better —
            # just take the highest-priority job off the front.
            return self.job_queue.popleft()

        if self.policy == "max_tool_life":
            # Assign to machine with most life — but we already chose that machine.
            # Just take the highest-priority job.
            return self.job_queue.popleft()

        # multi_objective: score each job for this machine, pick best
        if self.policy == "multi_objective":
            best_score = float("inf")
            best_idx   = 0
            for i, job in enumerate(self.job_queue):
                score = self._objective_score(job, machine)
                if score < best_score:
                    best_score = best_idx = i  # type: ignore[assignment]
                    best_score = score
                    best_idx   = i
            # Remove from deque
            jobs_list = list(self.job_queue)
            chosen    = jobs_list.pop(best_idx)
            self.job_queue = deque(jobs_list)
            return chosen

        # Fallback
        return self.job_queue.popleft()

    def _objective_score(self, job: Job, machine: Machine) -> float:
        """
        Compute J = w1·Tm̂ + w2·Ĉ + w3·(1/L̂) + w4·Ŝ
        All terms normalised to [0,1] using heuristic scales.
        Lower is better.
        """
        # Normalised machining time (reference: 1 hour = 3600s)
        Tm = min(job.estimated_time_s / 3_600.0, 1.0)

        # Normalised cost  (reference: $10 for a 1-hour job at $10/hr)
        cost = machine.cost_per_s * job.estimated_time_s
        C    = min(cost / 10.0, 1.0)

        # Tool life risk: 1/L where L is the fraction remaining
        L    = max(machine.tool_life_fraction, 0.01)   # avoid div/0
        risk = min(1.0 / L, 10.0) / 10.0              # normalise to ~[0,1]

        # Surface-finish penalty: proportional to feed rate (higher feed → worse finish)
        S    = min(job.feed_rate_mmpm / 5_000.0, 1.0)

        J = (config.SCHED_W_TIME   * Tm +
             config.SCHED_W_COST   * C  +
             config.SCHED_W_LIFE   * risk +
             config.SCHED_W_FINISH * S)
        return J

    def _start_job(self, m: Machine, job: Job, result: TickResult) -> None:
        job.status              = "running"
        job.assigned_machine_id = m.machine_id
        job.started_at_tick     = self.tick
        m.current_job           = job
        m.remaining_s           = job.estimated_time_s
        m.status                = "running"
        result.jobs_started.append(job.job_id)

    # ── State / reporting ─────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """
        Serialisable snapshot of the scheduler for Phase 6 CNC Agents.
        All fields are plain Python types (str, int, float, list, dict).
        """
        return {
            "tick":    self.tick,
            "t_real_s": self.t_real_s,
            "policy":  self.policy,
            "machines": [
                {
                    "machine_id":         m.machine_id,
                    "status":             m.status,
                    "tool_life_pct":      round(m.tool_life_pct, 2),
                    "time_accumulated_s": round(m.time_accumulated_s, 1),
                    "remaining_s":        round(m.remaining_s, 1),
                    "current_job_id":     m.current_job.job_id
                                          if m.current_job else None,
                }
                for m in self.machines
            ],
            "queue_depth":    len(self.job_queue),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs":    len(self.failed_jobs),
            "tool_crib_log":  [
                {
                    "tick":       e.tick,
                    "machine_id": e.machine_id,
                    "event_type": e.event_type,
                    "life_pct":   round(e.life_pct, 2),
                    "message":    e.message,
                }
                for e in self.tool_crib_log[-20:]   # last 20 events
            ],
        }

    def summary(self) -> str:
        lines = [
            f"Scheduler — tick {self.tick}  "
            f"(t_real = {self.t_real_s:.0f} s)  "
            f"policy = {self.policy}",
            f"  Queue     : {len(self.job_queue)} pending",
            f"  Completed : {len(self.completed_jobs)} jobs",
            f"  Failed    : {len(self.failed_jobs)} jobs",
        ]
        for m in self.machines:
            job_info = (f"job {m.current_job.job_id[:8]}… "
                        f"({m.remaining_s:.0f}s left)"
                        if m.current_job else "—")
            lines.append(
                f"  {m.machine_id}  {m.status:<22s}  "
                f"life {m.tool_life_pct:5.1f}%  {job_info}"
            )
        return "\n".join(lines)


# ── Job factory helpers ────────────────────────────────────────────────────────

def make_job(
    seed:             object,
    estimated_time_s: float,
    material:         str   = config.DEFAULT_MATERIAL,
    feed_rate_mmpm:   float = 3_000.0,
    cost_per_hour:    float = 100.0,
    gcode_lines:      list  = None,
    priority:         float = 1.0,
) -> Job:
    """Convenience factory — create a Job without a full pipeline run."""
    return Job(
        job_id           = str(uuid.uuid4()),
        seed             = seed,
        estimated_time_s = estimated_time_s,
        material         = material,
        feed_rate_mmpm   = feed_rate_mmpm,
        cost_per_s       = cost_per_hour / 3_600.0,
        priority         = priority,
        gcode_lines      = gcode_lines or [],
    )


def job_from_gcode(
    gc,           # GCodeResult
    cost_per_hour: float = 100.0,
    priority:      float = 1.0,
) -> Job:
    """Create a Job directly from a Phase 4 GCodeResult."""
    from feeds_speeds_engine import FeedsSpeedsResult
    return Job(
        job_id           = str(uuid.uuid4()),
        seed             = gc.seed,
        estimated_time_s = gc.estimated_time_s,
        material         = "unknown",
        feed_rate_mmpm   = 3_000.0,
        cost_per_s       = cost_per_hour / 3_600.0,
        priority         = priority,
        gcode_lines      = gc.lines,
    )

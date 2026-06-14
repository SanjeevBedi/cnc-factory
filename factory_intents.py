"""
factory_intents.py — Factory-level intent catalogue, state ledger, and
                     AI advisory integration.

Mirrors the structure of agent_intents.py but operates at the factory level —
across all machines, the scheduler, the tool inventory, and completed-part
records.  The factory agent has its own intent/state/policy triplet:

  Intent  — what questions can the factory answer and what actions can it take
  State   — the FactoryLedger, which is the single persistent record of:
               • every job: queued → allocated → machining → done / rework
               • every tool event: warn → change → breakage
               • every machine idle period (start tick, end tick, machine)
               • every scheduling policy in effect and when it changed
  Policy  — the active scheduling policy and how it influences decisions

Key additions required by the user
───────────────────────────────────
  Queries
  -------
  factory_active_machines     — which machines are running right now
  factory_machining_parts     — parts currently being machined (per machine)
  factory_machined_parts      — parts that have completed machining (history)
  factory_queue               — parts currently in the scheduler queue
  factory_schedule_history    — full schedule allocation log
  factory_tool_usage_history  — aggregate tool usage across all machines
  factory_tool_usage_on_machine — tool usage log for a specific machine
  factory_idle_time           — idle time per machine (current shift / all time)
  factory_status              — fleet-level KPI snapshot

  Queries with parameters
  -----------------------
  factory_estimate_time       — estimated machining time for a seed under a policy
  factory_policy_comparison   — machining time for a seed under ALL policies
  factory_tool_usage_on_machine(machine_id) — tool events for one machine

  Actions
  -------
  factory_remove_from_queue   — remove a job by seed or job_id from the queue
  factory_add_to_queue        — build and enqueue a seed
  factory_replace_tool        — replace a specific tool on a specific machine
  factory_set_policy          — change the active scheduling policy
  factory_ai_advice           — compile full factory context → OpenAI → advice

  Implementation path (how AI advice gets applied)
  -------------------------------------------------
  OpenAI receives:
    1. FactoryIntentCatalogue (what the factory can do)
    2. MachineIntentCatalogue (what each machine can do)
    3. FactoryLedger snapshot (full history, idle times, tool events)
    4. Active policy and KPIs
    5. User question / optimisation goal

  OpenAI responds with:
    { "recommendation": str,
      "actions": [{"target": "factory"|"M01"|...,
                   "intent": <intent_id>,
                   "parameters": {...},
                   "reasoning": str}, ...],
      "policy_suggestion": str | None,
      "reasoning": str }

  The FactoryActionExecutor._do_factory_ai_advice() method:
    1. Calls _compile_factory_context() to build the prompt.
    2. Calls OpenAI (or rule-based fallback).
    3. Parses the response.
    4. For each action in response["actions"]:
         • If target == "factory" → calls FactoryActionExecutor locally.
         • If target == machine_id → routes through agent.handle_factory_response()
           using the same protocol as the existing error pipeline.
    5. Returns a chat summary of what was done.

Architecture note
─────────────────
This module is imported by:
  - factory_gui.py  (builds FactoryConversationWindow context)
  - agent_dialog.py (FactoryConversationWindow._send routing)

It does NOT import factory_gui.py or agent_dialog.py (no circular imports).
It imports factory_agent.py types only for type hints under TYPE_CHECKING.
"""

from __future__ import annotations

import re as _re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import config

if TYPE_CHECKING:
    from factory_agent import FactoryAgent
    from cnc_agent import CncAgent


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — FACTORY INTENT CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FactoryIntent:
    intent_id:       str
    category:        str   # query | action | advisory
    label:           str
    description:     str
    parameters:      tuple = ()
    requires_stop:   bool  = False
    reversible:      bool  = True
    example:         str   = ""


FACTORY_INTENT_CATALOGUE: list[FactoryIntent] = [

    # ── QUERY — fleet / machine status ────────────────────────────────────────
    FactoryIntent(
        "factory_status", "query",
        "Factory fleet status",
        "Return KPI snapshot: machines running/idle/stopped, jobs completed, "
        "rework depth, avg tool life, queue depth.",
        (),
        example='{"intent":"factory_status"}',
    ),
    FactoryIntent(
        "factory_active_machines", "query",
        "Active machines",
        "List all machines that are currently running a job, with job ID, "
        "seed, material, and elapsed time.",
        (),
        example='{"intent":"factory_active_machines"}',
    ),
    FactoryIntent(
        "factory_machining_parts", "query",
        "Parts currently being machined",
        "List all parts currently on a machine: seed, machine, material, "
        "progress %, estimated time remaining.",
        (),
        example='{"intent":"factory_machining_parts"}',
    ),
    FactoryIntent(
        "factory_machined_parts", "query",
        "Parts that have been machined",
        "Return the completed-parts ledger: seed, machine, material, start/end "
        "tick, actual time, cost, any errors.",
        ("n",),
        example='{"intent":"factory_machined_parts","n":10}',
    ),
    FactoryIntent(
        "factory_queue", "query",
        "Parts in the scheduler queue",
        "List all jobs waiting in the queue: position, seed, material, "
        "estimated time, priority.",
        (),
        example='{"intent":"factory_queue"}',
    ),
    FactoryIntent(
        "factory_schedule_history", "query",
        "Part schedule / allocation history",
        "Return the allocation log: which machine each seed was assigned to, "
        "when, and the policy in effect at the time.",
        ("n",),
        example='{"intent":"factory_schedule_history","n":20}',
    ),
    FactoryIntent(
        "factory_idle_time", "query",
        "Idle time per machine",
        "Return idle time summary for each machine: total idle ticks, "
        "idle percentage this shift, longest idle streak.",
        ("machine_id",),
        example='{"intent":"factory_idle_time","machine_id":"M02"}',
    ),
    FactoryIntent(
        "factory_tool_usage_history", "query",
        "Tool usage history — all machines",
        "Aggregate tool event log across all machines: changes, breakages, "
        "warn events, replacements — most recent N events.",
        ("n",),
        example='{"intent":"factory_tool_usage_history","n":20}',
    ),
    FactoryIntent(
        "factory_tool_usage_on_machine", "query",
        "Tool usage history — specific machine",
        "Tool event log filtered to one machine: which tools were used, "
        "when changed, breakage events, remaining life per tool.",
        ("machine_id",),
        example='{"intent":"factory_tool_usage_on_machine","machine_id":"M01"}',
    ),

    # ── QUERY — performance estimation ────────────────────────────────────────
    FactoryIntent(
        "factory_estimate_time", "query",
        "Estimate machining time for a seed under a policy",
        "Run a lightweight estimate (no full CAM) of how long a given seed "
        "would take on each machine under the specified policy.",
        ("seed", "policy"),
        example='{"intent":"factory_estimate_time","seed":1042,"policy":"min_time"}',
    ),
    FactoryIntent(
        "factory_policy_comparison", "query",
        "Policy comparison for a seed",
        "Return a table of estimated machining times and costs for a given seed "
        "under ALL five scheduling policies.",
        ("seed",),
        example='{"intent":"factory_policy_comparison","seed":1042}',
    ),

    # ── ACTION — queue management ──────────────────────────────────────────────
    FactoryIntent(
        "factory_remove_from_queue", "action",
        "Remove a part from the queue",
        "Remove a job from the scheduler queue by seed number or job_id. "
        "Does not affect jobs already on a machine.",
        ("seed", "job_id"),
        reversible=False,
        example='{"intent":"factory_remove_from_queue","seed":1042}',
    ),
    FactoryIntent(
        "factory_add_to_queue", "action",
        "Add a part to the queue",
        "Build a new job from a seed number via the full CAM pipeline and "
        "submit it to the scheduler queue.",
        ("seed", "priority"),
        example='{"intent":"factory_add_to_queue","seed":1043,"priority":1.0}',
    ),
    FactoryIntent(
        "factory_replace_tool", "action",
        "Replace a tool on a specific machine",
        "Force an immediate tool replacement on the named machine — "
        "equivalent to manually swapping the tool at the machine.",
        ("machine_id", "tool_id"),
        requires_stop=True,
        example='{"intent":"factory_replace_tool","machine_id":"M02","tool_id":"T3"}',
    ),
    FactoryIntent(
        "factory_set_policy", "action",
        "Change the scheduling policy",
        "Switch the active scheduling policy for all future job assignments. "
        "Does not affect jobs already on a machine.",
        ("policy",),
        example='{"intent":"factory_set_policy","policy":"best_finish"}',
    ),

    # ── ADVISORY — AI analysis ─────────────────────────────────────────────────
    FactoryIntent(
        "factory_ai_advice", "advisory",
        "AI factory performance analysis",
        "Compile full factory context (ledger, KPIs, machine intents, tool state) "
        "and send to OpenAI for analysis and actionable recommendations. "
        "The AI response specifies which intents to execute on which machines; "
        "confirmed actions are dispatched automatically.",
        ("question",),
        example='{"intent":"factory_ai_advice",'
                '"question":"how do I reduce idle time on M03?"}',
    ),
]

FACTORY_INTENT_MAP: dict[str, FactoryIntent] = {
    i.intent_id: i for i in FACTORY_INTENT_CATALOGUE
}


# ── Capability block for AI prompt ────────────────────────────────────────────

def build_factory_context_block() -> str:
    """
    Return a full capability description of the factory agent for use
    in the AI advisory system prompt.  Mirrors build_system_context() from
    agent_intents.py but describes factory-level intents.
    """
    lines = [
        "=== FACTORY AGENT CAPABILITIES ===",
        "",
        "You are the factory AI advisor.  The factory agent can execute the",
        "following intents across the fleet of CNC machines.",
        "Your response MUST use 'intent' from this list.  Do NOT invent names.",
        "",
    ]
    cat_label = {
        "query":    "QUERY — read fleet state and production records",
        "action":   "ACTION — modify queue, tools, and policy",
        "advisory": "ADVISORY — AI analysis and implementation",
    }
    for cat in ("query", "action", "advisory"):
        intents = [i for i in FACTORY_INTENT_CATALOGUE if i.category == cat]
        if not intents:
            continue
        lines.append(f"── {cat_label[cat]} ──")
        for intent in intents:
            flags = []
            if intent.requires_stop: flags.append("requires machine stop")
            if not intent.reversible: flags.append("irreversible")
            fs = f"  [{', '.join(flags)}]" if flags else ""
            lines.append(f"  {intent.intent_id:<40s} {intent.label}{fs}")
            lines.append(f"    {intent.description}")
            if intent.parameters:
                lines.append(f"    Parameters : {', '.join(intent.parameters)}")
            if intent.example:
                lines.append(f"    Example    : {intent.example}")
            lines.append("")
    lines += [
        "── AI RESPONSE FORMAT ──",
        "  {",
        '   "recommendation" : "<summary of what you advise>",',
        '   "actions"        : [',
        '      {"target"    : "factory" | "M01" | "M02" | "M03" | "M04",',
        '       "intent"    : "<intent_id from catalogue above>",',
        '       "parameters": { <key>:<value>, ... },',
        '       "reasoning" : "<one concise sentence>"}',
        "    ],",
        '   "policy_suggestion" : "<policy_name>" | null,',
        '   "reasoning"         : "<overall reasoning in 2-3 sentences>"',
        "  }",
        "",
        "=== END FACTORY CAPABILITIES ===",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — FACTORY LEDGER (persistent event record)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PartRecord:
    """Full lifecycle record of one machined part."""
    job_id:           str
    seed:             object
    machine_id:       str           = ""
    material:         str           = ""
    policy_at_alloc:  str           = ""   # scheduling policy when allocated
    status:           str           = "queued"
    # Timing
    queued_at_tick:   int           = 0
    allocated_at_tick: Optional[int] = None
    started_at_tick:  Optional[int] = None
    finished_at_tick: Optional[int] = None
    # Performance
    estimated_time_s: float         = 0.0
    actual_time_s:    float         = 0.0
    cost_usd:         float         = 0.0
    gcode_line_count: int           = 0
    # Quality
    error_count:      int           = 0
    rework:           bool          = False
    # Tools used (list of tool_id strings)
    tools_used:       list          = field(default_factory=list)


@dataclass
class ToolEvent:
    """One tool event on one machine."""
    event_id:    str
    tick:        int
    machine_id:  str
    tool_id:     str
    event_type:  str   # "warn" | "stop" | "replace" | "breakage" | "install"
    life_pct:    float = 0.0
    diameter_mm: float = 0.0
    message:     str   = ""
    substitute:  bool  = False   # True if a nearest-dia substitute was used


@dataclass
class IdleEvent:
    """One idle period on one machine."""
    machine_id:   str
    start_tick:   int
    end_tick:     int   = -1     # -1 = still idle
    reason:       str   = ""     # "no_jobs" | "tool_change" | "error" | "stopped"

    def duration_ticks(self, current_tick: int = 0) -> int:
        """Closed event: exact duration.  Open (end_tick==-1): use current_tick."""
        if self.end_tick >= 0:
            return max(0, self.end_tick - self.start_tick)
        return max(0, current_tick - self.start_tick) if current_tick > 0 else 0


@dataclass
class PolicyChange:
    """Record of when the factory policy was changed."""
    tick:       int
    old_policy: str
    new_policy: str
    reason:     str = ""


class FactoryLedger:
    """
    Single persistent record of all factory events.

    Kept as a member of FactoryAgent — all writes happen via the methods below.
    Reads (queries) are performed by FactoryIntentExecutor.

    Thread-safety note: the GUI simulation loop and the CAM background
    threads both write to the ledger.  The writes use simple list.append()
    which is GIL-protected in CPython and therefore safe in practice.
    For production use a threading.Lock would be appropriate.
    """

    def __init__(self) -> None:
        # Part lifecycle records (job_id → PartRecord)
        self.parts:           dict[str, PartRecord]  = {}
        # Tool event log (append-only)
        self.tool_events:     list[ToolEvent]        = []
        # Idle event log
        self.idle_events:     list[IdleEvent]        = []
        # Active idle events (machine_id → IdleEvent that is still open)
        self._open_idle:      dict[str, IdleEvent]   = {}
        # Policy changes
        self.policy_changes:  list[PolicyChange]     = []
        # Allocation log: [(tick, job_id, machine_id, policy)]
        self.allocation_log:  list[tuple]            = []

    # ── Part lifecycle ─────────────────────────────────────────────────────────

    def record_queued(self, job_id: str, seed, estimated_time_s: float,
                      material: str, tick: int) -> PartRecord:
        rec = PartRecord(
            job_id           = job_id,
            seed             = seed,
            material         = material,
            status           = "queued",
            queued_at_tick   = tick,
            estimated_time_s = estimated_time_s,
        )
        self.parts[job_id] = rec
        return rec

    def record_allocated(self, job_id: str, machine_id: str,
                         policy: str, tick: int) -> None:
        rec = self.parts.get(job_id)
        if rec:
            rec.machine_id        = machine_id
            rec.policy_at_alloc   = policy
            rec.allocated_at_tick = tick
            rec.status            = "allocated"
        self.allocation_log.append((tick, job_id, machine_id, policy))

    def record_started(self, job_id: str, tick: int) -> None:
        rec = self.parts.get(job_id)
        if rec:
            rec.started_at_tick = tick
            rec.status          = "machining"

    def record_finished(self, job_id: str, tick: int,
                        actual_time_s: float = 0.0,
                        cost_usd: float = 0.0,
                        rework: bool = False,
                        error_count: int = 0) -> None:
        rec = self.parts.get(job_id)
        if rec:
            rec.finished_at_tick = tick
            rec.actual_time_s    = actual_time_s
            rec.cost_usd         = cost_usd
            rec.rework           = rework
            rec.error_count      = error_count
            rec.status           = "rework" if rework else "done"

    # ── Tool events ────────────────────────────────────────────────────────────

    def record_tool_event(self, tick: int, machine_id: str, tool_id: str,
                          event_type: str, life_pct: float = 0.0,
                          diameter_mm: float = 0.0, message: str = "",
                          substitute: bool = False) -> None:
        self.tool_events.append(ToolEvent(
            event_id   = str(uuid.uuid4()),
            tick       = tick,
            machine_id = machine_id,
            tool_id    = tool_id,
            event_type = event_type,
            life_pct   = life_pct,
            diameter_mm = diameter_mm,
            message    = message,
            substitute = substitute,
        ))

    # ── Idle tracking ──────────────────────────────────────────────────────────

    def machine_went_idle(self, machine_id: str, tick: int, reason: str = "no_jobs") -> None:
        if machine_id not in self._open_idle:
            ev = IdleEvent(machine_id=machine_id, start_tick=tick, reason=reason)
            self._open_idle[machine_id] = ev
            self.idle_events.append(ev)

    def machine_became_active(self, machine_id: str, tick: int) -> None:
        ev = self._open_idle.pop(machine_id, None)
        if ev is not None:
            ev.end_tick = tick

    # ── Policy changes ─────────────────────────────────────────────────────────

    def record_policy_change(self, old_policy: str, new_policy: str,
                             tick: int, reason: str = "") -> None:
        self.policy_changes.append(PolicyChange(
            tick=tick, old_policy=old_policy, new_policy=new_policy, reason=reason,
        ))

    # ── Summary helpers ────────────────────────────────────────────────────────

    def idle_summary(self, machine_id: Optional[str] = None,
                     current_tick: int = 0) -> dict:
        """
        Return total idle ticks and longest streak per machine.
        Pass current_tick so open (still-idle) periods are measured
        correctly — without it, open events contribute 0 ticks.
        """
        out: dict[str, dict] = defaultdict(lambda: {
            "total_idle_ticks": 0, "longest_streak": 0, "n_idle_periods": 0,
            "currently_idle": False, "idle_since_tick": -1,
        })
        for ev in self.idle_events:
            if machine_id and ev.machine_id != machine_id:
                continue
            dur = ev.duration_ticks(current_tick)
            d   = out[ev.machine_id]
            d["total_idle_ticks"] += dur
            d["longest_streak"]    = max(d["longest_streak"], dur)
            d["n_idle_periods"]   += 1
            if ev.end_tick < 0:   # still open
                d["currently_idle"]   = True
                d["idle_since_tick"]  = ev.start_tick
        return dict(out)

    def tool_events_for_machine(self, machine_id: str) -> list[ToolEvent]:
        return [e for e in self.tool_events if e.machine_id == machine_id]

    def completed_parts(self, n: Optional[int] = None) -> list[PartRecord]:
        done = [r for r in self.parts.values() if r.status in ("done", "rework")]
        done.sort(key=lambda r: r.finished_at_tick or 0, reverse=True)
        return done[:n] if n else done

    def queued_parts(self) -> list[PartRecord]:
        return [r for r in self.parts.values() if r.status == "queued"]

    def machining_parts(self) -> list[PartRecord]:
        return [r for r in self.parts.values() if r.status == "machining"]


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — MESSAGE CLASSIFICATION (mirrors agent_intents.py approach)
# ══════════════════════════════════════════════════════════════════════════════

_MACHINE_ID_RE = _re.compile(r'\b(M0[1-4])\b', _re.IGNORECASE)
_SEED_RE        = _re.compile(r'\bseed\s*(\d+)\b', _re.IGNORECASE)
_TOOL_ID_RE     = _re.compile(r'\bT(\d{1,2})\b',  _re.IGNORECASE)
_N_RE           = _re.compile(r'\blast\s*(\d+)\b|\b(\d+)\s+(?:jobs?|parts?|events?)\b',
                               _re.IGNORECASE)

# Policy aliases
_VALID_POLICIES = {
    "min_time":        "min_time",
    "minimum time":    "min_time",
    "fastest":         "min_time",
    "min_cost":        "min_cost",
    "minimum cost":    "min_cost",
    "cheapest":        "min_cost",
    "best_finish":     "best_finish",
    "best finish":     "best_finish",
    "finish quality":  "best_finish",
    "max_tool_life":   "max_tool_life",
    "max tool life":   "max_tool_life",
    "tool life":       "max_tool_life",
    "multi_objective": "multi_objective",
    "multi objective": "multi_objective",
    "balanced":        "multi_objective",
}

# Factory-level action signals (things that indicate an ACTION, not a query)
_FACTORY_ACTION_SIGNALS = [
    "remove", "delete", "cancel", "drop", "pull",      # queue removal
    "add", "submit", "enqueue", "queue up",             # queue addition
    "replace tool", "change tool", "swap tool",         # tool replacement
    "set policy", "change policy", "switch policy",     # policy change
    "advise", "analyse", "analyze", "optimise",         # AI advisory
    "optimize", "improve", "recommendation",
    "why is", "how can", "how do",                      # advisory questions
    "what should", "suggest", "tell me how",
]

# Keyword table for factory queries (ordered — most specific first)
_FACTORY_QUERY_KEYWORDS: dict[str, list[str]] = {
    "factory_policy_comparison":  [
        "compare policies", "all policies", "policy comparison",
        "which policy is best", "best policy for",
        "compare all policies", "policies for seed",
        "list policies", "show all policies",
    ],
    "factory_estimate_time":  [
        "estimate time", "how long would", "how long will",
        "time to machine seed", "time for seed", "estimated time for",
        "machining time for seed", "how long for seed",
    ],
    "factory_tool_usage_on_machine": [
        "tool usage on", "tools on machine", "tools used by",
        "tool history on", "tool events on", "which tools on",
        "tool log for", "tool log on",
    ],
    "factory_tool_usage_history": [
        "tool usage history", "tool history", "tool events",
        "all tool changes", "tool breakage", "tool replacements",
        "tool log", "tools changed", "tools replaced",
        "tool usage across", "tool wear history",
    ],
    "factory_idle_time": [
        "idle time", "how long idle", "machine idle",
        "idle hours", "idle percentage", "idle periods",
        "machines idle", "downtime", "non-productive time",
    ],
    "factory_schedule_history": [
        "schedule history", "allocation history", "scheduling log",
        "which machine ran", "assignment history", "job allocation",
        "part schedule", "which machine machined",
    ],
    "factory_machined_parts": [
        "machined parts", "completed parts", "finished parts",
        "parts done", "parts that have been", "parts completed",
        "what has been made", "what was machined", "production history",
        "parts history", "what parts have been",
    ],
    "factory_machining_parts": [
        "being machined", "currently machining", "on the machine",
        "what parts are being", "parts being", "in progress",
        "what is being cut", "what is on the machines",
    ],
    "factory_queue": [
        "queue", "queued parts", "parts waiting", "jobs waiting",
        "what is in the queue", "what's in the queue",
        "queue depth", "backlog", "parts queued",
        "how many parts", "parts in queue",
    ],
    "factory_active_machines": [
        "active machines", "machines running", "which machines",
        "what machines are running", "machines are active",
        "which machines are", "what is running",
        "machines working", "busy machines",
    ],
    "factory_status": [
        "factory status", "fleet status", "factory state",
        "how is the factory", "overall status", "factory kpi",
        "production summary", "factory summary", "how are the machines",
        "what is the factory", "factory overview",
    ],
}


def classify_factory_message(text: str) -> tuple[str, Optional[str]]:
    """
    Classify operator text at the factory level.

    Returns:
      ("query",  intent_id)  — data query; execute via FactoryIntentExecutor
      ("action", None)       — action / advisory; classify further with
                               classify_factory_action()
    """
    lower = text.lower()

    # 1. Action-signal words short-circuit
    for sig in _FACTORY_ACTION_SIGNALS:
        if sig in lower:
            return ("action", None)

    # 2. Phrase scan
    for intent_id, keywords in _FACTORY_QUERY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return ("query", intent_id)

    # 3. Token-combination fallback
    result = _factory_token_classify(lower)
    if result:
        return ("query", result)

    # 4. Default — route to action classifier
    return ("action", None)


def _factory_token_classify(lower: str) -> Optional[str]:
    """Word-token semantic fallback for factory-level queries."""

    def has(words):
        return any(w in lower for w in words)

    Q     = ["what", "which", "show", "list", "how many", "give",
             "tell", "display", "are there"]
    MACH  = ["machine", "machines", "m01", "m02", "m03", "m04"]
    PART  = ["part", "parts", "job", "jobs", "seed", "seeds", "piece"]
    TOOL  = ["tool", "tools", "insert", "crib", "life", "wear"]
    IDLE  = ["idle", "downtime", "waiting", "not cutting"]
    DONE  = ["done", "finished", "completed", "machined", "made"]
    QUEUE = ["queue", "queued", "waiting", "backlog", "pending"]
    SCHED = ["schedule", "allocated", "assignment", "allocation"]
    STAT  = ["status", "state", "kpi", "summary", "overview",
             "how is", "how are"]

    if has(TOOL) and has(["history", "log", "events", "usage"]):
        if has(MACH):
            return "factory_tool_usage_on_machine"
        return "factory_tool_usage_history"

    if has(IDLE) and has(MACH + Q):
        return "factory_idle_time"

    if has(SCHED) and has(["history", "log"]):
        return "factory_schedule_history"

    if has(PART) and has(DONE) and has(Q):
        return "factory_machined_parts"

    if has(PART) and has(["being", "currently", "in progress", "on machine"]):
        return "factory_machining_parts"

    if has(PART + ["job"]) and has(QUEUE + ["waiting"]):
        return "factory_queue"

    if has(MACH) and has(["running", "active", "busy", "working"]):
        return "factory_active_machines"

    if has(STAT):
        return "factory_status"

    return None


# Factory-level action types and their keyword triggers
_FACTORY_ACTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("factory_remove_from_queue", [
        "remove from queue", "cancel job", "cancel seed",
        "drop from queue", "delete from queue", "pull from queue",
        "remove seed", "remove job", "dequeue",
    ]),
    ("factory_add_to_queue", [
        "add to queue", "queue seed", "submit seed",
        "add seed", "enqueue seed", "add part",
        "machine seed", "schedule seed", "add job",
    ]),
    ("factory_replace_tool", [
        "replace tool", "change tool on", "swap tool on",
        "install tool", "replace t", "change t",
    ]),
    ("factory_set_policy", [
        "set policy", "change policy", "switch policy",
        "use policy", "policy to", "change to",
    ]),
    ("factory_ai_advice", [
        "advise", "advice", "analyse", "analyze",
        "optimise", "optimize", "improve",
        "recommendation", "suggest", "how can",
        "how do i", "why is", "what should",
        "tell me how", "help me", "ai analysis",
        "send to openai", "ask openai", "consult ai",
    ]),
]


def classify_factory_action(text: str) -> tuple[str, dict]:
    """
    Classify a factory-level action message.
    Returns (action_type, parameters_dict).
    """
    lower  = text.lower()
    params: dict = {}

    # Extract machine ID
    m = _MACHINE_ID_RE.search(text)
    if m:
        params["machine_id"] = m.group(1).upper()

    # Extract seed
    s = _SEED_RE.search(text)
    if s:
        params["seed"] = int(s.group(1))

    # Extract tool ID
    t = _TOOL_ID_RE.search(text)
    if t:
        params["tool_id"] = f"T{t.group(1)}"

    # Extract n
    n = _N_RE.search(text)
    if n:
        params["n"] = int(next(v for v in n.groups() if v))

    # Extract policy
    for alias, canonical in _VALID_POLICIES.items():
        if alias in lower:
            params["policy"] = canonical
            break

    # Extract question (full text minus any commands) for advisory
    params["question"] = text.strip()

    for action_type, keywords in _FACTORY_ACTION_PATTERNS:
        for kw in keywords:
            if kw in lower:
                return (action_type, params)

    return ("unknown", params)


def extract_factory_parameters(text: str, intent_id: str) -> dict:
    """Extract structured parameters from free text for factory intents."""
    params: dict = {}

    m = _MACHINE_ID_RE.search(text)
    if m:
        params["machine_id"] = m.group(1).upper()

    s = _SEED_RE.search(text)
    if s:
        params["seed"] = int(s.group(1))

    t = _TOOL_ID_RE.search(text)
    if t:
        params["tool_id"] = f"T{t.group(1)}"

    n = _N_RE.search(text)
    if n:
        params["n"] = int(next(v for v in n.groups() if v))

    lower = text.lower()
    for alias, canonical in _VALID_POLICIES.items():
        if alias in lower:
            params["policy"] = canonical
            break

    return params


# ══════════════════════════════════════════════════════════════════════════════
#  PART 4 — FACTORY INTENT EXECUTOR (queries)
# ══════════════════════════════════════════════════════════════════════════════

class FactoryIntentExecutor:
    """
    Execute factory-level query intents against the live FactoryAgent and
    its FactoryLedger.

    Returns {"intent": intent_id, "result": formatted_string, "raw": object}.

    Injected references:
      self._fa      — FactoryAgent
      self._ledger  — FactoryLedger (fa.ledger)
      self._msim    — dict[machine_id, _MachSim]  (set by GUI if available)
    """

    def __init__(self, factory_agent, app=None):
        self._fa     = factory_agent
        self._ledger: FactoryLedger = getattr(factory_agent, "ledger",
                                               FactoryLedger())
        self._msim   = {}     # injected by caller: machine_id → _MachSim
        self._app    = app

    def execute(self, intent_id: str, parameters: dict = None) -> dict:
        parameters = parameters or {}
        handler = getattr(self, f"_do_{intent_id}", self._do_unknown)
        return handler(parameters)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _tick(self) -> int:
        return getattr(self._fa.scheduler, "tick", 0)

    def _t_real_s(self) -> float:
        return getattr(self._fa.scheduler, "t_real_s", 0.0)

    # ── query handlers ────────────────────────────────────────────────────────

    def _do_factory_status(self, p: dict) -> dict:
        kpis  = self._fa.get_production_kpis()
        tick  = self._tick()
        t_s   = self._t_real_s()
        # Use GUI completed-parts count if available (more accurate in _MachSim mode)
        gui_done_count = len(getattr(self._app, "_completed_parts", []))
        completed_count = max(kpis["total_jobs_completed"], gui_done_count)
        lines = [
            f"Factory Status — tick {tick}  "
            f"(sim time {t_s/60:.1f} min)  policy: {self._fa.policy}",
            f"",
            f"  Machines running          : {kpis['machines_running']}",
            f"  Machines idle             : {kpis['machines_idle']}",
            f"  Machines stopped/waiting  : {kpis['machines_stopped']}",
            f"  Awaiting factory response : {kpis['machines_awaiting_factory']}",
            f"",
            f"  Jobs completed            : {completed_count}",
            f"  Jobs in rework            : {kpis['rework_queue_depth']}",
            f"  Jobs in scheduler queue   : {kpis['scheduler_queue_depth']}",
            f"",
            f"  Errors processed          : {kpis['total_errors_processed']}",
            f"  Tools replaced            : {kpis['total_tools_replaced']}",
            f"  Avg tool life             : {kpis['avg_tool_life_pct']:.1f}%",
        ]
        # Per-machine brief
        lines.append("")
        lines.append("  Per-machine:")
        for ag in self._fa.agents:
            s = ag.get_state()
            msim = self._msim.get(ag.machine_id)
            job_desc = "—"
            if msim and getattr(msim, "current_job", None):
                job = msim.current_job
                pct = 0.0
                if getattr(msim, "gcode_lines", []):
                    cur = getattr(msim, "gcode_cursor", 0)
                    tot = len(msim.gcode_lines)
                    pct = (cur / tot * 100) if tot else 0.0
                job_desc = (f"Seed {getattr(job,'seed','?')} "
                            f"{pct:.0f}%")
            lines.append(
                f"    {ag.machine_id}  {ag.status:<22s}  "
                f"job={job_desc}  "
                f"queue={s['queue_depth']}  "
                f"rework={s['rework_depth']}"
            )
        return {"intent": "factory_status", "result": "\n".join(lines), "raw": kpis}

    def _do_factory_active_machines(self, p: dict) -> dict:
        lines = ["Active machines (currently running a job):"]
        found = False
        for ag in self._fa.agents:
            msim = self._msim.get(ag.machine_id)
            if ag.status != "running":
                continue
            found = True
            job = None
            pct = 0.0
            if msim and getattr(msim, "current_job", None):
                job = msim.current_job
                if getattr(msim, "gcode_lines", []):
                    cur = getattr(msim, "gcode_cursor", 0)
                    tot = len(msim.gcode_lines)
                    pct = (cur / tot * 100) if tot else 0.0
            elif ag.current_job:
                job = ag.current_job
            seed   = getattr(job, "seed", "?") if job else "?"
            mat    = getattr(job, "material", "?") if job else "?"
            job_id = getattr(job, "job_id", "?")[:8] if job else "?"
            lines.append(
                f"  {ag.machine_id}  —  Job {job_id}…  "
                f"Seed {seed}  Material {mat}  Progress {pct:.0f}%"
            )
        if not found:
            lines.append("  No machines currently running.")
        return {"intent": "factory_active_machines",
                "result": "\n".join(lines), "raw": []}

    def _do_factory_machining_parts(self, p: dict) -> dict:
        lines = ["Parts currently being machined:"]
        found = False
        for ag in self._fa.agents:
            msim = self._msim.get(ag.machine_id)
            job  = None
            if msim and getattr(msim, "current_job", None):
                job = msim.current_job
            elif ag.current_job:
                job = ag.current_job
            if job is None:
                continue
            found = True
            pct = 0.0
            if msim and getattr(msim, "gcode_lines", []):
                cur = getattr(msim, "gcode_cursor", 0)
                tot = len(msim.gcode_lines)
                pct = (cur / tot * 100) if tot else 0.0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            else:
                bar = "░" * 20
            est_s = getattr(job, "estimated_time_s", 0)
            remaining_s = est_s * (1 - pct / 100) if pct < 100 else 0.0
            lines += [
                f"",
                f"  {ag.machine_id}  —  Seed {getattr(job,'seed','?')}",
                f"    Material   : {getattr(job,'material','?')}",
                f"    Job ID     : {getattr(job,'job_id','?')[:12]}…",
                f"    Progress   : [{bar}] {pct:.1f}%",
                f"    Est. total : {est_s/60:.1f} min",
                f"    Remaining  : {remaining_s/60:.1f} min",
                f"    Section    : {ag.current_section or '—'}",
            ]
        if not found:
            lines.append("  No parts on any machine right now.")
        return {"intent": "factory_machining_parts",
                "result": "\n".join(lines), "raw": []}

    def _do_factory_machined_parts(self, p: dict) -> dict:
        """
        Data-source priority (the GUI bypasses fa.enqueue_job so the scheduler
        never sees most jobs):
          1. FactoryLedger.completed_parts()  — populated since fix of 2025-06-14
          2. self._app._completed_parts       — GUI's own tracking (always correct)
          3. fa.scheduler.completed_jobs      — scheduler path (rarely used in GUI mode)
        """
        n      = int(p.get("n", 10))
        lines  = [f"Completed parts (last {n}):"]

        # ── Source 1: ledger ─────────────────────────────────────────────
        done_ledger = self._ledger.completed_parts(n)

        # ── Source 2: GUI _completed_parts list ──────────────────────────
        gui_done = []
        if self._app is not None:
            gui_done = list(getattr(self._app, "_completed_parts", []))

        if done_ledger:
            # Ledger is populated — show it (richest data)
            lines.append(
                f"  {'Seed':<8} {'Machine':<6} {'Material':<22} "
                f"{'Est(min)':>8} {'Act(min)':>8} {'Cost$':>7} {'Errors':>6}"
            )
            lines.append("  " + "─" * 70)
            for r in done_ledger:
                lines.append(
                    f"  {str(r.seed):<8} {r.machine_id:<6} {r.material:<22} "
                    f"{r.estimated_time_s/60:>8.1f} "
                    f"{r.actual_time_s/60:>8.1f} "
                    f"{r.cost_usd:>7.2f} "
                    f"{r.error_count:>6}"
                    + (" ⚠REWORK" if r.rework else "")
                )
            return {"intent": "factory_machined_parts",
                    "result": "\n".join(lines), "raw": done_ledger}

        if gui_done:
            # GUI list is the authoritative runtime source
            gui_recent = gui_done[-n:]
            total      = len(gui_done)
            lines.append(f"  Showing last {len(gui_recent)} of {total} total completions:")
            lines.append("")
            lines.append(
                f"  {'#':<4} {'Seed':<8} {'Machine':<6} {'Tool':<6} "
                f"{'Mach(min)':>10} {'Total(min)':>11} {'Tick':>6}"
            )
            lines.append("  " + "─" * 60)
            for i, cp in enumerate(reversed(gui_recent), 1):
                seed    = cp.get("seed", "?")
                machine = cp.get("machine", "?")
                tool    = cp.get("tool_id", "—")
                mach_m  = cp.get("machining_s", 0) / 60
                total_m = cp.get("total_s", 0) / 60
                tick_d  = cp.get("tick_done", "?")
                lines.append(
                    f"  {i:<4} {str(seed):<8} {machine:<6} {str(tool):<6} "
                    f"{mach_m:>10.1f} {total_m:>11.1f} {str(tick_d):>6}"
                )
            return {"intent": "factory_machined_parts",
                    "result": "\n".join(lines), "raw": gui_recent}

        # ── Source 3: scheduler completed_jobs ───────────────────────────
        sched_done = self._fa.scheduler.completed_jobs[-n:]
        if sched_done:
            lines.append(
                f"  (scheduler log — {len(sched_done)} job(s))"
            )
            for j in reversed(sched_done):
                lines.append(
                    f"  Seed {j.seed}  Job {j.job_id[:8]}\u2026  "
                    f"machine={j.assigned_machine_id or '?'}  "
                    f"material={j.material}  "
                    f"est={j.estimated_time_s/60:.1f} min"
                )
            return {"intent": "factory_machined_parts",
                    "result": "\n".join(lines), "raw": sched_done}

        lines.append("  No completed parts yet — parts complete shortly after machining starts.")
        return {"intent": "factory_machined_parts",
                "result": "\n".join(lines), "raw": []}

    def _do_factory_queue(self, p: dict) -> dict:
        queue = list(self._fa.scheduler.job_queue)
        lines = [f"Scheduler queue — {len(queue)} job(s) waiting:"]
        if not queue:
            lines.append("  Queue is empty.")
        else:
            lines.append(
                f"  {'#':<3} {'Seed':<8} {'Job ID':<14} "
                f"{'Material':<22} {'Est(min)':>8} {'Priority':>8}"
            )
            lines.append("  " + "─" * 68)
            for i, j in enumerate(queue, 1):
                lines.append(
                    f"  {i:<3} {str(j.seed):<8} {j.job_id[:12]:<14} "
                    f"{j.material:<22} {j.estimated_time_s/60:>8.1f} "
                    f"{j.priority:>8.2f}"
                )
        return {"intent": "factory_queue", "result": "\n".join(lines), "raw": queue}

    def _do_factory_schedule_history(self, p: dict) -> dict:
        n    = int(p.get("n", 20))
        log  = self._ledger.allocation_log[-n:]
        lines = [f"Schedule / allocation history (last {n}):"]
        if log:
            lines.append(
                f"  {'Tick':<6} {'Job ID':<14} {'Machine':<8} {'Policy'}"
            )
            lines.append("  " + "─" * 50)
            for tick_v, job_id, machine_id, policy in log:
                lines.append(
                    f"  {tick_v:<6} {job_id[:12]:<14} {machine_id:<8} {policy}"
                )
        else:
            # Fall back to GUI _all_jobs tracking list
            all_jobs = list(getattr(self._app, "_all_jobs", []))
            if all_jobs:
                recent = all_jobs[-n:]
                lines.append(
                    f"  (from GUI job registry — {len(all_jobs)} total submitted)"
                )
                lines.append(
                    f"  {'Seed':<8} {'Job ID':<14} {'Machine':<8} {'Status':<10} {'Lines'}"
                )
                lines.append("  " + "─" * 55)
                for entry in reversed(recent):
                    lines.append(
                        f"  {str(entry.get('seed','?')):<8} "
                        f"{entry.get('job_id','')[:12]:<14} "
                        f"{entry.get('machine','?'):<8} "
                        f"{entry.get('status','?'):<10} "
                        f"{entry.get('lines',0)}"
                    )
            else:
                lines.append("  No schedule history yet.")
        return {"intent": "factory_schedule_history",
                "result": "\n".join(lines), "raw": log}

    def _do_factory_idle_time(self, p: dict) -> dict:
        """
        Idle time per machine — always returns real numbers.

        Data sources (in priority order):
          1. FactoryLedger (if idle events are recorded)
          2. GUI _completed_parts → infer busy time, idle = total - busy
             + live _msim status for the current period
        Both paths show: idle ticks, idle %, idle time (min), current status.
        """
        machine_id = p.get("machine_id")
        tick       = self._tick()
        t_tick_s   = getattr(__import__("config"), "T_TICK_S", 5.0)

        # ── Build per-machine status from live sources ────────────────────
        # agent.status is the authoritative runtime flag (set by GUI _MachSim)
        agent_map: dict = {ag.machine_id: ag for ag in self._fa.agents}
        # _msim tells us if G-code is currently executing
        msim_map  = self._msim  # mid → _MachSim (may be empty)

        def _current_status(mid: str) -> str:
            ag   = agent_map.get(mid)
            msim = msim_map.get(mid)
            if msim is not None:
                if getattr(msim, "current_job", None) is not None:
                    return "machining"
                q = getattr(msim, "queue", [])
                if q:
                    return "loading"
            if ag:
                return ag.status
            return "unknown"

        # ── Source 1: ledger idle events ──────────────────────────────────
        ledger_summary = self._ledger.idle_summary(machine_id, current_tick=tick)

        # ── Source 2: GUI _completed_parts timing ─────────────────────────
        gui_done: list = []
        if self._app is not None:
            gui_done = list(getattr(self._app, "_completed_parts", []))

        # Compute busy ticks per machine from completed jobs + current job
        from collections import defaultdict
        import config as _cfg
        busy_ticks_map: dict = defaultdict(float)
        job_counts:     dict = defaultdict(int)

        for cp in gui_done:
            m2     = cp.get("machine", "")
            total_s = (cp.get("machining_s", 0)
                       + cp.get("load_s", 0)
                       + cp.get("unload_s", 0))
            busy_ticks_map[m2] += total_s / _cfg.T_TICK_S
            job_counts[m2]     += 1

        # Add currently running job (partial progress from _msim cursor)
        for ag in self._fa.agents:
            msim = msim_map.get(ag.machine_id)
            if msim is None:
                continue
            cur_job = getattr(msim, "current_job", None)
            if cur_job is None:
                continue
            glines = getattr(msim, "gcode_lines", [])
            cursor = getattr(msim, "gcode_cursor", 0)
            tot    = len(glines)
            if tot > 0:
                frac    = cursor / tot
                est_s   = getattr(cur_job, "estimated_time_s", 0)
                busy_s  = frac * (est_s + _cfg.PART_SETUP_TIME_S + _cfg.PART_REMOVAL_TIME_S)
                busy_ticks_map[ag.machine_id] += busy_s / _cfg.T_TICK_S

        # ── Choose which source has data ──────────────────────────────────
        has_ledger = bool(ledger_summary)
        has_gui    = bool(gui_done) and tick > 0
        all_mids   = sorted(
            set(list(agent_map.keys())
                + list(ledger_summary.keys())
                + [cp.get("machine","") for cp in gui_done if cp.get("machine")])
        )
        if machine_id:
            all_mids = [m for m in all_mids if m == machine_id]

        t_real_s = self._t_real_s()
        lines = [
            f"Idle time per machine — tick {tick}  "
            f"({t_real_s/60:.1f} sim-min  ×{_cfg.T_TICK_S:.0f}s/tick)",
            f"  Shift elapsed: {t_real_s/60:.1f} min  "
            f"(policy: {self._fa.policy})",
            "",
        ]

        # Header
        lines.append(
            f"  {'Machine':<6}  {'Status':<12}  "
            f"{'Idle time':>12}  {'Idle %':>7}  "
            f"{'Busy time':>12}  {'Jobs done':>10}  {'Idle periods'}"
        )
        lines.append("  " + "─" * 76)

        for mid in all_mids:
            cur_status = _current_status(mid)

            if has_ledger and mid in ledger_summary:
                d          = ledger_summary[mid]
                idle_ticks = d["total_idle_ticks"]
                n_periods  = d["n_idle_periods"]
            elif has_gui:
                bt         = busy_ticks_map.get(mid, 0.0)
                idle_ticks = max(0.0, tick - bt)
                n_periods  = job_counts.get(mid, 0)
            else:
                idle_ticks = 0.0
                n_periods  = 0

            busy_ticks = max(0.0, tick - idle_ticks)
            idle_pct   = (idle_ticks / tick * 100) if tick > 0 else 0.0
            idle_min   = idle_ticks * _cfg.T_TICK_S / 60
            busy_min   = busy_ticks * _cfg.T_TICK_S / 60
            n_jobs     = job_counts.get(mid, 0)

            # Status indicator
            if cur_status == "machining":
                status_str = "▶ machining"
            elif cur_status in ("idle", ""):
                status_str = "○ idle"
            elif cur_status == "loading":
                status_str = "◎ loading"
            elif cur_status == "awaiting_factory":
                status_str = "⚠ waiting"
            else:
                status_str = cur_status

            lines.append(
                f"  {mid:<6}  {status_str:<12}  "
                f"{idle_min:>10.1f}m  {idle_pct:>6.1f}%  "
                f"{busy_min:>10.1f}m  {n_jobs:>10}  "
                f"{n_periods} period(s)"
            )

        # ── Current idle streak detail ────────────────────────────────────
        lines.append("")
        lines.append("  Current idle streaks:")
        any_idle = False
        for mid in all_mids:
            ag  = agent_map.get(mid)
            if ag is None:
                continue
            cur = _current_status(mid)
            if cur not in ("idle", "○ idle", ""):
                continue
            # Find open idle event
            open_ev = self._ledger._open_idle.get(mid)
            if open_ev:
                streak_ticks = max(0, tick - open_ev.start_tick)
                streak_min   = streak_ticks * _cfg.T_TICK_S / 60
                lines.append(
                    f"    {mid}  idle since tick {open_ev.start_tick}  "
                    f"→  {streak_min:.1f} min  ({streak_ticks} ticks)"
                )
            else:
                lines.append(f"    {mid}  currently idle  (streak not in ledger)")
            any_idle = True
        if not any_idle:
            lines.append("    No machines currently idle.")

        data_src = ("ledger" if has_ledger else
                    "completed-parts estimate" if has_gui else
                    "live agent status only")
        lines.append("")
        lines.append(f"  Data source: {data_src}")

        return {"intent": "factory_idle_time",
                "result": "\n".join(lines), "raw": ledger_summary}

    def _do_factory_tool_usage_history(self, p: dict) -> dict:
        n      = int(p.get("n", 20))
        events = self._ledger.tool_events[-n:]
        lines  = [f"Tool usage history — all machines (last {n} events):"]
        if not events:
            # Build live snapshot from all cribs
            lines.append("  (No ledger events yet — live crib snapshot:)")
            for ag in self._fa.agents:
                tools = ag.tool_crib.state_list()
                for t in tools:
                    life   = t["remaining_life_pct"]
                    status = ("⛔ STOP" if t["needs_replacement"]
                              else ("⚠ WARN" if life < 20 else "✓ OK"))
                    lines.append(
                        f"  {ag.machine_id}  {t['tool_id']:<5}  "
                        f"Ø{t['diameter_mm']:.0f}mm  "
                        f"life={life:.1f}%  {status}"
                    )
        else:
            lines.append(
                f"  {'Tick':<6} {'Machine':<8} {'Tool':<6} {'Type':<12} "
                f"{'Life%':>7} {'Message'}"
            )
            lines.append("  " + "─" * 70)
            for ev in reversed(events):
                sub = " (substitute)" if ev.substitute else ""
                lines.append(
                    f"  {ev.tick:<6} {ev.machine_id:<8} {ev.tool_id:<6} "
                    f"{ev.event_type:<12} {ev.life_pct:>6.1f}%  "
                    f"{ev.message[:30]}{sub}"
                )
        return {"intent": "factory_tool_usage_history",
                "result": "\n".join(lines), "raw": events}

    def _do_factory_tool_usage_on_machine(self, p: dict) -> dict:
        machine_id = p.get("machine_id", "")
        if not machine_id:
            return {"intent": "factory_tool_usage_on_machine",
                    "result": "Please specify a machine (e.g. M01).",
                    "raw": []}
        events = self._ledger.tool_events_for_machine(machine_id)
        n      = int(p.get("n", 20))
        events = events[-n:]

        # Also get live crib state
        agent  = next((a for a in self._fa.agents
                       if a.machine_id == machine_id), None)
        lines  = [f"Tool usage — {machine_id}:"]
        if agent:
            lines.append("  Live crib state:")
            for t in agent.tool_crib.state_list():
                life   = t["remaining_life_pct"]
                status = ("⛔ STOP" if t["needs_replacement"]
                          else ("⚠ WARN" if life < 20 else "✓ OK"))
                bar    = "█" * int(life/5) + "░" * (20 - int(life/5))
                lines.append(
                    f"    {t['tool_id']:<5}  Ø{t['diameter_mm']:.0f}mm  "
                    f"[{bar}] {life:.1f}%  {status}"
                )
        if events:
            lines.append("")
            lines.append(f"  Event log (last {len(events)}):")
            for ev in reversed(events):
                lines.append(
                    f"    [{ev.tick:>6}] {ev.tool_id:<5}  "
                    f"{ev.event_type:<12}  {ev.life_pct:.1f}%  {ev.message}"
                )
        else:
            lines.append("  No tool events in ledger for this machine.")
        return {"intent": "factory_tool_usage_on_machine",
                "result": "\n".join(lines), "raw": events}

    def _do_factory_estimate_time(self, p: dict) -> dict:
        seed   = p.get("seed")
        policy = p.get("policy", self._fa.policy)
        if seed is None:
            return {"intent": "factory_estimate_time",
                    "result": "Please specify a seed number (e.g. seed 1042).",
                    "raw": {}}
        # Lightweight estimate: average the per-machine time estimates
        # without running the full CAM pipeline
        lines = [f"Estimated machining time — Seed {seed}  policy={policy}:"]
        for ag in self._fa.agents:
            # Use last completed job's timing as a proxy if we have ledger data
            # Otherwise report scheduler state
            sm = next((m for m in self._fa.scheduler.machines
                       if m.machine_id == ag.machine_id), None)
            queue_depth = len([j for j in self._fa.scheduler.job_queue
                               if getattr(j, "seed", None) == seed])
            lines.append(
                f"  {ag.machine_id}  —  "
                f"queue position: {'in queue' if queue_depth else 'not queued'}  "
                f"tool life: {sm.tool_life_pct:.1f}%  "
                f"status: {sm.status}"
            )
        lines.append("")
        lines.append(
            "  Note: exact estimate requires running the full CAM pipeline "
            "(factory_add_to_queue or factory_ai_advice)."
        )
        return {"intent": "factory_estimate_time",
                "result": "\n".join(lines), "raw": {}}

    def _do_factory_policy_comparison(self, p: dict) -> dict:
        """Return a comparison table of all policies for a given seed."""
        from scheduler import VALID_POLICIES
        seed = p.get("seed")
        lines = [
            f"Policy comparison — Seed {seed if seed else '(not specified)'}:",
            "",
            f"  {'Policy':<20} {'Description':<45} {'Active?'}",
            "  " + "─" * 72,
        ]
        policy_descs = {
            "min_time":        "Fastest throughput — use largest tools, highest feeds",
            "min_cost":        "Lowest cost — balance machine rate vs cutting time",
            "best_finish":     "Best surface quality — reduce feed 20%, smaller tools",
            "max_tool_life":   "Preserve tools — reduce DoC 10%, step-over 10%",
            "multi_objective": "Balanced: time + cost + tool-life + finish",
        }
        for pol in sorted(VALID_POLICIES):
            active = "✓ ACTIVE" if pol == self._fa.policy else ""
            lines.append(
                f"  {pol:<20} {policy_descs.get(pol,''):<45} {active}"
            )
        lines += [
            "",
            f"  Current policy   : {self._fa.policy}",
            f"  To change        : 'set policy <name>'",
            f"  For AI advice    : 'advise me on the best policy for seed {seed}'",
        ]
        if seed:
            lines += [
                "",
                "  To get an exact per-policy time estimate, run:",
                f"    factory_add_to_queue seed={seed} under each policy",
                f"    or use factory_ai_advice with question='compare policies for seed {seed}'",
            ]
        return {"intent": "factory_policy_comparison",
                "result": "\n".join(lines), "raw": {}}

    def _do_unknown(self, p: dict) -> dict:
        return {"intent": "unknown",
                "result": "Factory intent not recognised.", "raw": {}}


# ══════════════════════════════════════════════════════════════════════════════
#  PART 5 — FACTORY ACTION EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

class FactoryActionExecutor:
    """
    Execute factory-level actions.  Each handler returns a list of
    (speaker, text) tuples — the same protocol as ActionExecutor.

    Injected: self._fa, self._ledger, self._msim, self._app
    """

    def __init__(self, factory_agent, app=None):
        self._fa     = factory_agent
        self._ledger: FactoryLedger = getattr(factory_agent, "ledger",
                                               FactoryLedger())
        self._msim   = {}     # injected by caller
        self._app    = app

    def execute(self, action_type: str, parameters: dict) -> list[tuple]:
        """Returns list of (speaker, text) tuples."""
        handler = getattr(self, f"_do_{action_type}", self._do_unknown)
        return handler(parameters)

    def _tick(self) -> int:
        return getattr(self._fa.scheduler, "tick", 0)

    # ── Action handlers ───────────────────────────────────────────────────────

    def _do_factory_remove_from_queue(self, p: dict) -> list[tuple]:
        seed   = p.get("seed")
        job_id = p.get("job_id")
        queue  = self._fa.scheduler.job_queue

        removed = []
        remaining = []
        for j in queue:
            match = (
                (seed   is not None and getattr(j, "seed", None) == seed) or
                (job_id is not None and j.job_id == job_id)
            )
            if match:
                removed.append(j)
            else:
                remaining.append(j)

        from collections import deque
        self._fa.scheduler.job_queue = deque(remaining)

        if removed:
            desc = "  ".join(
                f"Job {j.job_id[:8]}… (Seed {j.seed})" for j in removed
            )
            return [
                ("factory",
                 f"✅ Removed {len(removed)} job(s) from queue:\n  {desc}"),
                ("factory",
                 f"Queue depth now: {len(self._fa.scheduler.job_queue)}"),
            ]
        else:
            return [("factory",
                     f"⚠ No matching job found "
                     f"(seed={seed}, job_id={job_id}) in the queue.")]

    def _do_factory_add_to_queue(self, p: dict) -> list[tuple]:
        import threading
        seed     = p.get("seed")
        priority = float(p.get("priority", 1.0))

        if seed is None:
            return [("factory",
                     "⚠ Please specify a seed number (e.g. 'add seed 1042 to queue').")]

        lines: list[tuple] = [
            ("factory",
             f"Building job for Seed {seed}  priority={priority}…\n"
             f"Running full CAM pipeline in background."),
        ]

        fa_ref  = self._fa
        app_ref = self._app

        def _build():
            job = fa_ref.build_job_from_seed(seed)
            if job is None:
                if app_ref:
                    app_ref.post_chat("factory", "factory",
                        f"⚠ CAM pipeline failed for Seed {seed}. "
                        f"Check that the .npy file exists.")
                return
            job.priority = priority
            fa_ref.enqueue_job(job, seed)
            # Update ledger
            ledger = getattr(fa_ref, "ledger", None)
            if ledger:
                ledger.record_queued(
                    job.job_id, seed,
                    job.estimated_time_s,
                    job.material,
                    getattr(fa_ref.scheduler, "tick", 0),
                )
            if app_ref:
                app_ref.post_chat("factory", "factory",
                    f"✅ Seed {seed} queued — "
                    f"Job {job.job_id[:8]}…  "
                    f"~{job.estimated_time_s/60:.1f} min  "
                    f"material={job.material}  "
                    f"priority={priority:.1f}\n"
                    f"Queue depth: {len(fa_ref.scheduler.job_queue)}")

        threading.Thread(target=_build, daemon=True).start()
        return lines

    def _do_factory_replace_tool(self, p: dict) -> list[tuple]:
        from cnc_agent import ToolRecord
        machine_id = p.get("machine_id", "")
        tool_id    = p.get("tool_id", "")

        if not machine_id:
            return [("factory",
                     "⚠ Please specify a machine (e.g. M01).")]
        agent = next((a for a in self._fa.agents
                      if a.machine_id == machine_id), None)
        if agent is None:
            return [("factory", f"⚠ Machine {machine_id} not found.")]

        worn = agent.tool_crib.get(tool_id) if tool_id else None
        if tool_id and worn is None:
            return [("factory",
                     f"⚠ Tool {tool_id} not found in {machine_id} crib.")]

        if worn is None:
            # No specific tool — replace the most worn one
            at_stop = agent.tool_crib.tools_at_stop()
            at_warn = agent.tool_crib.tools_at_warn()
            worn    = (at_stop or at_warn or [None])[0]

        if worn is None:
            return [("factory",
                     f"ℹ All tools on {machine_id} are above warn threshold — "
                     f"no replacement needed.")]

        tick = self._tick()
        cmd  = self._fa.install_replacement_tool(agent, worn, tick, urgency="stop")
        # Record in ledger
        self._ledger.record_tool_event(
            tick, machine_id, worn.tool_id, "replace",
            life_pct   = worn.remaining_life_pct,
            diameter_mm= worn.diameter_mm,
            message    = "operator-requested replacement",
        )

        if cmd is None:
            return [("factory",
                     f"⚠ install_replacement_tool returned None for {machine_id}.")]

        if cmd.action == "tool_removed_no_stock":
            return [
                ("factory",
                 f"🚫 No stock for {worn.tool_id} (Ø{worn.diameter_mm:.0f}mm) "
                 f"— tool removed from {machine_id} crib."),
                ("factory",
                 f"Machine will continue on remaining tools."),
            ]

        payload = cmd.payload
        new_id  = payload.get("new_tool_id", "?")
        new_dia = payload.get("diameter_mm", 0)
        sub_msg = (f" (substitute Ø{new_dia:.0f}mm — toolpath replan needed)"
                   if payload.get("substitute") else "")
        return [
            ("factory",
             f"✅ {machine_id} — {worn.tool_id} replaced with {new_id} "
             f"Ø{new_dia:.0f}mm{sub_msg}"),
        ]

    def _do_factory_set_policy(self, p: dict) -> list[tuple]:
        from scheduler import VALID_POLICIES
        new_policy = p.get("policy", "")
        if not new_policy or new_policy not in VALID_POLICIES:
            valid = ", ".join(sorted(VALID_POLICIES))
            return [("factory",
                     f"⚠ Unknown policy '{new_policy}'. Valid: {valid}")]

        old_policy = self._fa.policy
        if old_policy == new_policy:
            return [("factory",
                     f"ℹ Policy is already {new_policy}. No change.")]

        self._fa.policy          = new_policy
        self._fa.scheduler.policy = new_policy
        tick = self._tick()
        self._ledger.record_policy_change(old_policy, new_policy, tick,
                                          reason="operator instruction")
        return [
            ("factory",
             f"✅ Scheduling policy changed:\n"
             f"  {old_policy}  →  {new_policy}\n"
             f"  Takes effect for all future job assignments."),
        ]

    def _do_factory_ai_advice(self, p: dict) -> list[tuple]:
        """
        Compile full factory context → send to OpenAI → parse response
        → dispatch recommended actions.

        This is the most complex handler.  It runs synchronously but is
        always called from a background thread (_execute_factory_action).
        """
        question = p.get("question", "How can I improve factory performance?")
        lines: list[tuple] = [
            ("factory",
             f"Compiling factory context for AI analysis…\n"
             f"Question: {question}"),
        ]

        # ── 1. Compile context ────────────────────────────────────────────
        context = self._compile_factory_context(question)
        lines.append(("factory",
            f"Context compiled: "
            f"{len(context)} chars  "
            f"({len(context.splitlines())} lines)"))

        # ── 2. Call OpenAI (or fallback) ──────────────────────────────────
        if self._fa._use_llm:
            response = self._call_openai(context, question)
        else:
            response = self._fallback_advice(question)

        lines.append(("openai" if self._fa._use_llm else "fallback",
            f"Recommendation:\n{response.get('recommendation', '—')}"))
        lines.append(("openai" if self._fa._use_llm else "fallback",
            f"Reasoning:\n{response.get('reasoning', '—')}"))

        pol_sug = response.get("policy_suggestion")
        if pol_sug:
            lines.append(("factory",
                f"Suggested policy: {pol_sug}"))

        # ── 3. Dispatch recommended actions ───────────────────────────────
        actions = response.get("actions", [])
        if not actions:
            lines.append(("factory",
                "No specific actions recommended — review the advice above."))
            return lines

        lines.append(("factory",
            f"AI recommends {len(actions)} action(s) — awaiting confirmation."))

        # Present each action for operator awareness (auto-execute safe ones)
        for act in actions:
            target  = act.get("target", "factory")
            intent  = act.get("intent", "")
            params  = act.get("parameters", {})
            reason  = act.get("reasoning", "")
            lines.append(("factory",
                f"  • [{target}] {intent}  {params}  — {reason}"))

        # Auto-dispatch only query and non-destructive actions
        SAFE_AUTO = {"factory_set_policy", "factory_status", "factory_queue"}
        dispatched = 0
        for act in actions:
            target = act.get("target", "factory")
            intent = act.get("intent", "")
            params = act.get("parameters", {})
            if intent not in SAFE_AUTO:
                continue  # require operator confirmation for destructive actions
            if target == "factory":
                result = self.execute(intent, params)
                for speaker, text in result:
                    lines.append((speaker, f"[auto] {text}"))
                dispatched += 1
            else:
                # Route to machine agent
                agent = next((a for a in self._fa.agents
                              if a.machine_id == target), None)
                if agent:
                    agent.handle_factory_response({"action": intent, **params})
                    lines.append(("factory",
                        f"[auto] {intent} dispatched to {target}."))
                    dispatched += 1

        if dispatched < len(actions):
            pending = len(actions) - dispatched
            lines.append(("factory",
                f"⚠ {pending} action(s) require operator confirmation.\n"
                f"To execute: type 'confirm <intent> on <target>'"))

        return lines

    # ── AI advisory helpers ───────────────────────────────────────────────────

    def _compile_factory_context(self, question: str) -> str:
        """
        Build the full context string for the AI prompt.
        Includes:
          1. Factory intent catalogue (what the factory can do)
          2. Machine intent catalogue (what each machine can do)
          3. FactoryLedger snapshot (history, idle, tool events, KPIs)
          4. Active policy and scheduler state
          5. Per-machine live state
          6. User question
        """
        from agent_intents import build_system_context

        sections = []

        # 1. Factory capabilities
        sections.append(build_factory_context_block())

        # 2. Machine capabilities (abbreviated — one block covers all machines)
        sections.append(build_system_context("M01…M04", include_examples=False))

        # 3. KPIs and factory state
        kpis  = self._fa.get_production_kpis()
        tick  = getattr(self._fa.scheduler, "tick", 0)
        t_s   = getattr(self._fa.scheduler, "t_real_s", 0.0)
        state_lines = [
            "=== CURRENT FACTORY STATE ===",
            f"Tick: {tick}  Sim-time: {t_s/60:.1f} min  Policy: {self._fa.policy}",
            f"LLM: {'openai' if self._fa._use_llm else 'rule-based fallback'}",
            "",
            "KPIs:",
            f"  Machines running          : {kpis['machines_running']}",
            f"  Machines idle             : {kpis['machines_idle']}",
            f"  Machines stopped          : {kpis['machines_stopped']}",
            f"  Jobs completed            : {kpis['total_jobs_completed']}",
            f"  Jobs in rework            : {kpis['rework_queue_depth']}",
            f"  Jobs in queue             : {kpis['scheduler_queue_depth']}",
            f"  Total errors processed    : {kpis['total_errors_processed']}",
            f"  Total tools replaced      : {kpis['total_tools_replaced']}",
            f"  Avg tool life             : {kpis['avg_tool_life_pct']:.1f}%",
        ]
        sections.append("\n".join(state_lines))

        # 4. Per-machine state
        machine_lines = ["=== PER-MACHINE STATE ==="]
        for ag in self._fa.agents:
            s     = ag.get_state()
            tools = ag.tool_crib.state_list()
            machine_lines += [
                f"{ag.machine_id}  status={ag.status}  "
                f"queue={s['queue_depth']}  rework={s['rework_depth']}",
                f"  Tool crib:",
            ]
            for t in tools:
                machine_lines.append(
                    f"    {t['tool_id']}  Ø{t['diameter_mm']:.0f}mm  "
                    f"life={t['remaining_life_pct']:.1f}%  "
                    f"{'STOP' if t['needs_replacement'] else 'OK'}"
                )
        sections.append("\n".join(machine_lines))

        # 5. Recent tool events (last 10)
        recent_tools = self._ledger.tool_events[-10:]
        if recent_tools:
            tool_lines = ["=== RECENT TOOL EVENTS ==="]
            for ev in recent_tools:
                tool_lines.append(
                    f"  [{ev.tick}] {ev.machine_id} {ev.tool_id} "
                    f"{ev.event_type} life={ev.life_pct:.1f}%"
                )
            sections.append("\n".join(tool_lines))

        # 6. Recent policy changes
        if self._ledger.policy_changes:
            pol_lines = ["=== POLICY HISTORY ==="]
            for pc in self._ledger.policy_changes[-5:]:
                pol_lines.append(
                    f"  [{pc.tick}] {pc.old_policy} → {pc.new_policy}  ({pc.reason})"
                )
            sections.append("\n".join(pol_lines))

        # 7. Idle summary
        idle_sum = self._ledger.idle_summary()
        if idle_sum:
            idle_lines = ["=== IDLE TIME SUMMARY ==="]
            for mid, d in sorted(idle_sum.items()):
                idle_pct = (d["total_idle_ticks"] / tick * 100) if tick > 0 else 0.0
                idle_lines.append(
                    f"  {mid}  idle={idle_pct:.1f}%  "
                    f"periods={d['n_idle_periods']}  "
                    f"longest={d['longest_streak']} ticks"
                )
            sections.append("\n".join(idle_lines))

        # 8. Completed parts summary
        done = self._ledger.completed_parts(5)
        if done:
            done_lines = ["=== RECENTLY COMPLETED PARTS (last 5) ==="]
            for r in done:
                done_lines.append(
                    f"  Seed {r.seed}  {r.machine_id}  {r.material}  "
                    f"est={r.estimated_time_s/60:.1f}min  "
                    f"act={r.actual_time_s/60:.1f}min  "
                    f"errors={r.error_count}"
                )
            sections.append("\n".join(done_lines))

        # 9. The user's question
        sections.append(f"=== OPERATOR QUESTION ===\n{question}")

        return "\n\n".join(sections)

    def _call_openai(self, context: str, question: str) -> dict:
        """Call the OpenAI API and parse the structured JSON response."""
        import json
        system_prompt = (
            "You are an AI advisor for a CNC factory.  "
            "Analyse the provided factory state, ledger, and intent catalogues, "
            "then respond with a JSON object that specifies:\n"
            "  recommendation — plain-text advice (2-4 sentences)\n"
            "  actions        — list of specific intents to execute\n"
            "  policy_suggestion — recommended scheduling policy or null\n"
            "  reasoning      — explanation\n"
            "Do not include prose outside the JSON object."
        )
        try:
            comp = self._fa._llm_client.chat.completions.create(
                model       = config.OPENAI_MODEL,
                max_tokens  = 800,
                temperature = 0.3,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": context},
                ],
            )
            raw  = comp.choices[0].message.content or "{}"
            clean = (raw.strip()
                       .lstrip("```json").lstrip("```")
                       .rstrip("```").strip())
            return json.loads(clean)
        except Exception as exc:
            return {
                "recommendation": f"[OpenAI error: {exc}]",
                "actions": [],
                "policy_suggestion": None,
                "reasoning": "API call failed — using fallback.",
            }

    def _fallback_advice(self, question: str) -> dict:
        """Rule-based advice when OpenAI is not available."""
        lower = question.lower()
        kpis  = self._fa.get_production_kpis()

        recs  = []
        acts  = []

        if kpis["machines_idle"] > kpis["machines_running"]:
            recs.append(
                f"More than half the machines are idle "
                f"({kpis['machines_idle']} idle, {kpis['machines_running']} running).  "
                f"Consider adding more seeds to the queue."
            )
        if kpis["avg_tool_life_pct"] < 30:
            recs.append(
                f"Average tool life is critically low "
                f"({kpis['avg_tool_life_pct']:.1f}%).  "
                f"Replace tools now and consider switching to max_tool_life policy."
            )
            acts.append({
                "target": "factory", "intent": "factory_set_policy",
                "parameters": {"policy": "max_tool_life"},
                "reasoning": "Preserve remaining tool life.",
            })
        if kpis["rework_queue_depth"] > 3:
            recs.append(
                f"Rework queue is high ({kpis['rework_queue_depth']} parts).  "
                f"Consider switching to best_finish policy to reduce rework."
            )
            acts.append({
                "target": "factory", "intent": "factory_set_policy",
                "parameters": {"policy": "best_finish"},
                "reasoning": "Reduce rework by improving surface quality.",
            })
        if "idle" in lower:
            recs.append(
                "To reduce idle time: ensure the queue always has jobs ready "
                "by pre-running the CAM pipeline on the next seeds."
            )
        if not recs:
            recs.append(
                f"Factory is operating under the {self._fa.policy} policy.  "
                f"KPIs appear within normal range.  "
                f"Enable OpenAI for detailed analysis."
            )

        return {
            "recommendation": "  ".join(recs),
            "actions": acts,
            "policy_suggestion": None,
            "reasoning": "[Rule-based fallback — connect OpenAI for full analysis]",
        }

    def _do_unknown(self, p: dict) -> list[tuple]:
        return [("factory",
                 "Factory action not recognised — please rephrase.")]

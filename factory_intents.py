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

    # ── QUERY — detailed machining analysis ─────────────────────────────────
    FactoryIntent(
        "factory_machining_stats", "query",
        "Machining statistics for a seed",
        "Run the full CAM pipeline dry-run on a seed and return per-face "
        "statistics: feature width, tool selected, width/diameter ratio, "
        "passes, estimated time, path length — across one or all machines.",
        ("seed", "machine_id"),
        example='{"intent":"factory_machining_stats","seed":1042,"machine_id":"M02"}',
    ),

    # ── ADVISORY — AI analysis ─────────────────────────────────────────────────
    FactoryIntent(
        "factory_scheduling_advice", "advisory",
        "Scheduling strategy analysis",
        "Analyse current FIFO scheduling, compute per-machine utilisation from "
        "ledger data, compare batch/round-robin/random-assignment strategies "
        "with concrete numbers from the live run.",
        ("question",),
        example='{"intent":"factory_scheduling_advice","question":"how should we schedule parts?"}',
    ),
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
    FactoryIntent(
        "factory_tool_optimisation", "advisory",
        "Optimal tool selection analysis",
        "Aggregate width/diameter ratios from all recorded tool-use events, "
        "bucket by ratio range, then ask OpenAI which tool diameters and "
        "grades are optimal for each bucket and which tools should be "
        "stocked in every machine crib for best fleet-wide performance. "
        "Returns three labelled options (A/B/C); operator chooses one to implement.",
        ("question",),
        example='{"intent":"factory_tool_optimisation",'
                '"question":"which tools should be in all machines?"}',
    ),
    FactoryIntent(
        "factory_implement_tool_option", "action",
        "Implement a pending tool crib option",
        "Apply one of the tool crib options (A, B, or C) returned by "
        "factory_tool_optimisation to all machine cribs in memory. "
        "Changes are live immediately and discarded on program exit.",
        ("option_letter",),
        example='{"intent":"factory_implement_tool_option","option_letter":"A"}',
    ),
    FactoryIntent(
        "factory_set_routing", "action",
        "Set the active job routing strategy",
        "Hotswap the routing algorithm that assigns incoming jobs to machines. "
        "Valid strategies: round_robin (cycles M01→M02→M03→M04→M01), "
        "sequential (M01-first waterfall — overflow to M02/M03/M04), "
        "random (uniform random selection). "
        "Change is live immediately and forgotten when the program exits. "
        "Use this when the operator says 'implement round robin', "
        "'use sequential', 'switch to random routing', etc.",
        ("routing_strategy",),
        example='{"intent":"factory_set_routing","routing_strategy":"round_robin"}',
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
    event_type:  str   # "warn" | "stop" | "replace" | "breakage" | "install" | "used"
    life_pct:    float = 0.0
    diameter_mm: float = 0.0
    message:     str   = ""
    substitute:  bool  = False   # True if a nearest-dia substitute was used
    # Width-to-diameter ratio analysis (set when event_type=="used")
    # width_mm   = narrowest dimension of the feature's safe_region bounding box
    # wd_ratio   = width_mm / diameter_mm  (>1 → wider than tool; ~1 → tight slot)
    width_mm:    float = 0.0
    wd_ratio:    float = 0.0


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
                          substitute: bool = False,
                          width_mm: float = 0.0,
                          wd_ratio: float = 0.0) -> None:
        self.tool_events.append(ToolEvent(
            event_id    = str(uuid.uuid4()),
            tick        = tick,
            machine_id  = machine_id,
            tool_id     = tool_id,
            event_type  = event_type,
            life_pct    = life_pct,
            diameter_mm = diameter_mm,
            message     = message,
            substitute  = substitute,
            width_mm    = width_mm,
            wd_ratio    = wd_ratio,
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

# Matches M01–M04 AND the shorthand M1–M4 (operator convenience).
# The normalise_machine_id() helper below always returns the canonical M0x form.
_MACHINE_ID_RE = _re.compile(r'\bM(0?[1-4])\b', _re.IGNORECASE)


def _normalise_mid(raw: str) -> str:
    """'M2' → 'M02',  'M02' → 'M02',  'm3' → 'M03'  etc."""
    raw = raw.upper()   # e.g. 'm2' → 'M2'
    # strip the leading M, pad the digit to 2 chars, reattach
    digit = raw[1:]     # '2' or '02'
    return "M" + digit.zfill(2)
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
    "compute", "calculate", "run stats", "run analysis", # compute actions
    "implement", "apply option", "go with",             # option implementation
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
    "factory_machining_stats": [
        "machining stats", "machining statistics", "compute stats",
        "analyse seed", "analyze seed", "stats for seed",
        "statistics for seed", "tool usage for seed", "per face",
        "feature analysis", "which tool for seed", "tool selection for",
        "time breakdown", "face breakdown", "pass breakdown",
        "compute statistics", "run on machine",
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


# Advisory markers — if ANY appear in a message longer than 12 words,
# the message is treated as strategic advice, not a data query —
# even if it contains a query keyword (e.g. "idle time is 20%, not good,
# can we improve scheduling?" → advisory).
_ADVISORY_MARKERS = [
    "can we", "could we", "should we", "not good", "too much",
    "best way", "better way", "reduce", "avoid",
    "batch", "random", "scheduling", "release together",
    "what is the best", "how do we", "how should",
    "any ideas", "what do you think", "recommendation",
]


# ── LLM-based intent classifier ──────────────────────────────────────────────
#
# The user's system prompt is stored as a module-level constant so it is built
# once and reused on every call.  The full intent catalogue is appended, making
# the combined string ~1 200–1 800 tokens — well over the 1 024-token threshold
# for OpenAI's automatic prompt caching.  After the first API call the system
# message prefix is cached; subsequent classifications send only the operator's
# sentence (~10 tokens) as the user message.
#
# Latency (warm cache):  ~50–100 ms
# Latency (cold / first):  ~300–600 ms
# Falls back to the keyword classifier silently on any error.

_CLASSIFIER_SYSTEM_PROMPT: Optional[str] = None   # built lazily on first call


def _build_classifier_system_prompt() -> str:
    """
    Build the static system prompt used by the LLM classifier.
    Called once; result cached in _CLASSIFIER_SYSTEM_PROMPT.

    Structure (all static — suitable for OpenAI prompt caching):
      ── User-supplied system prompt (role / task / rules)
      ── Full FACTORY_INTENT_CATALOGUE as a structured reference list
      ── Output instruction: return only the intent_id string
    """
    lines = [
        # ── Verbatim system prompt provided by the operator ──
        "You are an intent classification model for a CNC factory system."
        " Your task is to read a user\u2019s natural-language sentence and select"
        " the single best matching intent from a provided list of candidate"
        " intents. Each intent includes a name, description, and parameters,"
        " and represents a concrete action, query, or advisory capability"
        " within the system. The user may phrase requests in many different"
        " ways, so you must match based on meaning, not keywords. Carefully"
        " consider whether the user is asking for information (query),"
        " requesting a change (action), or seeking advice or optimisation"
        " (advisory). Choose the intent whose purpose most closely aligns"
        " with the user\u2019s underlying goal. If the request implies"
        " improvement, recommendations, or analysis beyond simple data"
        " retrieval, prefer an advisory intent. Return only the selected"
        " intent_id and ensure it is one of the provided options. Do not"
        " invent new intents. If the meaning is ambiguous, choose the closest"
        " match based on overall intent rather than specific wording.",
        "",
        # ── Full intent catalogue as reference ──
        "Available intents:",
        "",
    ]

    # Group by category for readability (helps the model reason correctly)
    for cat in ("query", "action", "advisory"):
        cat_intents = [i for i in FACTORY_INTENT_CATALOGUE if i.category == cat]
        if not cat_intents:
            continue
        lines.append(f"--- {cat.upper()} INTENTS ---")
        for intent in cat_intents:
            lines.append(f"intent_id: {intent.intent_id}")
            lines.append(f"  category   : {intent.category}")
            lines.append(f"  label      : {intent.label}")
            lines.append(f"  description: {intent.description}")
            if intent.parameters:
                lines.append(f"  parameters : {', '.join(intent.parameters)}")
            if intent.example:
                lines.append(f"  example    : {intent.example}")
            lines.append("")

    lines += [
        # ── Output instruction ──
        "Respond with ONLY the intent_id string — nothing else.",
        "Do not include quotes, punctuation, or explanation.",
        "The intent_id must be exactly one of the options listed above.",
    ]
    return "\n".join(lines)


def classify_with_llm(
    text: str,
    llm_client,
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    """
    Classify *text* into a single factory intent_id using OpenAI.

    The static system message (intent catalogue + rules) is built once and
    reused on every call so OpenAI can cache it.  Only the operator's
    sentence changes between calls.

    Returns
    -------
    str   — a valid intent_id from FACTORY_INTENT_MAP, or
    None  — on any error (network, parse, invalid id, timeout).
             The caller falls back to the keyword classifier.
    """
    global _CLASSIFIER_SYSTEM_PROMPT
    if _CLASSIFIER_SYSTEM_PROMPT is None:
        _CLASSIFIER_SYSTEM_PROMPT = _build_classifier_system_prompt()

    try:
        import time as _time
        t0   = _time.monotonic()
        comp = llm_client.chat.completions.create(
            model       = model,
            max_tokens  = 12,       # intent_id is at most ~35 chars / ~4 tokens
            temperature = 0.0,      # deterministic — we want exact classification
            messages    = [
                {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
        )
        elapsed_ms = (_time.monotonic() - t0) * 1000

        raw       = (comp.choices[0].message.content or "").strip()
        intent_id = raw.strip('"\' \n')

        if intent_id not in FACTORY_INTENT_MAP:
            # Model hallucinated or returned prose — treat as a classifier miss
            return None

        # Attach timing/cache info as a module-level attribute for introspection
        usage = getattr(comp, "usage", None)
        ptd   = getattr(usage, "prompt_tokens_details", None) if usage else None
        cached = getattr(ptd, "cached_tokens", None)
        classify_with_llm._last_stats = (
            f"LLM classifier: {intent_id!r}  "
            f"latency={elapsed_ms:.0f}ms  "
            + (f"cached={cached}/{getattr(usage,'prompt_tokens',0)} tokens"
               if cached is not None else "")
        )
        return intent_id

    except Exception:
        return None

# Slot for introspection / test harness
classify_with_llm._last_stats: str = ""


def classify_factory_message(text: str) -> tuple[str, Optional[str]]:
    """
    Classify operator text at the factory level.

    Returns:
      ("query",  intent_id)  — data query; execute via FactoryIntentExecutor
      ("action", None)       — action / advisory; classify further with
                               classify_factory_action()

    Step 0: complexity guard — long messages with advisory markers → action
    Step 1: action-signal words → action
    Step 2: exact phrase scan → query
    Step 3: token-combination fallback → query
    Step 4: default → action
    """
    lower = text.lower()

    # 0. Advisory-marker guard — runs before phrase scan so "idle time is 20%,
    #    not good, can we look at scheduling?" is treated as advisory even
    #    though it contains the query keyword "idle time".
    #    Rule: if ANY advisory marker is present, treat as action regardless
    #    of message length.
    for marker in _ADVISORY_MARKERS:
        if marker in lower:
            return ("action", None)

    # 1. Action-signal words short-circuit
    for sig in _FACTORY_ACTION_SIGNALS:
        if sig in lower:
            return ("action", None)

    # 2. Phrase scan (only reached when no advisory markers were found)
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
    ("factory_scheduling_advice", [
        "scheduling", "schedule parts", "batch", "batching",
        "collect parts", "release together", "random machine",
        "random assignment", "not in order", "avoid idle",
        "reduce idle", "fifo", "order received", "dispatch",
        "how are parts scheduled", "how should parts be scheduled",
        "scheduling strategy", "best scheduling",
        "not good", "can we look at how", "what is the best way",
    ]),
    ("factory_machining_stats", [
        "compute statistics on machining", "compute stats for",
        "run stats on seed", "analyse machining of", "analyze machining of",
        "statistics on seed", "machining analysis for",
    ]),
    ("factory_set_routing", [
        "implement round robin", "use round robin", "activate round robin",
        "switch to round robin", "set routing round robin",
        "routing to round", "round robin routing",
        "implement sequential", "use sequential routing", "activate sequential",
        "switch to sequential", "set routing sequential",
        "implement random routing", "use random routing", "activate random routing",
        "set routing random", "random routing",
        "set routing", "change routing to", "switch routing to",
    ]),
    ("factory_implement_tool_option", [
        "implement option", "apply option", "use option",
        "choose option", "go with option", "select option",
        "activate option", "install option",
    ]),
    ("factory_tool_optimisation", [
        "best tool", "optimal tool", "tool optimis", "tool optimiz",
        "which tool", "tool recommendation", "tool for all machines",
        "tool crib optimis", "tool crib optimiz", "width diameter",
        "wd ratio", "feature ratio", "diameter ratio",
        "best tools to include", "tools for all machines",
        "reduced machining time", "improve tool",
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

    # Extract machine ID (accepts M1–M4 and M01–M04; always stores as M0x)
    m = _MACHINE_ID_RE.search(text)
    if m:
        params["machine_id"] = _normalise_mid("M" + m.group(1))

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

    # Extract routing strategy for factory_set_routing
    _rt_lo = lower
    if "round" in _rt_lo:
        params["routing_strategy"] = "round_robin"
    elif "sequen" in _rt_lo:
        params["routing_strategy"] = "sequential"
    elif "random" in _rt_lo:
        params["routing_strategy"] = "random"
    # (default applied in handler if key missing)

    # Extract option letter (A / B / C) for factory_implement_tool_option
    om = _re.search(r'\boption\s+([A-Ca-c])\b', text)
    if om:
        params["option_letter"] = om.group(1).upper()

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
        params["machine_id"] = _normalise_mid("M" + m.group(1))

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
        Idle time per machine — always shows real measured durations.

        Totals are ALWAYS derived from _completed_parts + _msim (reliable).
        The ledger provides idle-period HISTORY (when periods happened).
        This avoids showing zeros if the ledger was cold or mis-tracked.
        """
        machine_id = p.get("machine_id")
        tick       = self._tick()
        import config as _cfg
        t_tick_s   = _cfg.T_TICK_S
        t_real_s   = self._t_real_s()

        agent_map: dict = {ag.machine_id: ag for ag in self._fa.agents}
        msim_map        = self._msim           # mid → _MachSim

        # ── Build busy-tick totals from _completed_parts (always correct) ──
        from collections import defaultdict
        busy_ticks_map: dict = defaultdict(float)
        job_counts:     dict = defaultdict(int)

        gui_done: list = []
        if self._app is not None:
            gui_done = list(getattr(self._app, "_completed_parts", []))

        for cp in gui_done:
            m2      = cp.get("machine", "")
            total_s = (cp.get("machining_s", 0)
                       + cp.get("load_s", 0)
                       + cp.get("unload_s", 0))
            busy_ticks_map[m2] += total_s / t_tick_s
            job_counts[m2]     += 1

        # Add in-progress job fraction from _msim
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
            est_s  = getattr(cur_job, "estimated_time_s", 0)
            total_cycle_s = est_s + _cfg.PART_SETUP_TIME_S + _cfg.PART_REMOVAL_TIME_S
            frac   = (cursor / tot) if tot > 0 else 0.0
            busy_ticks_map[ag.machine_id] += frac * total_cycle_s / t_tick_s

        # ── Current machine status (from msim, not agent.status) ─────────
        def _msim_status(mid: str) -> str:
            ms = msim_map.get(mid)
            if ms is None:
                ag = agent_map.get(mid)
                return ag.status if ag else "unknown"
            st = getattr(ms, "status", "idle")
            if st == "machining":    return "▶ machining"
            if st in ("setup","unclamp"): return "◎ loading/unclamp"
            if st == "tool_change":  return "🔧 tool change"
            ag = agent_map.get(mid)
            if ag and ag.status == "awaiting_factory": return "⚠ waiting"
            return "○ idle"

        # ── Assemble per-machine rows ────────────────────────────────────
        all_mids = sorted(agent_map.keys())
        if machine_id:
            all_mids = [m for m in all_mids if m == machine_id]

        lines = [
            f"Idle time per machine — tick {tick}  "
            f"({t_real_s/60:.1f} sim-min  ×{t_tick_s:.0f}s/tick)",
            f"  Shift elapsed: {t_real_s/60:.1f} min   policy: {self._fa.policy}",
            "",
            f"  {'Machine':<6}  {'Status':<18}  "
            f"{'Busy time':>10}  {'Idle time':>10}  {'Idle %':>7}  "
            f"{'Jobs done':>10}",
            "  " + "─" * 70,
        ]

        for mid in all_mids:
            bt       = busy_ticks_map.get(mid, 0.0)
            it       = max(0.0, tick - bt)
            idle_pct = (it / tick * 100) if tick > 0 else 0.0
            busy_min = bt * t_tick_s / 60
            idle_min = it * t_tick_s / 60
            n_jobs   = job_counts.get(mid, 0)
            status   = _msim_status(mid)
            lines.append(
                f"  {mid:<6}  {status:<18}  "
                f"{busy_min:>9.1f}m  {idle_min:>9.1f}m  {idle_pct:>6.1f}%  "
                f"{n_jobs:>10}"
            )

        # ── Current idle streaks ─────────────────────────────────────────
        lines.append("")
        lines.append("  Current idle streaks:")
        any_idle = False
        for mid in all_mids:
            status = _msim_status(mid)
            if "idle" not in status:
                continue
            open_ev = self._ledger._open_idle.get(mid)
            if open_ev and open_ev.start_tick > 0:
                streak_ticks = max(0, tick - open_ev.start_tick)
                streak_min   = streak_ticks * t_tick_s / 60
                lines.append(
                    f"    {mid}  idle since tick {open_ev.start_tick}"
                    f"  →  {streak_min:.1f} min  ({streak_ticks} ticks)"
                )
            else:
                lines.append(f"    {mid}  currently idle  (no open ledger event)")
            any_idle = True
        if not any_idle:
            lines.append("    No machines currently idle.")

        # ── Idle period history (from ledger) ────────────────────────────
        # Only show if ledger has meaningful history (more than one period
        # per machine, or periods that have been closed).
        ledger_events = self._ledger.idle_events
        closed_events = [e for e in ledger_events if e.end_tick >= 0]
        if closed_events or len(ledger_events) > len(all_mids):
            lines.append("")
            lines.append("  Idle period history (last 5 closed periods per machine):")
            for mid in all_mids:
                machine_events = [
                    e for e in ledger_events
                    if e.machine_id == mid and e.end_tick >= 0
                ]
                # Filter out trivially short gaps (1-2 ticks = sync lag)
                machine_events = [
                    e for e in machine_events
                    if (e.end_tick - e.start_tick) > 2
                ]
                if not machine_events:
                    continue
                lines.append(f"    ─ {mid} ─" + "─" * 50)
                lines.append(
                    f"    {'#':<4} {'Start tick':>12} {'End tick':>10} "
                    f"{'Duration':>12} {'Reason'}"
                )
                for i, ev in enumerate(machine_events[-5:], 1):
                    dur_ticks = ev.end_tick - ev.start_tick
                    dur_min   = dur_ticks * t_tick_s / 60
                    lines.append(
                        f"    {i:<4} {ev.start_tick:>12} {ev.end_tick:>10} "
                        f"{dur_min:>10.1f}m  {ev.reason}"
                    )
                # Show open event if any
                open_ev = self._ledger._open_idle.get(mid)
                if open_ev:
                    dur_min = max(0, tick - open_ev.start_tick) * t_tick_s / 60
                    lines.append(
                        f"    {'now':<4} {open_ev.start_tick:>12} {'(open)':>10} "
                        f"{dur_min:>10.1f}m  {open_ev.reason} ← current"
                    )

        data_src = (
            "ledger + completed-parts" if closed_events
            else "completed-parts estimate"
            if gui_done
            else "live agent status"
        )
        lines.append("")
        lines.append(f"  Data source: {data_src}")

        return {"intent": "factory_idle_time",
                "result": "\n".join(lines), "raw": {}}


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

    def _do_factory_machining_stats(self, p: dict) -> dict:
        """
        Dry-run the CAM pipeline on a seed and return per-face machining
        statistics including width/diameter ratio, tool selection, estimated
        time, passes, and path length.

        machine_id specified -> that machine only.
        No machine_id        -> all four machines, side-by-side comparison.
        material specified   -> force that material through the pipeline.
        """
        seed       = p.get("seed")
        machine_id = p.get("machine_id")
        material   = p.get("material")

        if seed is None:
            return {"intent": "factory_machining_stats",
                    "result": "⚠ Please specify a seed number (e.g. 'stats for seed 1042').",
                    "raw": {}}

        target_mids = (
            [machine_id] if machine_id
            else sorted(ag.machine_id for ag in self._fa.agents)
        )

        # Optionally pin material so the user gets stats for their choice
        import timing_model as _tm
        _orig_assign = _tm.assign_material
        if material:
            _tm.assign_material = lambda: material
        results: dict = {}
        try:
            for mid in target_mids:
                job = self._fa.build_job_from_seed(seed, machine_id=mid)
                if job is not None:
                    results[mid] = job
        finally:
            _tm.assign_material = _orig_assign

        if not results:
            return {"intent": "factory_machining_stats",
                    "result": f"⚠ CAM pipeline failed for seed {seed}.",
                    "raw": {}}

        import config as _cfg
        mat_shown = material or next(iter(results.values())).material
        lines = [
            f"Machining Statistics — Seed {seed}  |  Material: {mat_shown}",
            "═" * 76,
        ]

        for mid, job in sorted(results.items()):
            tp    = job.toolpath_result
            t_dia = job.tool_diameter_used
            t_id  = job.tool_id_used
            setup = _cfg.PART_SETUP_TIME_S
            unc   = _cfg.PART_REMOVAL_TIME_S
            tot   = job.estimated_time_s + setup + unc

            lines += [
                "",
                f"┌─ {mid}   Tool: {t_id} ⌀{t_dia:.0f}mm   Material: {job.material}",
                f"│  G-code: {len(job.gcode_lines)} lines  "
                f"setup {setup:.0f}s + mach {job.estimated_time_s:.0f}s "
                f"+ unclamp {unc:.0f}s = ➔ {tot:.0f}s ({tot/60:.1f} min)",
                "│",
                f"│  {'Face':>5}  {'FeatW mm':>8}  {'⌀ mm':>6}  {'W/D':>6}  "
                f"{'Passes':>7}  {'Step mm':>7}  {'Path mm':>8}  {'Time s':>7}  Entry",
                "│  " + "─" * 72,
            ]

            face_data = []
            for ft in (tp.faces if tp else []):
                feat_w = 0.0
                if ft.safe_region is not None:
                    try:
                        b = ft.safe_region.bounds
                        feat_w = min(b[2] - b[0], b[3] - b[1])
                    except Exception:
                        pass
                wd = (feat_w / t_dia) if t_dia > 0 and feat_w > 0 else 0.0
                face_data.append((ft.face_id, feat_w, wd, ft))
                if ft.n_passes > 0:
                    lines.append(
                        f"│  {ft.face_id:>5}  {feat_w:>8.1f}  {t_dia:>6.1f}  {wd:>6.2f}  "
                        f"{ft.n_passes:>7}  {ft.stepover_mm:>7.2f}  "
                        f"{ft.path_length_mm:>8.1f}  {ft.estimated_time_s:>7.1f}  {ft.entry_type}"
                    )

            active = [(fid, fw, wd, ft) for fid, fw, wd, ft in face_data if ft.n_passes > 0]
            if active:
                wds    = [wd for _, _, wd, _ in active if wd > 0]
                avg_wd = sum(wds) / len(wds) if wds else 0.0
                lines += [
                    "│  " + "─" * 72,
                    f"├─ Active: {len(active)}/{len(face_data)} faces   "
                    f"W/D  avg={avg_wd:.2f}  "
                    f"min={min(wds) if wds else 0:.2f}  max={max(wds) if wds else 0:.2f}",
                    f"└─ Machining: {job.estimated_time_s:.0f}s ({job.estimated_time_s/60:.1f}min)  "
                    f"Path: {(tp.total_path_length_mm if tp else 0):.0f}mm",
                ]

        if len(results) > 1:
            lines += [
                "",
                "═" * 76,
                "Comparison — all machines:",
                f"  {'Machine':>8}  {'Tool':>5}  {'⌀mm':>5}  {'Faces':>6}  "
                f"{'Mach s':>7}  {'Total s':>7}  {'AvgW/D':>7}  {'G-lines':>8}",
                "  " + "─" * 62,
            ]
            best = min(results, key=lambda m: results[m].estimated_time_s)
            for mid, job in sorted(results.items()):
                tp2 = job.toolpath_result
                wds2: list = []
                for ft in (tp2.faces if tp2 else []):
                    if ft.n_passes > 0 and ft.safe_region is not None:
                        try:
                            b2 = ft.safe_region.bounds
                            fw2 = min(b2[2]-b2[0], b2[3]-b2[1])
                            if fw2 > 0 and job.tool_diameter_used > 0:
                                wds2.append(fw2 / job.tool_diameter_used)
                        except Exception:
                            pass
                avg2   = sum(wds2) / len(wds2) if wds2 else 0.0
                nact2  = sum(1 for f in (tp2.faces if tp2 else []) if f.n_passes > 0)
                total2 = job.estimated_time_s + _cfg.PART_SETUP_TIME_S + _cfg.PART_REMOVAL_TIME_S
                tag    = "  ★ fastest" if mid == best else ""
                lines.append(
                    f"  {mid:>8}  {job.tool_id_used:>5}  {job.tool_diameter_used:>5.1f}  "
                    f"{nact2:>6}  {job.estimated_time_s:>7.0f}  {total2:>7.0f}  "
                    f"{avg2:>7.2f}  {len(job.gcode_lines):>8}{tag}"
                )

        return {
            "intent": "factory_machining_stats",
            "result": "\n".join(lines),
            "raw": {mid: {
                "tool_id":     results[mid].tool_id_used,
                "tool_dia":    results[mid].tool_diameter_used,
                "material":    results[mid].material,
                "estimated_s": results[mid].estimated_time_s,
                "gcode_lines": len(results[mid].gcode_lines),
            } for mid in results},
        }

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
        # Normalise: factory_scheduling_advice can also arrive as
        # factory_ai_advice with scheduling keywords in the question
        q = parameters.get("question", "").lower()
        _SCHED_KEYS = [
            "schedul", "batch", "fifo", "random", "dispatch",
            "collect parts", "release together", "idle time",
            "not in order", "round.robin", "assign",
        ]
        if action_type == "factory_ai_advice" and any(k in q for k in _SCHED_KEYS):
            action_type = "factory_scheduling_advice"
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
        """
        Build and queue a seed for one or all machines.

        Routing:
          - If ``machine_id`` is present in params → queue to that machine only.
          - If the operator phrase contains "all machines" / "all four" / "each machine" /
            "every machine" → queue one copy to every enabled machine.
          - Otherwise → route via _route_dest_only() (default: shortest queue).

        Jobs are inserted directly into _msim[dest].queue so they are visible to
        _tick_machine and can start without the scheduler being involved.
        """
        import threading
        seed     = p.get("seed")
        priority = float(p.get("priority", 1.0))

        if seed is None:
            return [("factory",
                     "⚠ Please specify a seed number (e.g. 'add seed 1042 to queue').")]

        fa_ref  = self._fa
        app_ref = self._app

        # ── Decide which machines to target ───────────────────────────────
        question_lower = p.get("question", "").lower()
        _ALL_PHRASES   = [
            "all machines", "all four", "every machine",
            "each machine", "all 4", "all the machines",
        ]
        explicit_mid = p.get("machine_id")

        if explicit_mid:
            # Operator specified one machine by name
            target_mids = [explicit_mid]
        elif any(ph in question_lower for ph in _ALL_PHRASES):
            # Operator wants one copy on every enabled machine
            target_mids = sorted(
                mid for mid, ms in (getattr(app_ref, "_msim", None) or {}).items()
                if ms.enabled
            )
            if not target_mids:
                # Fallback: all agents
                target_mids = sorted(a.machine_id for a in fa_ref.agents)
        else:
            # Default: route to shortest queue via _route_dest_only()
            if app_ref is not None:
                dest = app_ref._route_dest_only()
            else:
                dest = sorted(a.machine_id for a in fa_ref.agents)[0]
            target_mids = [dest] if dest else []

        if not target_mids:
            return [("factory",
                     "⚠ No enabled machines available — cannot queue the job.")]

        if len(target_mids) > 1:
            announce = (
                f"Building Seed {seed} for all {len(target_mids)} machines: "
                f"{', '.join(target_mids)}  priority={priority}…\n"
                f"Running CAM pipeline in background for each machine."
            )
        else:
            announce = (
                f"Building job for Seed {seed}  "
                f"→ {target_mids[0]}  priority={priority}…\n"
                f"Running full CAM pipeline in background."
            )

        lines: list[tuple] = [("factory", announce)]

        # ── Background builder — one thread per target machine ────────────
        def _build_for(dest_mid: str) -> None:
            job = fa_ref.build_job_from_seed(seed, machine_id=dest_mid)
            if job is None:
                if app_ref:
                    app_ref.post_factory_chat(
                        "factory",
                        f"⚠ CAM pipeline failed for Seed {seed} → {dest_mid}. "
                        f"Check that the .npy file exists.")
                return

            job.priority = priority

            # ── Route into _msim queue (GUI path) ─────────────────────────
            msim_map = getattr(app_ref, "_msim", None) if app_ref else None
            if msim_map and dest_mid in msim_map:
                msim_map[dest_mid].queue.append(job)
                q_depth = len(msim_map[dest_mid].queue)
            else:
                # Fallback: scheduler queue (non-GUI use)
                fa_ref.enqueue_job(job, seed)
                q_depth = len(fa_ref.scheduler.job_queue)

            # ── Ledger write ──────────────────────────────────────────────
            ledger = getattr(fa_ref, "ledger", None)
            if ledger:
                tick = getattr(fa_ref.scheduler, "tick", 0)
                ledger.record_queued(
                    job.job_id, seed,
                    job.estimated_time_s,
                    getattr(job, "material", "unknown"),
                    tick,
                )
                ledger.record_allocated(
                    job.job_id, dest_mid, fa_ref.policy, tick)

            # ── Track in _all_jobs ────────────────────────────────────────
            if app_ref is not None:
                entry = {
                    "seed":    seed,
                    "job_id":  job.job_id,
                    "lines":   len(job.gcode_lines),
                    "status":  "queued",
                    "machine": dest_mid,
                }
                app_ref._all_jobs.append(entry)

            if app_ref:
                app_ref.post_factory_chat(
                    "factory",
                    f"✅ Seed {seed} → {dest_mid}  "
                    f"Job {job.job_id[:8]}…  "
                    f"~{job.estimated_time_s/60:.1f} min  "
                    f"material={getattr(job,'material','?')}  "
                    f"G-code {len(job.gcode_lines)} lines  "
                    f"queue depth now: {q_depth}")

        for mid in target_mids:
            threading.Thread(target=_build_for, args=(mid,), daemon=True).start()

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

    def _do_factory_set_routing(self, p: dict) -> list[tuple]:
        """
        Hotswap the active routing strategy.
        Installs a closure into app._routing_fn — live immediately,
        forgotten when the program exits.  No files modified.
        """
        txt = (p.get("routing_strategy", "")
               or p.get("question", "")).lower()
        if "round" in txt:
            strategy = "round_robin"
        elif "seq" in txt:
            strategy = "sequential"
        elif "rand" in txt:
            strategy = "random"
        else:
            strategy = "round_robin"   # default when ambiguous
        reasoning = f"Operator requested {strategy.replace('_', '-')} routing."
        return self._hotswap_routing(strategy, reasoning)

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


    def _do_factory_scheduling_advice(self, p: dict) -> list[tuple]:
        """
        Deterministic scheduling strategy analysis.
        Answers: FIFO vs batch vs random assignment, with real numbers
        from the live run (no OpenAI required).
        """
        import config as _cfg
        from collections import defaultdict

        tick     = self._tick()
        t_tick_s = _cfg.T_TICK_S
        t_real_s = tick * t_tick_s

        # ── Gather completed-job data ─────────────────────────────────────
        gui_done: list = []
        if self._app is not None:
            gui_done = list(getattr(self._app, "_completed_parts", []))

        # Per-machine busy ticks from completed jobs
        busy_map:  dict = defaultdict(float)
        job_count: dict = defaultdict(int)
        job_times: dict = defaultdict(list)   # mid → [total_s, ...]

        for cp in gui_done:
            m2      = cp.get("machine", "")
            total_s = (cp.get("machining_s", 0)
                       + cp.get("load_s", 0)
                       + cp.get("unload_s", 0))
            busy_map[m2]  += total_s / t_tick_s
            job_count[m2] += 1
            job_times[m2].append(total_s)

        # Add currently running jobs
        for ag in self._fa.agents:
            msim = self._msim.get(ag.machine_id)
            if msim is None:
                continue
            cur = getattr(msim, "current_job", None)
            if cur is None:
                continue
            g = getattr(msim, "gcode_lines", [])
            c = getattr(msim, "gcode_cursor", 0)
            if len(g) > 0:
                frac  = c / len(g)
                est_s = getattr(cur, "estimated_time_s", 0)
                busy_map[ag.machine_id] += frac * (
                    est_s + _cfg.PART_SETUP_TIME_S + _cfg.PART_REMOVAL_TIME_S
                ) / t_tick_s

        all_mids = sorted(self._fa.agents, key=lambda a: a.machine_id)
        all_mids = [a.machine_id for a in all_mids]
        n_machines = len(all_mids)
        total_jobs = sum(job_count.values())
        total_busy = sum(busy_map.values())

        # ── Section 1: What the data actually shows ───────────────────────
        lines = [
            "Scheduling Strategy Analysis",
            "═" * 58,
            "",
            "SECTION 1 — Current utilisation (what the data shows)",
            "─" * 58,
        ]
        if tick == 0 or total_jobs == 0:
            lines.append("  No completed jobs yet — run the simulation longer.")
        else:
            lines.append(
                f"  {'Machine':<6}  {'Jobs':>6}  {'Busy time':>10}  "
                f"{'Idle time':>10}  {'Util %':>7}  {'Avg job':>9}"
            )
            lines.append("  " + "─" * 54)
            for mid in all_mids:
                bt   = busy_map.get(mid, 0.0)
                it   = max(0.0, tick - bt)
                pct  = (bt / tick * 100) if tick > 0 else 0.0
                nj   = job_count.get(mid, 0)
                avg_s = (sum(job_times.get(mid, [0])) / max(1, nj))
                lines.append(
                    f"  {mid:<6}  {nj:>6}  {bt*t_tick_s/60:>9.1f}m  "
                    f"{it*t_tick_s/60:>9.1f}m  {pct:>6.1f}%  "
                    f"{avg_s/60:>8.1f}m"
                )
            # Overall
            overall_util = (total_busy / (tick * n_machines) * 100) if tick > 0 else 0
            lines.append("  " + "─" * 54)
            lines.append(
                f"  {'FLEET':<6}  {total_jobs:>6}  "
                f"{total_busy*t_tick_s/60:>9.1f}m  "
                f"  {'':>9}  {overall_util:>6.1f}%"
            )
            lines.append("")
            # Imbalance diagnosis
            if n_machines > 1:
                job_counts_list = [job_count.get(m, 0) for m in all_mids]
                max_j  = max(job_counts_list)
                min_j  = min(job_counts_list)
                busiest = all_mids[job_counts_list.index(max_j)]
                idlest  = all_mids[job_counts_list.index(min_j)]
                imbalance_ratio = (max_j / max(1, min_j))
                if imbalance_ratio >= 3:
                    lines.append(
                        f"  ⚠  SEVERE IMBALANCE: {busiest} ran {max_j} jobs  vs  "
                        f"{idlest} ran {min_j} jobs  (ratio {imbalance_ratio:.1f}×)"
                    )
                    lines.append(
                        f"     This is a ROUTING problem, not a scheduling problem.")
                    lines.append(
                        f"     Jobs are being preferentially sent to {busiest}/{all_mids[1]}.")
                elif imbalance_ratio >= 1.5:
                    lines.append(
                        f"  ⚡  Mild imbalance: {busiest} ({max_j} jobs) vs "
                        f"{idlest} ({min_j} jobs).  Routing improvement recommended.")

        # ── Section 2: Current strategy (FIFO + shortest-queue routing) ──
        lines += [
            "",
            "SECTION 2 — Your current strategy",
            "─" * 58,
            f"  Routing:    {getattr(getattr(self, '_app', None), '_routing_strategy_name', 'sequential')}",
            "  Dispatch:   Immediate (job enters queue as soon as CAM completes)",
            "  Order:      Seed-order (jobs are offered as auto-seed generates them)",
            "",
            "  ✓  Low latency — each part starts as soon as a machine is free",
            "  ✓  Simple and predictable",
            "  ✗  No look-ahead — cannot group similar parts for faster setup",
            "  ✗  Starvation if CAM throughput is slower than machining throughput",
            "     (M03/M04 sit idle waiting for seeds to be generated)",
        ]

        # ── Section 3: Hotswappable strategy options ───────────────────────────
        avg_job_min = 0.0
        if gui_done:
            all_totals = [
                cp.get("machining_s", 0) + cp.get("load_s", 0) + cp.get("unload_s", 0)
                for cp in gui_done
            ]
            avg_job_min = (sum(all_totals) / len(all_totals)) / 60

        active_strategy = getattr(
            getattr(self, "_app", None), "_routing_strategy_name", "sequential"
        )
        def _tag(s):
            return "  ★ ACTIVE" if s == active_strategy else ""

        lines += [
            "",
            "SECTION 3 — Available routing strategies",
            "  All changes are live-only — forgotten when the program exits.",
            "─" * 58,
            "",
            f"  A)  ROUND-ROBIN{_tag('round_robin')}",
            f"      job 1→M01, job 2→M02, job 3→M03, job 4→M04, job 5→M01…",
            f"      Every machine receives every {n_machines}th job in strict rotation.",
            f"      Latency: none.  Distribution: perfectly even.",
            f"      To activate: type  'implement round robin'",
            "",
            f"  B)  SEQUENTIAL — M01-first waterfall{_tag('sequential')}",
            f"      Each part tries M01 first (idle or queue has space),",
            f"      then M02, M03, M04 in order.  M01 absorbs all work it can;",
            f"      later machines receive overflow only.",
            f"      To activate: type  'implement sequential'",
            "",
            f"  C)  RANDOM{_tag('random')}",
            f"      Uniform random selection from enabled machines.",
            f"      Statistically approaches round-robin over many jobs.",
            f"      To activate: type  'implement random routing'",
        ]

        # ── Section 4: Recommendation — hotswap fires here, no trigger-word gate ────
        lines += [
            "",
            "SECTION 4 — Recommendation for your situation",
            "─" * 58,
        ]

        imb = (max(job_count.values()) / max(1, min(job_count.values()))
               if job_count else 1)

        if active_strategy == "round_robin":
            best_strategy = "round_robin"
            reason = (
                f"Round-robin is already active ({imb:.1f}× imbalance ratio — "
                + ("distribution looks good." if imb < 1.5
                   else "some variance is expected when machines have different cycle times.")
                + ")"
            )
            action_hint = "  No change needed."
        elif imb >= 2.5:
            best_strategy = "round_robin"
            reason = (
                f"Job distribution is unbalanced ({imb:.1f}× ratio).  "
                f"Round-robin will immediately share load evenly across all {n_machines} machines "
                f"with no latency penalty."
            )
            action_hint = "  → To implement: type  'implement round robin'"
        else:
            best_strategy = active_strategy or "sequential"
            reason = (
                f"Distribution is balanced ({imb:.1f}× ratio) and utilisation "
                f"is {overall_util:.0f}%.  Current strategy ({active_strategy}) is appropriate."
            )
            action_hint = "  No change needed."

        lines += [
            f"  BEST STRATEGY: {best_strategy.replace('_', '-').upper()}",
            f"  Reason: {reason}",
            action_hint,
            "",
            "  Also useful regardless of strategy:",
            "    • 'set policy min_time'  — uses largest tools / highest feeds",
            "    • Monitor tool life — tool changes cause unexpected idle gaps",
        ]

        return [("factory", "\n".join(lines))]

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

        # ── 4. Hotswap routing strategy if AI recommended one ────────────────
        routing_strat   = response.get("routing_strategy")
        routing_reason  = response.get("routing_reasoning", "")
        if routing_strat and routing_strat != "shortest_queue":
            lines.extend(
                self._hotswap_routing(routing_strat, routing_reason)
            )

        return lines

    # ── AI advisory helpers ───────────────────────────────────────────────────


    def _hotswap_routing(self, strategy: str, reasoning: str) -> list[tuple]:
        """
        Install a live routing function into the GUI at runtime.

        The function is stored as app._routing_fn — a plain callable that
        _route_dest_only() consults on every job assignment.  It lives only
        in process memory: program exit removes it automatically with no
        config files touched.

        strategy: "round_robin" | "random" | "shortest_queue" (clears override)
        """
        app = self._app
        if app is None:
            return [("factory", "⚠ Routing hotswap unavailable — no app reference.")]

        prev_strategy = getattr(app, "_routing_strategy_name", "shortest_queue")

        if strategy == "round_robin":
            # Round-robin: cycle through eligible machines in sorted order.
            # A counter in the closure advances on every job dispatch.
            state = {"idx": 0}
            def _round_robin_fn(eligible):
                mids = [mid for mid, _ in eligible]   # already sorted by caller
                mid  = mids[state["idx"] % len(mids)]
                state["idx"] += 1
                return mid
            app._routing_fn            = _round_robin_fn
            app._routing_strategy_name = "round_robin"
            strategy_desc = (
                "Round-robin  —  job 1→M01, job 2→M02, job 3→M03, job 4→M04, "
                "job 5→M01…  every machine gets every 4th job in strict rotation"
            )
        elif strategy == "sequential":
            # Sequential M01-first waterfall — same logic as the built-in
            # default in _route_dest_only(), installed as an explicit closure
            # so it can be selected after a different strategy was active.
            import config as _cfg_seq
            def _sequential_fn(eligible):
                for mid, m in eligible:   # eligible already sorted M01..M04
                    is_idle   = (getattr(m, "current_job", None) is None
                                 or getattr(m, "status", "") == "idle")
                    has_space = (len(m.queue)
                                 < _cfg_seq.SIM_MAX_AHEAD_PER_MACHINE)
                    if is_idle or has_space:
                        return mid
                return eligible[0][0]   # all full — overflow to M01
            app._routing_fn            = _sequential_fn
            app._routing_strategy_name = "sequential"
            strategy_desc = (
                "Sequential  —  M01-first waterfall: every new part tries M01 "
                "first (free or queue has space), then M02, M03, M04.  "
                "M01 absorbs all work it can; later machines receive overflow only."
            )
        elif strategy == "random":
            import random as _rnd
            def _random_fn(eligible):
                return _rnd.choice([mid for mid, _ in eligible])
            app._routing_fn            = _random_fn
            app._routing_strategy_name = "random"
            strategy_desc = "Random — uniform selection from enabled machines"
        else:
            # Unknown strategy — clear override, revert to built-in sequential default
            app._routing_fn            = None
            app._routing_strategy_name = "sequential"
            strategy_desc = f"Sequential (reverted from unknown strategy {strategy!r})"

        import config as _cfg
        from collections import Counter
        gui_done   = list(getattr(app, "_completed_parts", []))
        dist       = Counter(cp.get("machine", "") for cp in gui_done)
        tick       = self._tick()
        n_mach     = len(self._fa.agents)

        lines: list[tuple] = [
            ("factory",
             "🔄 ROUTING HOTSWAP  [" + prev_strategy + "  →  " + strategy + "]"),
            ("factory",
             "Strategy now active: " + strategy_desc + "\n"
             "Reasoning: " + reasoning + "\n\n"
             "This override lives in memory only.\n"
             "It is removed automatically when the program exits."),
        ]

        if dist:
            max_j  = max(dist.values())
            bar_w  = 20
            blines = ["Before hotswap — job distribution:"]
            for mid in sorted(dist):
                n   = dist[mid]
                bar = "█" * round(n / max_j * bar_w)
                blines.append("  " + mid + "  " + bar.ljust(bar_w) + "  " + str(n) + " jobs")
            lines.append(("factory", "\n".join(blines)))

        total_busy_ticks = sum(
            (cp.get("machining_s", 0) + cp.get("load_s", 0) + cp.get("unload_s", 0))
            / _cfg.T_TICK_S
            for cp in gui_done
        )
        current_util = (
            total_busy_ticks / max(1, tick * n_mach) * 100
            if tick > 0 else 0.0
        )
        # With round-robin, all machines share equally → utilisation approaches
        # total_busy / (tick * n_machines) but distributed evenly
        expected_util = min(98.0, current_util * n_mach / max(1, len(dist)))
        expected_each = len(gui_done) / max(1, n_mach)

        lines.append(("factory",
            "Expected effect after hotswap:\n"
            "  Each machine will receive ~" + f"{expected_each:.0f}" + " jobs"
            " (was: uneven — see distribution above)\n"
            "  Fleet utilisation target: " + f"{expected_util:.0f}" + "%"
            "  (current: " + f"{current_util:.0f}" + "%)\n\n"
            "To revert: type 'set routing shortest_queue' in this chat."))

        return lines


    # ── Prompt caching helpers ────────────────────────────────────────────────
    #
    # OpenAI automatically caches prompt prefixes ≥ 1024 tokens.
    # To maximise cache hits we split the context into two parts:
    #
    #   system message  — STATIC capabilities + format instructions
    #                     (identical on every call → cached after first request)
    #   user message    — DYNAMIC factory state + KPIs + question
    #                     (changes each call → never cached, always fresh)
    #
    # The static system message is ~1 000–1 400 tokens, which crosses the
    # 1 024-token threshold and becomes cache-eligible on OpenAI's infrastructure.
    # Cache hits are reported in usage.prompt_tokens_details.cached_tokens
    # (available in openai-python ≥ 1.40; silently ignored on older versions).

    def _build_static_system_prompt(self) -> str:
        """
        Static, never-changing system message.  Always sent first so it
        accumulates in OpenAI's prompt cache.  Do NOT include live data here.
        """
        from agent_intents import build_system_context
        parts = [
            "You are an AI advisor for an autonomous CNC factory with four machines "
            "(M01, M02, M03, M04).  Your job is to analyse factory state data and "
            "respond with structured JSON recommendations.",
            "",
            "=== OUTPUT FORMAT ===",
            "Always respond with a JSON object containing:",
            '  "recommendation"    — plain-text summary of what you advise (2–4 sentences)',
            '  "actions"           — list of intents to execute (may be empty [])',
            '  "policy_suggestion" — scheduling policy name, or null',
            '  "routing_strategy"  — null | "round_robin" | "random" | "shortest_queue"',
            '                        Set when job distribution is unbalanced.',
            '  "routing_reasoning" — one sentence explaining the routing choice',
            '  "reasoning"         — overall explanation (2–3 sentences)',
            "Do NOT include prose outside the JSON object.",
            "",
            build_factory_context_block(),
            "",
            build_system_context("M01…M04", include_examples=False),
        ]
        return "\n".join(parts)

    def _build_dynamic_user_context(self, question: str) -> str:
        """
        Dynamic factory snapshot — changes every call so it is NEVER cached.
        Kept small to reduce token spend on the un-cached portion.
        """
        kpis = self._fa.get_production_kpis()
        tick = getattr(self._fa.scheduler, "tick", 0)
        t_s  = getattr(self._fa.scheduler, "t_real_s", 0.0)

        parts = [
            f"=== LIVE FACTORY STATE  tick={tick}  sim={t_s/60:.1f}min  policy={self._fa.policy} ===",
            "",
            "KPIs:",
            f"  running={kpis['machines_running']}  idle={kpis['machines_idle']}  "
            f"stopped={kpis['machines_stopped']}",
            f"  completed={kpis['total_jobs_completed']}  rework={kpis['rework_queue_depth']}  "
            f"queue={kpis['scheduler_queue_depth']}",
            f"  errors={kpis['total_errors_processed']}  tools_replaced={kpis['total_tools_replaced']}  "
            f"avg_tool_life={kpis['avg_tool_life_pct']:.1f}%",
            "",
            "Machines:",
        ]
        for ag in self._fa.agents:
            s     = ag.get_state()
            tools = ag.tool_crib.state_list()
            crib  = "  ".join(
                f"{t['tool_id']}⌀{t['diameter_mm']:.0f}={t['remaining_life_pct']:.0f}%"
                + ("⛔" if t["needs_replacement"] else "")
                for t in tools
            )
            parts.append(
                f"  {ag.machine_id}  {ag.status:<22s}  "
                f"q={s['queue_depth']}  crib: {crib}"
            )

        # Recent tool events
        recent_tools = self._ledger.tool_events[-8:]
        if recent_tools:
            parts.append("")
            parts.append("Recent tool events:")
            for ev in recent_tools:
                wd_str = f"  W/D={ev.wd_ratio:.2f}" if ev.wd_ratio > 0 else ""
                parts.append(
                    f"  [{ev.tick}] {ev.machine_id} {ev.tool_id} "
                    f"{ev.event_type} life={ev.life_pct:.0f}%{wd_str}"
                )

        # Idle summary
        idle_sum = self._ledger.idle_summary(current_tick=tick)
        if idle_sum:
            parts.append("")
            parts.append("Idle summary:")
            for mid, d in sorted(idle_sum.items()):
                pct = (d["total_idle_ticks"] / tick * 100) if tick > 0 else 0.0
                parts.append(
                    f"  {mid}  idle={pct:.1f}%  periods={d['n_idle_periods']}  "
                    f"longest={d['longest_streak']}ticks"
                )

        # Policy changes
        if self._ledger.policy_changes:
            parts.append("")
            parts.append("Policy history:")
            for pc in self._ledger.policy_changes[-3:]:
                parts.append(f"  [{pc.tick}] {pc.old_policy} → {pc.new_policy}")

        # Completed parts (last 5)
        gui_done = list(getattr(getattr(self, "_app", None), "_completed_parts", []))
        if gui_done:
            parts.append("")
            parts.append(f"Completed parts (last 5 of {len(gui_done)}):")
            for cp in gui_done[-5:]:
                parts.append(
                    f"  Seed {cp.get('seed','?')} on {cp.get('machine','?')}  "
                    f"mach={cp.get('machining_s',0):.0f}s  total={cp.get('total_s',0):.0f}s"
                )

        parts += [
            "",
            f"=== OPERATOR QUESTION ===",
            question,
        ]
        return "\n".join(parts)

    def _compile_factory_context(self, question: str) -> str:
        """
        Legacy single-string context builder kept for non-OpenAI paths
        (fallback advice, logging).  For OpenAI calls use
        _build_static_system_prompt() + _build_dynamic_user_context().
        """
        return self._build_static_system_prompt() + "\n\n" + self._build_dynamic_user_context(question)

    def _call_openai(self, context: str, question: str) -> dict:
        """
        Call OpenAI with a prompt-cache-friendly split:
          system message = static capabilities (cached after first call)
          user   message = live state + question (fresh each call)

        Reports cache savings when the openai-python SDK exposes them
        (requires openai >= 1.40; gracefully degrades on older versions).
        """
        import json, time as _time

        system_msg = self._build_static_system_prompt()
        user_msg   = self._build_dynamic_user_context(question)

        t0 = _time.monotonic()
        try:
            comp = self._fa._llm_client.chat.completions.create(
                model       = config.OPENAI_MODEL,
                max_tokens  = 800,
                temperature = 0.3,
                messages    = [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
            )
            elapsed_ms = (_time.monotonic() - t0) * 1000

            raw   = comp.choices[0].message.content or "{}"
            clean = (raw.strip()
                       .lstrip("```json").lstrip("```")
                       .rstrip("```").strip())
            result = json.loads(clean)

            # ── Cache statistics (openai >= 1.40) ────────────────────────
            usage = getattr(comp, "usage", None)
            if usage:
                ptd           = getattr(usage, "prompt_tokens_details", None)
                cached_tokens = getattr(ptd, "cached_tokens", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                comp_tokens   = getattr(usage, "completion_tokens", 0)
                total_tokens  = getattr(usage, "total_tokens", 0)
                cache_info = (
                    f"  cached={cached_tokens}/{prompt_tokens} prompt tokens  "
                    f"({cached_tokens/max(1,prompt_tokens)*100:.0f}% hit)"
                    if cached_tokens is not None else
                    "  (cache stats unavailable — upgrade openai-python to ≥1.40)"
                )
                result["_cache_stats"] = (
                    f"tokens: prompt={prompt_tokens} cached={'?' if cached_tokens is None else cached_tokens} "
                    f"completion={comp_tokens} total={total_tokens}  "
                    f"latency={elapsed_ms:.0f}ms{cache_info}"
                )

            return result

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
        if "idle" in lower or "schedul" in lower or "routing" in lower:
            # Sequential M01-first waterfall naturally stacks work onto M01
            # leaving M03/M04 underutilised.  Detect via job-count imbalance.
            gui_done = list(getattr(getattr(self, "_app", None),
                                    "_completed_parts", []))
            from collections import Counter
            dist  = Counter(cp.get("machine", "") for cp in gui_done)

            if dist:
                max_j = max(dist.values())
                min_j = min(dist.values())
                # A machine with zero completions doesn't appear in dist;
                # count it as 0 jobs if we have 4 agents but fewer than 4
                # machines in dist.
                n_agents = len(self._fa.agents)
                if len(dist) < n_agents:
                    min_j = 0
                job_imbalance = (max_j / max(1, min_j)) if min_j > 0 else float("inf")

                if max_j > 0 and (job_imbalance >= 2.5 or min_j == 0):
                    busiest = max(dist, key=dist.get)
                    idlest  = (min(dist, key=dist.get)
                               if len(dist) == n_agents
                               else next(a.machine_id for a in self._fa.agents
                                         if a.machine_id not in dist))
                    imb_str = (f"{job_imbalance:.1f}×"
                               if job_imbalance != float("inf")
                               else "∞ (zero jobs)")
                    recs.append(
                        f"Job distribution is severely unbalanced: {busiest} ran "
                        f"{max_j} jobs vs {idlest} ran {dist.get(idlest, 0)} jobs "
                        f"(ratio {imb_str}).  "
                        f"This is caused by sequential M01-first routing — "
                        f"M01 absorbs every job it can, leaving later machines idle.  "
                        f"Round-robin will immediately distribute work evenly."
                    )
                    acts.append({
                        "target": "factory",
                        "intent": "factory_set_routing",
                        "parameters": {"routing_strategy": "round_robin"},
                        "reasoning": "Eliminates job starvation on M02/M03/M04.",
                    })
                    return {
                        "recommendation": "  ".join(recs),
                        "actions": acts,
                        "policy_suggestion": None,
                        "routing_strategy": "round_robin",
                        "routing_reasoning": (
                            f"Sequential M01-first routing gave {busiest} {max_j} jobs "
                            f"and {idlest} only {dist.get(idlest, 0)}.  "
                            f"Round-robin cycles M01→M02→M03→M04→M01 so every "
                            f"machine receives exactly every 4th job."
                        ),
                        "reasoning": "[Rule-based — job-count imbalance from sequential routing]",
                    }

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

    def _do_factory_tool_optimisation(self, p: dict) -> list[tuple]:
        """
        Demo arc — mirrors the routing hotswap:

          1. Compile width/diameter (W/D) ratio stats from all ledger tool-use events.
          2. Send current crib + W/D distribution to OpenAI.
          3. OpenAI returns three labelled options (A / B / C) — each a complete
             delta (add / replace / remove) against the current crib.
          4. Options are displayed and stored in  app._pending_tool_options.
          5. Operator types "implement option A"  →  _do_factory_implement_tool_option()
             patches every machine's in-memory ToolCrib objects.
          6. On program exit the mutations are garbage-collected.  No files touched.
        """
        import json, copy as _copy
        from collections import defaultdict, Counter

        question = p.get("question",
                         "Which tools should be in all machines for reduced machining time?")
        app_ref  = self._app

        # ── 1. Compile W/D ratio distribution ──────────────────────────────────
        use_events = [
            ev for ev in self._ledger.tool_events
            if ev.event_type == "used" and ev.wd_ratio > 0
        ]
        gui_done = list(getattr(app_ref, "_completed_parts", []))

        # Bucket edges and human labels
        EDGES  = [0.0, 1.5, 3.0, 6.0, float("inf")]
        LABELS = ["W/D < 1.5  (tight slot / hole)",
                  "W/D 1.5–3  (medium pocket)",
                  "W/D 3–6    (wide pocket)",
                  "W/D > 6    (large face)"]

        buckets: dict = defaultdict(list)   # bucket_idx → [{tool_id, dia_mm, wd}, …]
        for ev in use_events:
            for i, (lo, hi) in enumerate(zip(EDGES, EDGES[1:])):
                if lo <= ev.wd_ratio < hi:
                    buckets[i].append({"tool_id": ev.tool_id,
                                       "dia_mm":  ev.diameter_mm,
                                       "wd":      ev.wd_ratio})
                    break

        # ── 2. Current crib snapshot (all machines) ─────────────────────────────
        crib_now: list = []
        for ag in self._fa.agents:
            for t in ag.tool_crib.state_list():
                crib_now.append({
                    "machine":   ag.machine_id,
                    "tool_id":   t["tool_id"],
                    "dia_mm":    t["diameter_mm"],
                    "life_pct":  t["remaining_life_pct"],
                })

        # ── 3. Build stats summary lines (always shown) ─────────────────────────
        sum_lines = [
            "Tool W/D Ratio Distribution:",
            "═" * 58,
        ]
        bucket_summary = []
        for i, label in enumerate(LABELS):
            items = buckets.get(i, [])
            if items:
                cnt   = Counter(it["tool_id"] for it in items)
                best  = cnt.most_common(1)[0][0]
                dias  = [it["dia_mm"] for it in items]
                avg_d = sum(dias) / len(dias)
                wds   = [it["wd"] for it in items]
                avg_w = sum(wds) / len(wds)
                sum_lines.append(
                    f"  {label:<30}  {len(items):>5} uses  "
                    f"most-used: {best} (avg ⌀{avg_d:.0f}mm  avgW/D={avg_w:.2f})"
                )
                bucket_summary.append({
                    "range": label, "n": len(items),
                    "best_tool": best, "avg_dia": avg_d, "avg_wd": avg_w,
                    "tool_counts": dict(cnt),
                })
            else:
                sum_lines.append(f"  {label:<30}    (no data)")
                bucket_summary.append({"range": label, "n": 0})

        sum_lines += [
            "",
            "Current tool crib (per machine):",
        ]
        seen = set()
        for e in crib_now:
            key = (e["tool_id"], e["dia_mm"])
            if key not in seen:
                seen.add(key)
                sum_lines.append(f"  {e['tool_id']:>4}  ⌀{e['dia_mm']:>5.1f}mm")

        out: list[tuple] = [("factory", "\n".join(sum_lines))]

        if not bucket_summary or all(b["n"] == 0 for b in bucket_summary):
            out.append(("factory",
                "⚠ No W/D ratio data yet.\n"
                "  Run 'machining stats for seed <N>' first to generate per-face data,\n"
                "  then ask again once jobs have completed and tool events are recorded."))
            return out

        # ── 4. Ask OpenAI for 3 options ─────────────────────────────────────────
        system_prompt = (
            "You are a CNC process engineer.  Given width/diameter (W/D) ratio "
            "statistics from a 4-machine CNC factory and the current tool crib, "
            "propose exactly THREE alternative tool crib configurations "
            "(labelled A, B, C) that would reduce overall machining time.\n\n"
            "Rules:\n"
            "  • Each option must be a DELTA from the current crib: list only "
            "    tools to ADD, REPLACE, or REMOVE.\n"
            "  • Specify realistic CNC tool parameters.\n"
            "  • Apply changes to ALL machines unless stated otherwise.\n"
            "  • expected_gain_pct must be a realistic integer (5–35 typical).\n"
            "  • Keep each option to ≤ 4 changes.\n\n"
            "Respond with JSON only (no prose outside the object):\n"
            "{\n"
            '  "options": {\n'
            '    "A": {\n'
            '      "label":            "short name",\n'
            '      "reasoning":        "one sentence",\n'
            '      "expected_gain_pct": 12,\n'
            '      "changes": [\n'
            '        {"op":"replace","tool_id":"T1","diameter_mm":80.0,'
            '"n_inserts":6,"shank_length_mm":100.0,'
            '"max_life_hrs":30.0,"holder_cost_usd":450.0,"consumable_cost_usd":18.0},\n'
            '        {"op":"add",    "tool_id":"T11","diameter_mm":6.0,'
            '"n_inserts":2,"shank_length_mm":50.0,'
            '"max_life_hrs":20.0,"holder_cost_usd":120.0,"consumable_cost_usd":8.0},\n'
            '        {"op":"remove", "tool_id":"T5"}\n'
            "      ]\n"
            "    },\n"
            '    "B": { ... },\n'
            '    "C": { ... }\n'
            "  },\n"
            '  "recommendation": "A",\n'
            '  "reasoning": "two sentences"\n'
            "}"
        )
        user_msg = json.dumps({
            "wd_distribution": bucket_summary,
            "current_crib":    crib_now,
            "n_machines":      len(self._fa.agents),
            "policy":          self._fa.policy,
            "question":        question,
        }, indent=2)

        out.append(("factory",
            "Sending W/D data to OpenAI — requesting 3 tool crib options…"))

        ai_resp = None
        if self._fa._use_llm:
            try:
                import time as _time
                t0   = _time.monotonic()
                comp = self._fa._llm_client.chat.completions.create(
                    model       = config.OPENAI_MODEL,
                    max_tokens  = 900,
                    temperature = 0.25,
                    messages    = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_msg},
                    ],
                )
                elapsed = (_time.monotonic() - t0) * 1000
                raw    = comp.choices[0].message.content or "{}"
                clean  = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                ai_resp = json.loads(clean)

                # Cache stats (openai >= 1.40)
                usage = getattr(comp, "usage", None)
                ptd   = getattr(usage, "prompt_tokens_details", None) if usage else None
                cached = getattr(ptd, "cached_tokens", None)
                prompt_tok = getattr(usage, "prompt_tokens", 0)
                cache_note = (
                    f"  [{cached}/{prompt_tok} tokens cached  latency {elapsed:.0f}ms]"
                    if cached is not None else
                    f"  [latency {elapsed:.0f}ms]"
                )
                ai_resp["_cache_note"] = cache_note
            except Exception as exc:
                out.append(("factory", f"⚠ OpenAI error: {exc}\nFalling back to rule-based options."))

        # ── 5. Rule-based fallback options ─────────────────────────────────────
        if ai_resp is None:
            ai_resp = self._rule_based_tool_options(bucket_summary, crib_now)

        # ── 6. Display the three options ────────────────────────────────────────
        options_raw = ai_resp.get("options", {})
        rec         = ai_resp.get("recommendation", "A")
        cache_note  = ai_resp.get("_cache_note", "")

        opt_lines = [
            "",
            "OpenAI Tool Crib Options" + cache_note,
            "═" * 60,
            f"  Recommendation: Option {rec}",
            f"  Reasoning: {ai_resp.get('reasoning', '—')}",
            "",
        ]
        for letter, opt in options_raw.items():
            tag = "  ★ RECOMMENDED" if letter == rec else ""
            opt_lines += [
                f"┌── Option {letter}: {opt.get('label','?')} "
                f"(+{opt.get('expected_gain_pct','?')}% gain){tag}",
                f"│   {opt.get('reasoning','—')}",
                "│   Changes vs current crib:",
            ]
            for ch in opt.get("changes", []):
                op  = ch.get("op", "?")
                tid = ch.get("tool_id", "?")
                if op == "remove":
                    opt_lines.append(f"│     ✂  REMOVE  {tid}")
                elif op == "add":
                    opt_lines.append(
                        f"│     ➕ ADD     {tid}  ⌀{ch.get('diameter_mm','?')}mm  "
                        f"{ch.get('n_inserts','?')} inserts  "
                        f"life {ch.get('max_life_hrs','?')}h"
                    )
                else:  # replace
                    opt_lines.append(
                        f"│     🔄 REPLACE {tid} → ⌀{ch.get('diameter_mm','?')}mm  "
                        f"{ch.get('n_inserts','?')} inserts  "
                        f"life {ch.get('max_life_hrs','?')}h"
                    )
            opt_lines.append("└" + "─" * 55)

        opt_lines += [
            "",
            "These options exist in memory only — they will be deleted when the",
            "program exits.  No config files or tool databases are modified.",
            "",
            "To implement: type  'implement option A'  (or B or C)",
            "To discard:   type  'discard tool options'",
        ]
        out.append(("openai" if self._fa._use_llm else "factory",
                    "\n".join(opt_lines)))

        # ── 7. Store options on app for retrieval by implement handler ──────────
        if app_ref is not None:
            app_ref._pending_tool_options  = options_raw
            app_ref._tool_option_rec       = rec          # AI's recommendation letter
        return out

    def _rule_based_tool_options(self, bucket_summary: list, crib_now: list) -> dict:
        """
        Generate three plausible tool crib options without OpenAI.
        Logic: identify the most-used W/D range and propose tools
        optimised for that range across three risk levels.
        """
        # Find the busiest bucket
        busiest = max(bucket_summary, key=lambda b: b.get("n", 0), default={})
        avg_wd  = busiest.get("avg_wd", 3.0)

        # Heuristic: optimal tool diameter ≈ feature_width / optimal_wd_ratio
        # For speed: wd ~ 2.0 (half-width engagement)
        # For quality: wd ~ 1.2 (tight, more passes but better finish)
        # Balanced: wd ~ 1.6

        # Current max diameter in crib
        max_dia = max((e["dia_mm"] for e in crib_now), default=50.0)

        def _change(op, tid, dia=None, inserts=2, life=20.0, holder=200.0, cons=10.0, shank=75.0):
            c = {"op": op, "tool_id": tid}
            if op != "remove":
                c.update({"diameter_mm": dia, "n_inserts": inserts,
                           "shank_length_mm": shank, "max_life_hrs": life,
                           "holder_cost_usd": holder, "consumable_cost_usd": cons})
            return c

        options = {
            "A": {
                "label":             "High-throughput (speed priority)",
                "reasoning":         f"Larger tools reduce pass count for W/D avg={avg_wd:.1f}× — best when surface finish is secondary.",
                "expected_gain_pct": 18,
                "changes": [
                    _change("replace", "T1", dia=min(max_dia*1.2, 120), inserts=8, life=28, holder=480, cons=22, shank=80),
                    _change("replace", "T2", dia=min(max_dia*0.8, 80),  inserts=6, life=25, holder=380, cons=18, shank=75),
                    _change("add",     "T11", dia=8.0, inserts=2, life=18, holder=110, cons=7, shank=45),
                ],
            },
            "B": {
                "label":             "Balanced (time + finish)",
                "reasoning":         "Moderate tool upgrades with one extra small-dia tool for tight-slot coverage.",
                "expected_gain_pct": 10,
                "changes": [
                    _change("replace", "T2", dia=min(max_dia, 60), inserts=6, life=26, holder=360, cons=16, shank=75),
                    _change("add",     "T11", dia=6.0, inserts=2, life=20, holder=120, cons=8, shank=50),
                ],
            },
            "C": {
                "label":             "Tool-life priority (fewer replacements)",
                "reasoning":         "Larger inserts and better grades extend crib life, reducing downtime from tool changes.",
                "expected_gain_pct": 6,
                "changes": [
                    _change("replace", "T3", dia=30.0, inserts=4, life=40, holder=300, cons=14, shank=70),
                    _change("replace", "T4", dia=12.0, inserts=3, life=35, holder=180, cons=10, shank=55),
                ],
            },
        }
        return {"options": options, "recommendation": "A",
                "reasoning": "Rule-based: Option A gives highest throughput gain for your W/D distribution."}

    def _do_factory_implement_tool_option(self, p: dict) -> list[tuple]:
        """
        Apply a pending tool crib option (A / B / C) to every machine's
        in-memory ToolCrib.

        Pattern mirrors _hotswap_routing():
          • snapshot original cribs → app._original_tool_cribs
          • apply deltas (add / replace / remove) via ToolCrib API
          • record as PolicyChange in ledger
          • no files written; mutations GC-collected on program exit
        """
        import copy as _copy

        app_ref = self._app
        if app_ref is None:
            return [("factory", "⚠ No app reference — cannot apply option.")]

        # Which option did the operator choose?
        question = p.get("question", "").upper()
        option_letter = p.get("option_letter")  # set by classifier if present
        if not option_letter:
            for letter in ("A", "B", "C"):
                if letter in question:
                    option_letter = letter
                    break
        if not option_letter:
            return [("factory",
                "⚠ Please specify which option: 'implement option A', B, or C.")]

        pending = getattr(app_ref, "_pending_tool_options", {})
        if not pending:
            return [("factory",
                "⚠ No pending tool options found.\n"
                "  Ask for tool optimisation first: "
                "'which tools should we use to reduce machining time?'")]

        opt = pending.get(option_letter)
        if opt is None:
            available = ", ".join(sorted(pending.keys()))
            return [("factory",
                f"⚠ Option {option_letter!r} not found.  "
                f"Available: {available}")]

        label   = opt.get("label", option_letter)
        changes = opt.get("changes", [])
        gain    = opt.get("expected_gain_pct", "?")

        # ── Snapshot current cribs (for log / audit; real revert = restart) ───
        original: dict = {}
        for ag in self._fa.agents:
            original[ag.machine_id] = {
                tid: _copy.copy(tr)
                for tid, tr in ag.tool_crib._tools.items()
            }
        if app_ref is not None:
            app_ref._original_tool_cribs = original

        # ── Apply changes to every machine ────────────────────────────────────
        from cnc_agent import ToolRecord
        import config as _cfg

        applied: list[str] = []
        errors:  list[str] = []

        for ag in self._fa.agents:
            crib = ag.tool_crib
            for ch in changes:
                op  = ch.get("op",      "?")
                tid = ch.get("tool_id", "?")
                try:
                    if op == "remove":
                        removed = crib.remove_tool(tid)
                        if removed:
                            applied.append(f"  ✂  {ag.machine_id}: removed {tid}")
                        else:
                            errors.append(f"  ⚠  {ag.machine_id}: {tid} not found (skip remove)")

                    elif op in ("add", "replace"):
                        if op == "replace":
                            crib.remove_tool(tid)   # silently ok if missing

                        new_tr = ToolRecord(
                            tool_id              = tid,
                            n_inserts            = int(ch.get("n_inserts", 4)),
                            diameter_mm          = float(ch.get("diameter_mm", 10.0)),
                            shank_length_mm      = float(ch.get("shank_length_mm", 75.0)),
                            max_life_hrs         = float(ch.get("max_life_hrs", 20.0)),
                            remaining_life_hrs   = float(ch.get("max_life_hrs", 20.0)),
                            holder_cost_usd      = float(ch.get("holder_cost_usd", 200.0)),
                            consumable_cost_usd  = float(ch.get("consumable_cost_usd", 10.0)),
                            requires_coolant     = ch.get("requires_coolant", True),
                        )
                        crib.add_tool(new_tr)
                        verb = "replaced" if op == "replace" else "added"
                        applied.append(
                            f"  {'🔄' if op=='replace' else '➕'}  {ag.machine_id}: "
                            f"{verb} {tid} ⌀{new_tr.diameter_mm:.0f}mm"
                        )
                    else:
                        errors.append(f"  ⚠  unknown op {op!r} for {tid} — skipped")

                except Exception as exc:
                    errors.append(f"  ⚠  {ag.machine_id} / {tid}: {exc}")

        # Deduplicate applied lines (same change × 4 machines → collapse to one)
        from collections import Counter
        collapsed: list[str] = []
        seen_changes: Counter = Counter()
        for line in applied:
            # Strip machine prefix to detect duplicates
            parts = line.split(":", 1)
            key   = parts[1].strip() if len(parts) == 2 else line
            seen_changes[key] += 1
        done_keys: set = set()
        for line in applied:
            parts = line.split(":", 1)
            key   = parts[1].strip() if len(parts) == 2 else line
            if key not in done_keys:
                n = seen_changes[key]
                icon = line.split()[1]
                collapsed.append(f"  {icon}  ALL machines: {key}  (×{n})")
                done_keys.add(key)

        # ── Record in ledger ──────────────────────────────────────────────────
        tick = self._tick()
        self._ledger.record_policy_change(
            old_policy = f"tool_crib:default",
            new_policy = f"tool_crib:option_{option_letter}:{label}",
            tick       = tick,
            reason     = f"operator-selected tool option {option_letter}",
        )

        # Mark active option on app
        if app_ref is not None:
            app_ref._active_tool_option        = option_letter
            app_ref._active_tool_option_label  = label

        # ── Build response ────────────────────────────────────────────────────
        result_lines = [
            f"🔧 TOOL CRIB HOTSWAP — Option {option_letter}: {label}",
            "═" * 58,
            f"  Expected machining time reduction: ~{gain}%",
            "",
            "Changes applied to all machines:",
        ] + (collapsed if collapsed else ["  (no changes)"])

        if errors:
            result_lines += ["", "Warnings:"] + errors

        result_lines += [
            "",
            "New crib state (all machines):",
            f"  {'Tool':>5}  {'⌀ mm':>6}  {'Inserts':>8}  {'Life hrs':>9}",
            "  " + "─" * 34,
        ]
        # Show merged crib (same across all machines after apply)
        sample_ag = self._fa.agents[0] if self._fa.agents else None
        if sample_ag:
            for t in sample_ag.tool_crib.state_list():
                result_lines.append(
                    f"  {t['tool_id']:>5}  {t['diameter_mm']:>6.1f}  "
                    f"  {'—':>6}    {t['remaining_life_pct']/100 * sample_ag.tool_crib.get(t['tool_id']).max_life_hrs if sample_ag.tool_crib.get(t['tool_id']) else 0:>7.1f}h"
                )

        result_lines += [
            "",
            "This change lives in memory only.",
            "It is removed automatically when the program exits.",
            "Original crib snapshot saved — restart to revert.",
        ]

        return [("factory", "\n".join(result_lines))]

    def _do_unknown(self, p: dict) -> list[tuple]:
        return [("factory",
                 "Factory action not recognised — please rephrase.")]

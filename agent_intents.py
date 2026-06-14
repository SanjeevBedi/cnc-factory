"""
agent_intents.py  —  Canonical intent catalogue for a CNC machine agent.

Every action the agent can perform is declared here as an AgentIntent.
The catalogue is the single source of truth for:
  • LLM system-prompt context  (what the model may instruct the agent to do)
  • Rule-based fallback engine  (maps disturbance → allowed intents)
  • GUI intent-suggestion list  (shown in the chat entry bar)
  • Validation gate             (reject LLM responses that request unknown intents)

Structure
─────────
  INTENT_CATALOGUE  — ordered list[AgentIntent]
  INTENT_MAP        — dict[str, AgentIntent]  (keyed by intent_id)
  INTENT_GROUPS     — dict[str, list[str]]    groups of related intents

  build_system_context()    → str          full capability block for system prompt
  build_suggestions_block() → str          scoped suggestions for a disturbance
  intent_suggestions()      → list[str]    flat list for GUI autocomplete
  allowed_intents()         → list[AgentIntent]  filter by category / safety flag
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Intent dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentIntent:
    intent_id:       str
    category:        str   # execution | tool | program | query | schedule | safety
    label:           str
    description:     str
    parameters:      tuple
    requires_stop:   bool = False
    safety_critical: bool = False
    reversible:      bool = True
    example:         str  = ""


# ── Catalogue ─────────────────────────────────────────────────────────────────

INTENT_CATALOGUE: list[AgentIntent] = [

    # EXECUTION
    AgentIntent("continue",        "execution", "Continue machining",
        "Resume the current G-code program without changes.",
        (), example='{"intent":"continue"}'),

    AgentIntent("pause",           "execution", "Pause program",
        "Suspend at the next safe retract position; spindle stays on.",
        ("reason",), example='{"intent":"pause","reason":"dimensional check"}'),

    AgentIntent("resume",          "execution", "Resume from pause",
        "Re-start a paused program from the line where it stopped.",
        (), example='{"intent":"resume"}'),

    AgentIntent("abort",           "execution", "Abort job",
        "Stop immediately; spindle and coolant off; axes retract; part to rework.",
        ("reason",), requires_stop=True, reversible=False, safety_critical=True,
        example='{"intent":"abort","reason":"tool broken"}'),

    AgentIntent("emergency_stop",  "execution", "Emergency stop",
        "Full immediate stop: spindle off, feed hold, coolant off, all axes halt. "
        "Requires manual reset.",
        ("reason",), requires_stop=True, reversible=False, safety_critical=True,
        example='{"intent":"emergency_stop","reason":"fixture loose"}'),

    AgentIntent("skip_feature",    "execution", "Skip current feature",
        "Jump to the next feature, leaving the current one unmachined.",
        ("feature_id",), example='{"intent":"skip_feature","feature_id":"Face 2"}'),

    # FEED & SPEED
    AgentIntent("reduce_feed",     "execution", "Reduce feed rate",
        "Apply a percentage feed-rate override to reduce cutting forces.",
        ("feed_override_pct",),
        example='{"intent":"reduce_feed","feed_override_pct":75}'),

    AgentIntent("increase_feed",   "execution", "Increase feed rate",
        "Apply a positive override when material is softer than specified.",
        ("feed_override_pct",),
        example='{"intent":"increase_feed","feed_override_pct":115}'),

    AgentIntent("change_rpm",      "execution", "Change spindle RPM",
        "Override programmed spindle speed for chatter avoidance or thermal management.",
        ("rpm",), example='{"intent":"change_rpm","rpm":3400}'),

    AgentIntent("reduce_doc",      "execution", "Reduce depth of cut",
        "Reduce axial depth of cut; triggers toolpath regeneration for remaining features.",
        ("depth_of_cut_mm",),
        example='{"intent":"reduce_doc","depth_of_cut_mm":0.3}'),

    AgentIntent("recalculate",     "execution", "Recalculate feeds & speeds",
        "Recompute optimal feed, RPM, and DoC for a different material or tool; "
        "continues with regenerated toolpath.",
        ("material", "tool_id", "feed_rate_mmpm", "rpm", "depth_of_cut_mm"),
        example='{"intent":"recalculate","material":"alloy_steel_4140",'
                '"tool_id":"T3","feed_rate_mmpm":1800,"rpm":2200,"depth_of_cut_mm":0.4}'),

    AgentIntent("add_finish_pass", "execution", "Add finishing pass",
        "Append one light finishing pass to recover surface finish.",
        ("feed_rate_mmpm", "step_over_pct"),
        example='{"intent":"add_finish_pass","feed_rate_mmpm":800,"step_over_pct":10}'),

    AgentIntent("add_rough_pass",  "execution", "Add roughing pass",
        "Insert an additional roughing pass to remove excess stock.",
        ("depth_of_cut_mm", "feed_rate_mmpm"),
        example='{"intent":"add_rough_pass","depth_of_cut_mm":1.5,"feed_rate_mmpm":2500}'),

    AgentIntent("change_path",     "execution", "Change toolpath strategy",
        "Switch to an alternative path strategy (e.g. trochoidal) to reduce chatter.",
        ("strategy",), example='{"intent":"change_path","strategy":"trochoidal"}'),

    # TOOL
    AgentIntent("change_tool",     "tool", "Change tool",
        "Retract to tool-change position, swap to specified tool from crib. "
        "Costs TOOL_CHANGE_TIME_S (150 s) downtime.",
        ("tool_id",), requires_stop=True,
        example='{"intent":"change_tool","tool_id":"T4"}'),

    AgentIntent("query_tool_life", "tool", "Query tool life",
        "Return remaining life %, hours, and wear status for every tool in the crib.",
        ("tool_id",), example='{"intent":"query_tool_life"}'),

    AgentIntent("query_tool_crib", "tool", "Query full tool crib",
        "Return complete crib details: ID, diameter, inserts, life, cost, replacement status.",
        (), example='{"intent":"query_tool_crib"}'),

    AgentIntent("set_tool_override", "tool", "Override tool selection",
        "Force a specific tool for the next feature, bypassing automatic selection.",
        ("tool_id", "feature_type"),
        example='{"intent":"set_tool_override","tool_id":"T6","feature_type":"face"}'),

    # PROGRAM
    AgentIntent("load_program",    "program", "Load G-code program",
        "Load a program by seed ID or file path into the machine buffer.",
        ("seed", "filepath"), example='{"intent":"load_program","seed":1042}'),

    AgentIntent("unload_program",  "program", "Unload program",
        "Clear the current program from the machine buffer; part stays on table.",
        (), example='{"intent":"unload_program"}'),

    AgentIntent("query_program",   "program", "Query current program",
        "Return the loaded program: seed, line count, cursor, face labels, "
        "and estimated remaining time.",
        (), example='{"intent":"query_program"}'),

    AgentIntent("query_execution_history", "program", "Query execution history",
        "Return last N programs executed: seed, material, machining time, cost, errors.",
        ("n",), example='{"intent":"query_execution_history","n":5}'),

    AgentIntent("replan_remaining","program", "Replan remaining features",
        "Regenerate toolpath and G-code for unmachined features using the current crib. "
        "Completed faces (identified by FACE_END sentinels) are preserved.",
        ("tool_id",),
        example='{"intent":"replan_remaining","tool_id":"T5"}'),

    AgentIntent("query_current_part",  "program", "Query current part being machined",
        "Return seed ID, material, tool in use, current section, completed sections, "
        "G-code progress bar, and estimated time for the part on the table.",
        (), example='{"intent":"query_current_part"}'),

    AgentIntent("query_gcode_current", "program", "Query current G-code line",
        "Return the G-code instruction the machine is executing right now.",
        (), example='{"intent":"query_gcode_current"}'),

    AgentIntent("query_gcode_history", "program", "Query last N G-code lines executed",
        "Return the last N G-code lines that have already been executed.",
        ("n",), example='{"intent":"query_gcode_history","n":5}'),

    AgentIntent("query_gcode_upcoming","program", "Query next N G-code lines",
        "Return the next N G-code lines that are about to be executed.",
        ("n",), example='{"intent":"query_gcode_upcoming","n":5}'),

    AgentIntent("query_gcode_full",    "program", "Show complete G-code program",
        "Return the full G-code listing. Programs >30 lines are summarised.",
        (), example='{"intent":"query_gcode_full"}'),

    # QUERY
    AgentIntent("query_status",    "query", "Query machine status",
        "Return full machine state: status, current job, section, queue depth, "
        "rework count, active error, pending tool-change flag.",
        (), example='{"intent":"query_status"}'),

    AgentIntent("query_cycle_time","query", "Query cycle time",
        "Return estimated and actual time breakdown: setup, cutting, tool-change, unclamp.",
        ("job_id",), example='{"intent":"query_cycle_time"}'),

    AgentIntent("query_cost",      "query", "Query job cost",
        "Return cost breakdown: machine amortisation, tool consumable, power, total.",
        ("job_id",), example='{"intent":"query_cost"}'),

    AgentIntent("query_power",     "query", "Query spindle power",
        "Return current estimated spindle power draw vs machine limit.",
        (), example='{"intent":"query_power"}'),

    AgentIntent("query_feed_rpm",  "query", "Query feed & RPM",
        "Return active feed rate, RPM, depth of cut, step-over, and active overrides.",
        (), example='{"intent":"query_feed_rpm"}'),

    AgentIntent("query_material",  "query", "Query material",
        "Return material assigned to current job: name, Kc, and chip-load parameters.",
        (), example='{"intent":"query_material"}'),

    AgentIntent("query_queue",     "query", "Query job queue",
        "Return queued jobs: seed IDs, estimated times, materials, tool requirements.",
        (), example='{"intent":"query_queue"}'),

    AgentIntent("query_error_log", "query", "Query error log",
        "Return active error and last N events: description, section, G-code line, resolution.",
        ("n",), example='{"intent":"query_error_log","n":3}'),

    # SCHEDULE
    AgentIntent("rework",          "schedule", "Move part to rework",
        "Mark current job as rework, add to rework queue, free machine for next job.",
        ("reason",), reversible=False,
        example='{"intent":"rework","reason":"dimension out of tolerance"}'),

    AgentIntent("transfer",        "schedule", "Transfer job to another machine",
        "Move current or queued job to a different machine; triggers routing negotiation.",
        ("target_machine_id", "job_id"),
        example='{"intent":"transfer","target_machine_id":"M03","job_id":"J-042"}'),

    AgentIntent("reorder_queue",   "schedule", "Reorder job queue",
        "Change processing order of queued jobs by priority, time, or material.",
        ("order_by",), example='{"intent":"reorder_queue","order_by":"priority"}'),

    AgentIntent("continue_then_reorder", "schedule", "Continue then reorder",
        "Finish current job uninterrupted; flag queue for reorder before next job.",
        ("reason",),
        example='{"intent":"continue_then_reorder","reason":"priority change received"}'),

    AgentIntent("pause_and_inspect","schedule", "Pause for quality inspection",
        "Pause at current retract position; notify QC for dimensional check.",
        ("feature_id", "dimension_mm", "tolerance_mm"),
        example='{"intent":"pause_and_inspect","feature_id":"Face 1",'
                '"dimension_mm":45.0,"tolerance_mm":0.05}'),

    AgentIntent("wait_and_resume", "schedule", "Wait then resume",
        "Hold in safe state for N ticks then auto-resume. "
        "Use for coolant purge, fixture settle, thermal stabilisation.",
        ("wait_ticks", "reason"),
        example='{"intent":"wait_and_resume","wait_ticks":6,"reason":"coolant purge"}'),

    # SAFETY
    AgentIntent("manual_intervention", "safety", "Request manual intervention",
        "Flag machine as requiring a human operator; pause and notify supervisor. "
        "No auto actions until operator clears condition.",
        ("reason",), requires_stop=True, safety_critical=True,
        example='{"intent":"manual_intervention","reason":"fixture vibration detected"}'),

    AgentIntent("continue_dry",    "safety", "Continue in dry-run mode",
        "Re-run remaining program without coolant. "
        "Permitted only for light finishing cuts in aluminium; not steel/Ti/Inconel.",
        ("reason",),
        example='{"intent":"continue_dry","reason":"coolant line blocked"}'),

    AgentIntent("skip_tool_change","safety", "Skip scheduled tool change",
        "Bypass pending tool-change and continue with worn tool. "
        "Permitted only if life > STOP_PCT and operator explicitly accepts increased risk.",
        ("tool_id", "accepted_risk"), safety_critical=True,
        example='{"intent":"skip_tool_change","tool_id":"T2","accepted_risk":true}'),

    AgentIntent("notify_factory",  "safety", "Notify factory supervisor",
        "Send a structured alert to the factory supervisor; machine continues unless "
        "separately instructed to pause.",
        ("message", "urgency"),
        example='{"intent":"notify_factory","message":"delivery delay on M03",'
                '"urgency":"normal"}'),
]


# ── Derived lookups ───────────────────────────────────────────────────────────

INTENT_MAP:    dict[str, AgentIntent] = {i.intent_id: i for i in INTENT_CATALOGUE}
INTENT_GROUPS: dict[str, list[str]]   = {}
for _i in INTENT_CATALOGUE:
    INTENT_GROUPS.setdefault(_i.category, []).append(_i.intent_id)


# ── Public helpers ────────────────────────────────────────────────────────────

def allowed_intents(
    category:               Optional[str] = None,
    exclude_safety_critical: bool = False,
    exclude_requires_stop:   bool = False,
) -> list[AgentIntent]:
    result = list(INTENT_CATALOGUE)
    if category:
        result = [i for i in result if i.category == category]
    if exclude_safety_critical:
        result = [i for i in result if not i.safety_critical]
    if exclude_requires_stop:
        result = [i for i in result if not i.requires_stop]
    return result


def intent_suggestions() -> list[str]:
    """Flat list of intent_ids for GUI autocomplete / suggestion chips."""
    return [i.intent_id for i in INTENT_CATALOGUE]


def build_system_context(
    machine_id:           str  = "M??",
    include_examples:     bool = True,
    include_safety_notes: bool = True,
) -> str:
    """
    Build the capability block prepended to every LLM system prompt.
    Tells the model what the agent can do and how to request each action.
    """
    lines: list[str] = [
        f"=== MACHINE AGENT CAPABILITIES  ({machine_id}) ===",
        "",
        f"You are the factory supervisor AI.  The CNC machine agent {machine_id} "
        "can execute the following intents.",
        "Your response MUST use 'intent' as the top-level action key and MUST "
        "choose an intent_id from this list.  Do NOT invent intent names.",
        "",
    ]

    cat_order = ["execution", "tool", "program", "query", "schedule", "safety"]
    cat_label = {
        "execution": "EXECUTION — control the machining operation",
        "tool":      "TOOL — manage tools and the crib",
        "program":   "PROGRAM — load, inspect, and manage G-code",
        "query":     "QUERY — read machine state and production data",
        "schedule":  "SCHEDULE — job queue and routing",
        "safety":    "SAFETY — protective and human-escalation actions",
    }

    for cat in cat_order:
        cat_intents = [i for i in INTENT_CATALOGUE if i.category == cat]
        if not cat_intents:
            continue
        lines.append(f"── {cat_label[cat]} ──")
        for intent in cat_intents:
            flags = []
            if intent.requires_stop:    flags.append("spindle-stop required")
            if intent.safety_critical:  flags.append("⚠ safety-critical")
            if not intent.reversible:   flags.append("irreversible")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            lines.append(f"  {intent.intent_id:<30s} {intent.label}{flag_str}")
            lines.append(f"    {intent.description}")
            if intent.parameters:
                lines.append(f"    Parameters : {', '.join(intent.parameters)}")
            if include_examples and intent.example:
                lines.append(f"    Example    : {intent.example}")
            lines.append("")

    if include_safety_notes:
        lines += [
            "── SAFETY RULES ──",
            "  1. Never emit emergency_stop or abort without a concrete technical reason.",
            "  2. ⚠ safety-critical intents require 'accepted_risk':true in parameters.",
            "  3. Prefer the least-disruptive intent that resolves the condition.",
            "  4. If the intent requires spindle-stop, include 'stop_spindle_first':true.",
            "",
        ]

    lines += [
        "── RESPONSE FORMAT ──",
        '  {"intent"     : "<intent_id from list above>",',
        '   "parameters" : { <key>:<value>, ... },',
        '   "risk_level" : "safe" | "risky" | "catastrophic",',
        '   "reasoning"  : "<one concise sentence>"}',
        "",
        "=== END OF AGENT CAPABILITIES ===",
    ]
    return "\n".join(lines)


def build_suggestions_block(
    disturbance_key:  Optional[str]          = None,
    possible_actions: Optional[tuple]        = None,
) -> str:
    """
    Build a short SUGGESTED INTENTS block for the LLM prompt body.
    Scoped to the disturbance type when possible_actions is provided.
    """
    if possible_actions:
        suggested = [INTENT_MAP[a] for a in possible_actions if a in INTENT_MAP]
    else:
        suggested = allowed_intents(
            category="execution", exclude_safety_critical=True)

    if not suggested:
        return ""

    lines = ["SUGGESTED INTENTS FOR THIS DISTURBANCE:"]
    for i in suggested:
        lines.append(f"  {i.intent_id:<30s} — {i.description[:80]}")
    return "\n".join(lines)


# ── Message classification ────────────────────────────────────────────────────

# Keyword sets that map operator text → likely query intent
_QUERY_KEYWORDS: dict[str, list[str]] = {
    # tool crib / specific tool property
    "query_tool_crib":          ["tool crib", "tools in crib", "what tools", "list tools",
                                  "all tools", "show tools", "available tools",
                                  "what is in", "what's in", "in the crib",
                                  "crib contents"],
    "query_status":             ["status", "machine state",
                                  "what is the machine", "what's happening",
                                  "current state", "is the machine"],
    "query_tool_life":          ["tool life", "life remaining",
                                  "how much life", "tool condition", "life pct",
                                  "how worn"],
    "query_tool_detail":        ["radius", "diameter", "shank", "flute",
                                  "flutes", "tool t", "details of t",
                                  "show t", "tell me about t",
                                  "spec of", "specification",
                                  "properties of"],
    "query_queue":              ["queue", "queued jobs", "jobs waiting", "backlog",
                                  "what jobs", "next job", "how many jobs"],
    "query_cycle_time":         ["cycle time", "machining time",
                                  "time for job", "estimated time", "how fast"],
    "query_cost":               ["cost", "how much does", "job cost", "total cost",
                                  "price", "usd"],
    "query_power":              ["power", "spindle load", "kw", "watt", "power draw"],
    "query_feed_rpm":           ["feed rate", "rpm", "spindle speed", "cutting speed",
                                  "feed and speed", "feeds and speeds", "feedrate",
                                  "what speed", "what feed"],
    "query_material":           ["material", "what material", "workpiece material",
                                  "what alloy", "what grade"],
    # G-code / live execution  (must come BEFORE query_program so more-specific
    # keywords shadow the generic "g-code" / "gcode" entries)
    "query_current_part":       ["what part", "which part", "what are you machining",
                                  "what are you cutting", "current part", "part being",
                                  "what are you working on", "what job are you",
                                  "what seed are you", "tell me about the part",
                                  "what is on the machine"],
    "query_gcode_current":      ["current g-code", "current gcode", "current line",
                                  "what line are you", "g-code you are running",
                                  "gcode you are running", "running right now",
                                  "what g-code are you", "what gcode are you",
                                  "current instruction", "executing right now",
                                  "g-code is running", "gcode is running"],
    "query_gcode_history":      ["last g-code", "last gcode", "last few g",
                                  "last few lines", "previously ran", "g-code you ran",
                                  "gcode history", "g-code history",
                                  "last 5 g", "last 3 g", "last 10 g",
                                  "g-codes you executed", "gcodes you executed",
                                  "what did you run", "last line you ran"],
    "query_gcode_upcoming":     ["next g-code", "next gcode", "next line",
                                  "next few g", "next few lines", "upcoming g",
                                  "about to execute", "about to run",
                                  "going to run", "going to execute",
                                  "next 5 g", "next 3 g", "what comes next",
                                  "what is next to run"],
    "query_gcode_full":         ["g-code of the part", "gcode of the part",
                                  "show me the g-code", "full g-code", "all g-code",
                                  "complete g-code", "entire g-code",
                                  "list the gcode", "g-code for the part",
                                  "gcode for the part"],
    "query_program":            ["what program", "which program", "what is loaded",
                                  "what program is loaded"],
    "query_execution_history":  ["last jobs", "previous jobs", "job history", "past jobs"],
    "query_error_log":          ["error log", "fault", "alarms", "error history",
                                  "what errors", "what went wrong"],
}

# Words that signal a disturbance / action request rather than a query
import re as _re
# Pattern that matches T1, T2, … T99 as a standalone token
_TOOL_ID_RE = _re.compile(r'\bT(\d{1,2})\b', _re.IGNORECASE)

_ACTION_SIGNALS: list[str] = [
    "abort", "stop", "emergency",
    "broken", "break", "broke", "snapped", "shattered", "fail",
    "chatter", "vibration", "resonance",
    "overload", "dimension", "rework",
    "change tool", "reduce feed", "increase", "recalculate",
    "problem", "issue", "wrong", "error", "fault",
    "incorrect material", "material mismatch", "wrong material",
    "not cutting", "poor finish", "wear", "worn",
]


def classify_message(text: str) -> tuple[str, Optional[str]]:
    """
    Classify operator free text as:
      ("query",  intent_id)   — a data query; execute directly
      ("action", None)        — a disturbance / action request; route to LLM pipeline

    Priority:
      1. Action-signal words short-circuit to ("action", None).
      2. Phrase scan  — exact keyword phrases from _QUERY_KEYWORDS.
      3. Token scan   — presence of semantic word sets regardless of order.
      4. Bare tool-ID (T1…T10) with no action signals → tool detail query.
      5. Intent_id word in text.
      6. Default: ("action", None).
    """
    lower = text.lower()
    # normalise "gcode" / "g code" → "g-code" for uniform matching
    lower = lower.replace("gcode", "g-code").replace("g code", "g-code")

    # 1. Action signals override everything
    for sig in _ACTION_SIGNALS:
        if sig in lower:
            return ("action", None)

    # 2. Phrase scan
    for intent_id, keywords in _QUERY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return ("query", intent_id)

    # 3. Token-combination fallback — word-order independent
    result = _token_classify(lower)
    if result:
        return ("query", result)

    # 4. Bare tool-ID
    if _TOOL_ID_RE.search(text):
        return ("query", "query_tool_detail")

    # 5. Exact intent_id word
    for intent_id in INTENT_MAP:
        if intent_id.replace("_", " ") in lower or intent_id in lower:
            if INTENT_MAP[intent_id].category == "query":
                return ("query", intent_id)

    return ("action", None)


def _token_classify(lower: str) -> Optional[str]:
    """
    Word-token based classification — fires when phrase matching fails.
    Checks for the co-presence of semantic word groups regardless of order.
    Returns an intent_id or None.
    """
    # helpers
    def has(words):
        return any(w in lower for w in words)

    GCODE     = ["g-code", "gcode"]
    QUESTION  = ["what", "which", "show", "tell", "give", "list",
                 "display", "print", "dump", "is it", "are you"]
    PART_HINT = ["part", "job", "seed", "machining", "cutting",
                 "working on", "being made", "being cut", "being machined"]
    HIST_HINT = ["last", "previous", "ran", "history", "did", "just ran",
                 "executed", "already ran", "recently"]
    NEXT_HINT = ["next", "upcoming", "will", "going to", "about to",
                 "future", "after this", "execute next", "run next",
                 "following", "then execute"]
    FULL_HINT = ["all", "full", "complete", "entire", "whole",
                 "listing", "list", "print", "dump", "give me", "show me all",
                 "show all"]
    CUR_HINT  = ["running", "executing", "current", "now", "active",
                 "right now", "at the moment", "currently", "is it",
                 "are you running", "is running", "is executing",
                 "you running", "you executing"]
    STAT_HINT = ["status", "state", "what is the machine",
                 "machine doing", "happening"]
    TOOL_HINT = ["tool", "insert", "crib", "wear", "life",
                 "diameter", "radius", "shank"]
    QUEUE_HINT= ["queue", "queued", "backlog", "waiting", "jobs"]

    # G-code queries
    if has(GCODE):
        if has(HIST_HINT):
            return "query_gcode_history"
        if has(NEXT_HINT):
            return "query_gcode_upcoming"
        if has(FULL_HINT):
            return "query_gcode_full"
        if has(CUR_HINT):
            # explicitly "running / executing / current / now" → current line
            return "query_gcode_current"
        # question word alone (show / what / give / list) with no
        # qualifier → full listing
        return "query_gcode_full"

    # Part / job queries
    if has(PART_HINT) and has(QUESTION + CUR_HINT):
        return "query_current_part"

    # "what will execute next" / "what executes next" — no g-code word but clear intent
    if has(["execute next", "executing next", "run next", "runs next"]):
        return "query_gcode_upcoming"

    # Generic "what is executing / running" without g-code
    if has(["executing", "running", "being executed"]) and has(QUESTION):
        return "query_gcode_current"

    # Machine status
    if has(STAT_HINT) and has(QUESTION):
        return "query_status"

    # Tool / crib (broad catch)
    if has(TOOL_HINT) and has(QUESTION):
        return "query_tool_crib"

    # Queue
    if has(QUEUE_HINT) and has(QUESTION):
        return "query_queue"

    return None


def extract_parameters(text: str, intent_id: str) -> dict:
    """
    Extract structured parameters from free-text for the given intent.
    Used so handlers can answer questions like 'radius of T1' precisely.
    """
    params: dict = {}
    # Tool ID: T1 … T99
    m = _TOOL_ID_RE.search(text)
    if m:
        params["tool_id"] = f"T{m.group(1)}"
    # Number of items (for history / error log)
    n_m = _re.search(r'\blast\s+(\d+)\b|\b(\d+)\s+(?:jobs|items|events)\b',
                     text, _re.IGNORECASE)
    if n_m:
        params["n"] = int(next(v for v in n_m.groups() if v))
    return params


# ── Intent executor ───────────────────────────────────────────────────────────

class IntentExecutor:
    """
    Execute query intents directly against the live agent and factory state.
    Returns a dict with:
      intent   : intent_id
      result   : the data (formatted as a human-readable string)
      raw      : raw Python object for programmatic use
    """

    def __init__(self, agent, factory_agent):
        self._agent = agent
        self._fa    = factory_agent

    def execute(self, intent_id: str, parameters: dict = None) -> dict:
        parameters = parameters or {}
        handler = getattr(self, f"_do_{intent_id}", self._do_unknown)
        return handler(parameters)

    # ── query handlers ────────────────────────────────────────────────────────

    def _do_query_tool_crib(self, p: dict) -> dict:
        tools = self._agent.tool_crib.state_list()
        lines = [f"Tool crib — {self._agent.machine_id}  ({len(tools)} tools):",
                 f"  {'ID':<5} {'Dia':>6}  {'Inserts':>7}  {'Life':>6}  {'Status'}"]
        lines.append("  " + "─" * 48)
        for t in tools:
            life = t.get("remaining_life_pct", 0)
            status = ("⛔ STOP" if t.get("needs_replacement")
                      else ("⚠ WARN" if life < 20 else "✓ OK"))
            lines.append(
                f"  {t['tool_id']:<5} {t['diameter_mm']:>5.0f}mm  "
                f"{t['n_inserts']:>7}  {life:>5.1f}%  {status}"
            )
        return {"intent": "query_tool_crib", "result": "\n".join(lines), "raw": tools}

    def _do_query_tool_life(self, p: dict) -> dict:
        tools = self._agent.tool_crib.state_list()
        tool_id = p.get("tool_id")
        if tool_id:
            tools = [t for t in tools if t["tool_id"] == tool_id]
        lines = [f"Tool life — {self._agent.machine_id}:"]
        for t in tools:
            life     = t.get("remaining_life_pct", 0)
            life_hrs = t.get("remaining_life_hrs", 0)
            bar_len  = max(0, min(20, int(life / 5)))
            bar      = "█" * bar_len + "░" * (20 - bar_len)
            status   = ("⛔ NEEDS REPLACEMENT" if t.get("needs_replacement")
                        else ("⚠ WARN — replace soon" if life < 20 else "✓ OK"))
            lines.append(
                f"  {t['tool_id']}  [{bar}] {life:5.1f}%  "
                f"({life_hrs:.2f} hrs)  {status}"
            )
        return {"intent": "query_tool_life", "result": "\n".join(lines), "raw": tools}

    def _do_query_status(self, p: dict) -> dict:
        state = self._agent.get_state()
        lines = [
            f"Machine status — {self._agent.machine_id}:",
            f"  Status          : {state['status']}",
            f"  Current job     : {state.get('current_job') or '—'}",
            f"  Current seed    : {state.get('current_seed') or '—'}",
            f"  Section         : {state.get('current_section') or '—'}",
            f"  Sections done   : {state.get('completed_sections', [])}",
            f"  Queue depth     : {state.get('queue_depth', 0)}",
            f"  Rework depth    : {state.get('rework_depth', 0)}",
            f"  Active error    : {state.get('active_error') or 'None'}",
            f"  Pending tool Δ  : {state.get('pending_tool_change', False)}",
        ]
        return {"intent": "query_status", "result": "\n".join(lines), "raw": state}

    def _do_query_queue(self, p: dict) -> dict:
        import config as _cfg
        msim_ref = getattr(self, '_msim', None)   # injected by GUI if available
        queue_list = []
        if msim_ref is not None:
            queue_list = list(msim_ref.queue)
        state = self._agent.get_state()
        depth = state.get("queue_depth", len(queue_list))
        if not queue_list:
            result = (f"Queue — {self._agent.machine_id}:  "
                      f"{depth} job(s) queued  (detail not available in dry-run mode)")
        else:
            lines = [f"Queue — {self._agent.machine_id}  ({depth} jobs):"]
            for i, job in enumerate(queue_list):
                lines.append(
                    f"  [{i+1}] Seed {getattr(job,'seed','?')}  "
                    f"~{getattr(job,'estimated_time_s',0)/60:.1f} min  "
                    f"material={getattr(job,'material','?')}"
                )
            result = "\n".join(lines)
        return {"intent": "query_queue", "result": result, "raw": queue_list}

    def _do_query_cycle_time(self, p: dict) -> dict:
        agent = self._agent
        job   = agent.current_job
        lines = [f"Cycle time — {agent.machine_id}:"]
        if job:
            lines += [
                f"  Estimated mach  : {job.estimated_time_s/60:.1f} min",
                f"  Setup           : {2.0:.1f} min",
                f"  Tool change     : 2.5 min (if scheduled)",
                f"  Unclamp/removal : 2.0 min",
                f"  Total est.      : {(job.estimated_time_s + 360)/60:.1f} min",
            ]
        else:
            lines.append("  No active job.")
        return {"intent": "query_cycle_time", "result": "\n".join(lines), "raw": {}}

    def _do_query_cost(self, p: dict) -> dict:
        agent = self._agent
        job   = agent.current_job
        lines = [f"Job cost — {agent.machine_id}:"]
        if job:
            cost = agent.compute_cost(job)
            lines += [
                f"  Machine cost    : ${cost.get('machine_cost', 0):.4f}",
                f"  Tool cost       : ${cost.get('tool_cost', 0):.4f}",
                f"  Total cost      : ${cost.get('total_cost', 0):.4f}",
            ]
        else:
            lines.append("  No active job.")
        return {"intent": "query_cost", "result": "\n".join(lines), "raw": {}}

    def _do_query_feed_rpm(self, p: dict) -> dict:
        job = self._agent.current_job
        lines = [f"Feed & RPM — {self._agent.machine_id}:"]
        if job and hasattr(job, 'toolpath_result') and job.toolpath_result:
            lines.append("  (G-code params from last computed toolpath)")
        lines.append(
            f"  Feed rate       : "
            f"{getattr(job, 'feed_rate_mmpm', '—')} mm/min" if job else
            "  No active job — feed/RPM from last job not cached."
        )
        return {"intent": "query_feed_rpm", "result": "\n".join(lines), "raw": {}}

    def _do_query_material(self, p: dict) -> dict:
        job = self._agent.current_job
        if job and getattr(job, 'material', None):
            import config as _cfg
            mat  = job.material
            info = _cfg.MATERIAL_TABLE.get(mat, {})
            sfm  = info[0] if info else "—"
            kc   = info[1] if info else "—"
            result = (f"Material — {self._agent.machine_id}:\n"
                      f"  Material  : {mat}\n"
                      f"  SFM range : {sfm}\n"
                      f"  Kc (MPa)  : {kc}")
        else:
            result = f"Material — {self._agent.machine_id}:  no active job."
        return {"intent": "query_material", "result": result, "raw": {}}

    def _do_query_program(self, p: dict) -> dict:
        job = self._agent.current_job
        lines = [f"Program — {self._agent.machine_id}:"]
        if job:
            lines += [
                f"  Job ID    : {job.job_id}",
                f"  Seed      : {getattr(job, 'seed', '—')}",
                f"  Lines     : {len(job.gcode_lines)}",
                f"  Section   : {self._agent.current_section or '—'}",
                f"  Done      : {list(self._agent.completed_sections)}",
            ]
        else:
            lines.append("  No program loaded.")
        return {"intent": "query_program", "result": "\n".join(lines), "raw": {}}

    def _do_query_power(self, p: dict) -> dict:
        import config as _cfg
        job = self._agent.current_job
        lines = [f"Power — {self._agent.machine_id}:"]
        if job:
            lines += [
                f"  Machine limit   : {_cfg.MACHINE_POWER_LIMIT_KW:.1f} kW",
                f"  (Real-time draw requires connected simulator mode)",
            ]
        else:
            lines.append("  No active job.")
        return {"intent": "query_power", "result": "\n".join(lines), "raw": {}}

    def _do_query_execution_history(self, p: dict) -> dict:
        n = int(p.get("n", 5))
        hist = list(self._agent._error_history)[-n:]
        lines = [f"Error/event history — {self._agent.machine_id} (last {n}):"]
        if hist:
            for ev in hist:
                lines.append(
                    f"  [{ev.tick:>6}] {ev.description[:60]}  "
                    f"→ {ev.factory_response or 'pending'}"
                )
        else:
            lines.append("  No history.")
        return {"intent": "query_execution_history", "result": "\n".join(lines), "raw": hist}

    def _do_query_error_log(self, p: dict) -> dict:
        return self._do_query_execution_history(p)

    def _do_query_tool_detail(self, p: dict) -> dict:
        """Return full details for a specific tool or all tools if none specified."""
        tools = self._agent.tool_crib.state_list()
        tool_id = p.get("tool_id")
        if tool_id:
            tools = [t for t in tools if t["tool_id"].upper() == tool_id.upper()]
        if not tools:
            result = f"Tool {tool_id or '?'} not found in crib for {self._agent.machine_id}."
            return {"intent": "query_tool_detail", "result": result, "raw": []}

        import config as _cfg
        lines = [f"Tool detail — {self._agent.machine_id}:"]
        for t in tools:
            dia      = t.get("diameter_mm", 0)
            radius   = dia / 2.0
            life     = t.get("remaining_life_pct", 0)
            life_hrs = t.get("remaining_life_hrs", 0)
            bar_len  = max(0, min(20, int(life / 5)))
            bar      = "█" * bar_len + "░" * (20 - bar_len)
            status   = ("⛔ NEEDS REPLACEMENT" if t.get("needs_replacement")
                        else ("⚠ WARN — replace soon" if life < 20 else "✓ OK"))
            lines += [
                f"",
                f"  {t['tool_id']}  —  {t.get('label', 'End mill')}",
                f"  ├─ Diameter       : {dia:.1f} mm",
                f"  ├─ Radius         : {radius:.2f} mm",
                f"  ├─ Inserts/flutes : {t.get('n_inserts', '—')}",
                f"  ├─ Shank length   : {t.get('shank_length_mm', '—')} mm",
                f"  ├─ Tool type      : {t.get('tool_type', '—')}",
                f"  ├─ Life remaining : [{bar}] {life:.1f}%  ({life_hrs:.2f} hrs)",
                f"  ├─ Cost/hr        : ${t.get('cost_per_edge_usd', 0):.2f}",
                f"  └─ Status         : {status}",
            ]
        return {"intent": "query_tool_detail", "result": "\n".join(lines), "raw": tools}

    # ── G-code / live-execution handlers (require self._msim) ─────────────

    def _do_query_current_part(self, p: dict) -> dict:
        msim = self._msim
        if msim is None or getattr(msim, "current_job", None) is None:
            result = f"No part on machine {self._agent.machine_id} — machine is idle."
            return {"intent": "query_current_part", "result": result, "raw": {}}
        job    = msim.current_job
        cursor = getattr(msim, "gcode_cursor", 0)
        total  = len(getattr(msim, "gcode_lines", []))
        pct    = (cursor / total * 100) if total else 0.0
        bar    = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        result = (
            f"Part on machine {self._agent.machine_id}:\n"
            f"  Seed          : {getattr(job, 'seed', '?')}\n"
            f"  Job ID        : {getattr(job, 'job_id', '?')}\n"
            f"  Material      : {getattr(job, 'material', '?')}\n"
            f"  Tool in use   : {getattr(job, 'tool_id_used', '?')}  "
            f"Ø{getattr(job, 'tool_diameter_used', 0):.0f}mm\n"
            f"  Section       : {self._agent.current_section or '—'}\n"
            f"  Sections done : {list(self._agent.completed_sections)}\n"
            f"  Progress      : line {cursor}/{total}  [{bar}] {pct:.1f}%\n"
            f"  Est. time     : {getattr(job, 'estimated_time_s', 0)/60:.1f} min"
        )
        return {"intent": "query_current_part", "result": result, "raw": {}}

    def _do_query_gcode_current(self, p: dict) -> dict:
        msim   = self._msim
        if msim is None:
            return {"intent": "query_gcode_current",
                    "result": "G-code state not available — no simulation context.",
                    "raw": {}}
        lines  = getattr(msim, "gcode_lines", [])
        cursor = getattr(msim, "gcode_cursor", 0)
        if not lines:
            result = "No G-code loaded."
        elif cursor >= len(lines):
            result = f"Cursor past end of program (line {cursor}/{len(lines)})."
        else:
            cur  = lines[cursor].strip()
            prev = lines[cursor - 1].strip() if cursor > 0 else "—"
            result = (
                f"Current G-code — {self._agent.machine_id}  "
                f"(line {cursor + 1} of {len(lines)}):\n"
                f"  executing: [{cursor:>4}] {cur}\n"
                f"  previous : [{cursor - 1:>4}] {prev}"
            )
        return {"intent": "query_gcode_current", "result": result, "raw": {}}

    def _do_query_gcode_history(self, p: dict) -> dict:
        n      = int(p.get("n", 5))
        msim   = self._msim
        if msim is None:
            return {"intent": "query_gcode_history",
                    "result": "G-code state not available.", "raw": {}}
        lines  = getattr(msim, "gcode_lines", [])
        cursor = getattr(msim, "gcode_cursor", 0)
        start  = max(0, cursor - n)
        window = lines[start:cursor]
        if not window:
            result = f"No G-code executed yet (cursor at {cursor})."
        else:
            header = (f"Last {len(window)} line(s) executed — "
                      f"{self._agent.machine_id}  (cursor={cursor}/{len(lines)}):\n")
            rows   = "\n".join(
                f"  [{start + i:>4}]{'►' if (start + i) == cursor - 1 else ' '} "
                f"{ln.strip()}"
                for i, ln in enumerate(window)
            )
            result = header + rows
        return {"intent": "query_gcode_history", "result": result, "raw": window}

    def _do_query_gcode_upcoming(self, p: dict) -> dict:
        n      = int(p.get("n", 5))
        msim   = self._msim
        if msim is None:
            return {"intent": "query_gcode_upcoming",
                    "result": "G-code state not available.", "raw": {}}
        lines  = getattr(msim, "gcode_lines", [])
        cursor = getattr(msim, "gcode_cursor", 0)
        window = lines[cursor: cursor + n]
        if not window:
            result = "No upcoming lines — program complete or not started."
        else:
            header = (f"Next {len(window)} line(s) to execute — "
                      f"{self._agent.machine_id}  (cursor={cursor}/{len(lines)}):\n")
            rows   = "\n".join(
                f"  [{cursor + i:>4}]{'◄' if i == 0 else ' '} {ln.strip()}"
                for i, ln in enumerate(window)
            )
            result = header + rows
        return {"intent": "query_gcode_upcoming", "result": result, "raw": window}

    def _do_query_gcode_full(self, p: dict) -> dict:
        msim   = self._msim
        if msim is None:
            return {"intent": "query_gcode_full",
                    "result": "G-code state not available.", "raw": []}
        lines  = getattr(msim, "gcode_lines", [])
        cursor = getattr(msim, "gcode_cursor", 0)
        job    = getattr(msim, "current_job", None)
        if not lines:
            return {"intent": "query_gcode_full", "result": "No G-code loaded.", "raw": []}
        total = len(lines)
        seed  = getattr(job, "seed", "?") if job else "?"
        header = (f"G-code listing — {self._agent.machine_id}  "
                  f"Seed {seed}  ({total} lines)  cursor={cursor}:\n")
        MAX = 30
        if total <= MAX:
            body = "\n".join(
                f"  [{i:>4}]{'►' if i == cursor else ' '} {ln.strip()}"
                for i, ln in enumerate(lines)
            )
        else:
            HEAD, TAIL = 20, 5
            head_rows = "\n".join(
                f"  [{i:>4}]{'►' if i == cursor else ' '} {ln.strip()}"
                for i, ln in enumerate(lines[:HEAD])
            )
            tail_rows = "\n".join(
                f"  [{total - TAIL + i:>4}]"
                f"{'►' if (total - TAIL + i) == cursor else ' '} {ln.strip()}"
                for i, ln in enumerate(lines[total - TAIL:])
            )
            body = head_rows + f"\n  …  ({total - HEAD - TAIL} lines omitted)  …\n" + tail_rows
        return {"intent": "query_gcode_full", "result": header + body, "raw": lines}

    def _do_unknown(self, p: dict) -> dict:
        return {"intent": "unknown", "result": "Intent not recognised or not executable.", "raw": {}}


# ── Deterministic action classifier ──────────────────────────────────────────
# Known action types and their keyword triggers.
# Checked in order; first match wins.

_ACTION_PATTERNS: list[tuple[str, list[str]]] = [
    # Tool completely broken / snapped
    ("tool_broken", [
        "broken", "snapped", "shattered", "catastrophic",
        "tool broke", "broke mid", "no replacement",
        "insert broken", "insert snapped",
    ]),
    # Tool issue / suspected wear / poor cutting (but not confirmed broken)
    ("tool_issue", [
        "issue with t", "problem with t", "t is worn", "t not cutting",
        "t is not cutting", "poor cutting", "bad finish on", "tool issue",
        "t not working", "suspected wear",
        "not cutting well", "not cutting", "worn out",
        "cutting poorly", "poor surface", "finish is poor",
    ]),
    # Chatter / vibration
    ("chatter", [
        "chatter", "vibration", "resonance", "squealing",
        "ringing", "harmonics", "tool bounce",
    ]),
    # Feed / spindle overload
    ("reduce_feed", [
        "overload", "spindle overload", "reduce feed", "too high feed",
        "feed too high", "reduce feedrate", "feedrate too high",
        "spindle load", "cutting force", "power exceeded",
    ]),
    # Abort / emergency stop requested explicitly
    ("abort", [
        "abort", "stop the machine", "halt machine",
        "emergency stop", "e-stop", "estop",
    ]),
    ("wrong_material", [
        "wrong material", "incorrect material", "material mismatch",
        "wrong alloy", "wrong grade", "wrong stock",
        "material is wrong", "loaded wrong material",
    ]),
]


def classify_action(text: str) -> tuple[str, dict]:
    """
    Classify an operator action message into one of the known deterministic
    action types.  Returns (action_type, parameters).

    action_type is one of:
      "tool_broken"   — broken insert; find replacement or nearest-diameter
      "tool_issue"    — suspected issue with a specific tool; same resolution path
      "chatter"       — reduce feed 10%, optionally change RPM
      "reduce_feed"   — reduce feed 10% (overload / high cutting force)
      "abort"         — retract Z-safe, stop spindle, go home
      "unknown"       — not a recognised deterministic action

    parameters dict may contain:
      tool_id   — e.g. "T3" if a specific tool was named
      face      — face/feature name if mentioned ("Face 2")
    """
    lower = text.lower()
    params: dict = {}

    # Extract tool_id if present
    m = _TOOL_ID_RE.search(text)
    if m:
        params["tool_id"] = f"T{m.group(1)}"

    # Extract face / feature name
    face_m = _re.search(r'\bface\s*(\d+)\b', lower)
    if face_m:
        params["face"] = f"Face {face_m.group(1)}"

    for action_type, keywords in _ACTION_PATTERNS:
        for kw in keywords:
            if kw in lower:
                return (action_type, params)

    return ("unknown", params)


# ── Deterministic action executor ─────────────────────────────────────────────

class ActionExecutor:
    """
    Execute deterministic actions directly — no LLM call required.

    Each handler:
      1. Calls agent.inject_error() to set the active_error state.
      2. Executes the action (tool swap, feed reduction, abort, etc.).
      3. Calls agent.handle_factory_response() to clear the error.
      4. Returns a list of chat lines to display.

    The app reference is optional.  When provided, _trigger_replan() is
    called after any toolpath-invalidating change.
    """

    Z_SAFE_MM = 50.0   # retract height for abort / tool change

    def __init__(self, agent, factory_agent, app=None):
        self._agent = agent
        self._fa    = factory_agent
        self._app   = app
        self._msim  = None      # injected by caller if available

    def execute(self, action_type: str, parameters: dict) -> list[str]:
        """Returns list of (speaker, text) tuples for the chat log."""
        handler = getattr(self, f"_do_{action_type}", self._do_unknown)
        return handler(parameters)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _tick(self) -> int:
        return getattr(self._fa.scheduler, "tick", 0)

    def _inject(self, description: str) -> None:
        """Set active_error so handle_factory_response() will fire."""
        self._agent.inject_error(description)

    def _resolve(self, action: str, **kwargs) -> None:
        self._agent.handle_factory_response({"action": action, **kwargs})

    def _replan(self) -> str:
        """Trigger toolpath replan if app is available; return status line."""
        if self._app is not None:
            self._app._trigger_replan(self._agent.machine_id)
            return "Toolpath replan queued — remaining features will be re-CAMed."
        return "Replan needed — call _trigger_replan() manually."

    def _try_replace_tool(self, tool_id: str) -> list[str]:
        """
        Try to replace a specific tool.
        Returns list of chat lines describing what happened.
        Steps:
          1. Exact match from inventory.
          2. Nearest-diameter substitute.
          3. Remove from crib — run on remaining tools.
        """
        lines: list[str] = []
        worn = self._agent.tool_crib.get(tool_id)
        if worn is None:
            lines.append(("factory",
                f"Tool {tool_id} not found in {self._agent.machine_id} crib."))
            return lines

        cmd = self._fa.install_replacement_tool(
            self._agent, worn, self._tick(), urgency="stop")

        if cmd is None:
            lines.append(("factory",
                f"⚠ {tool_id}: install_replacement_tool returned None."))
            return lines

        p = cmd.payload
        if cmd.action == "tool_removed_no_stock":
            dia = p.get("diameter_mm", 0)
            lines += [
                ("factory",
                 f"🚫 No stock for {tool_id} (Ø{dia:.0f}mm) — tool removed from crib."),
                ("factory",
                 f"Machine will continue on remaining tools."),
            ]
            lines.append(("factory", self._replan()))
        elif cmd.action == "replace_tool":
            new_id  = p.get("new_tool_id", "?")
            new_dia = p.get("diameter_mm", 0)
            if p.get("substitute"):
                lines += [
                    ("factory",
                     f"⚠ No exact {tool_id} stock — installing nearest substitute "
                     f"{new_id} (Ø{new_dia:.0f}mm)."),
                    ("factory",
                     f"Different diameter: toolpath must be recomputed."),
                ]
                lines.append(("factory", self._replan()))
            else:
                lines.append(("factory",
                    f"✅ {tool_id} → {new_id} (Ø{new_dia:.0f}mm) installed from inventory."))
                # Same diameter — no replan needed unless the job references the tool explicitly
                lines.append(("machine",
                    f"Tool {new_id} loaded.  Resuming program from current section."))
        return lines

    # ── action handlers ───────────────────────────────────────────────────────

    def _do_tool_broken(self, p: dict) -> list[str]:
        tool_id = p.get("tool_id")
        mid     = self._agent.machine_id
        lines: list[str] = []

        self._inject(f"Tool broken: {tool_id or 'unknown'}")

        # If no specific tool named, check what's at STOP_PCT in the crib
        if not tool_id:
            at_stop = self._agent.tool_crib.tools_at_stop()
            if at_stop:
                tool_id = at_stop[0].tool_id
                lines.append(("factory",
                    f"🔴 No tool ID specified — using first STOP-threshold tool: {tool_id}"))
            else:
                # Use currently active tool if job is running
                job = self._agent.current_job
                if job:
                    active = self._agent.tool_crib.select_for_face(200.0)
                    tool_id = active.tool_id if active else None

        if not tool_id:
            lines.append(("factory", "⚠ Cannot identify broken tool — please specify T1…T10."))
            self._resolve("continue")
            return lines

        lines.append(("machine",
            f"🔧 {mid}: {tool_id} reported broken.  Spindle retracted to Z{self.Z_SAFE_MM:.0f}."))
        lines.append(("factory",
            f"Requesting replacement for {tool_id} from factory inventory…"))

        # Does the remaining job even USE this tool?
        remaining_uses_tool = self._remaining_job_uses_tool(tool_id)
        replace_lines = self._try_replace_tool(tool_id)
        lines.extend(replace_lines)

        if not remaining_uses_tool:
            lines.append(("factory",
                f"ℹ Remaining toolpath does not require {tool_id} — "
                f"continuing without replan."))
        if self._agent.active_error is not None:
            self._resolve("continue")

        lines.append(("machine",
            f"G0 Z{self.Z_SAFE_MM:.0f}  ; retract to safe height\n"
            f"M5               ; spindle stop\n"
            f"T{tool_id[1:]} M6       ; tool change block (operator confirms)\n"
            f"M3 S[rpm]        ; restart spindle"))
        return lines

    def _do_tool_issue(self, p: dict) -> list[str]:
        """Suspected issue with a tool — same resolution path as broken."""
        tool_id = p.get("tool_id")
        lines: list[str] = [
            ("factory",
             f"⚠ Suspected issue with {tool_id or 'tool'} — "
             f"treating as worn/broken; seeking replacement."),
        ]
        lines.extend(self._do_tool_broken(p))
        return lines

    def _do_chatter(self, p: dict) -> list[str]:
        """Chatter / vibration — reduce feed by 10%, suggest RPM shift."""
        mid     = self._agent.machine_id
        face    = p.get("face", "current feature")
        self._inject(f"Chatter detected on {face}")

        job          = self._agent.current_job
        current_feed = getattr(job, "feed_rate_mmpm", 1000) if job else 1000
        new_feed     = round(current_feed * 0.90)
        new_pct      = 90.0

        if self._agent.active_error is not None:
            self._resolve("reduce_feed", feed_override_pct=new_pct)

        return [
            ("machine",
             f"⚠ {mid}: chatter detected on {face}."),
            ("factory",
             f"Action: reduce feed rate by 10%\n"
             f"  Current feed : {current_feed:.0f} mm/min\n"
             f"  New feed     : {new_feed:.0f} mm/min  (override = {new_pct:.0f}%)"),
            ("factory",
             f"If chatter persists: shift RPM ±10–15% to escape resonance.\n"
             f"  G-code: F{new_feed}  ; apply immediately\n"
             f"          S[new_rpm]   ; ±10–15% of current spindle speed"),
            ("machine",
             f"Feed override {new_pct:.0f}% applied.  Monitoring spindle load."),
        ]

    def _do_reduce_feed(self, p: dict) -> list[str]:
        """Spindle overload / high cutting force — reduce feed by 10%."""
        mid          = self._agent.machine_id
        new_pct      = 90.0
        job          = self._agent.current_job
        current_feed = getattr(job, "feed_rate_mmpm", 1000) if job else 1000
        new_feed     = round(current_feed * 0.90)

        self._inject("Feed / spindle overload")
        if self._agent.active_error is not None:
            self._resolve("reduce_feed", feed_override_pct=new_pct)

        return [
            ("machine",
             f"⚠ {mid}: overload detected."),
            ("factory",
             f"Action: reduce feed rate by 10%\n"
             f"  Current feed : {current_feed:.0f} mm/min\n"
             f"  New feed     : {new_feed:.0f} mm/min  (override = {new_pct:.0f}%)\n"
             f"  G-code       : F{new_feed}  ; apply immediately"),
            ("factory",
             f"If spindle load remains >90% after one full pass, reduce a further 10%."),
            ("machine",
             f"Feed override {new_pct:.0f}% applied.  Continuing machining."),
        ]

    def _do_abort(self, p: dict) -> list[str]:
        """Explicit abort — retract Z, stop spindle, go home."""
        mid = self._agent.machine_id
        self._inject("Operator requested abort")
        if self._agent.active_error is not None:
            self._resolve("abort")
        if self._msim is not None:
            self._msim.status = "idle"

        return [
            ("machine",
             f"🛑 {mid}: abort received."),
            ("factory",
             f"Executing safe abort sequence:\n"
             f"  G0 Z{self.Z_SAFE_MM:.0f}    ; retract to Z-safe\n"
             f"  M5           ; spindle stop\n"
             f"  M9           ; coolant off\n"
             f"  G28 G91 Z0   ; return Z to home\n"
             f"  G28 G90 X0 Y0 ; return XY to home\n"
             f"  M30          ; program end"),
            ("machine",
             f"Machine stopped.  Part moved to rework queue.  "
             f"Reset required before next job."),
        ]

    def _do_wrong_material(self, p: dict) -> list[str]:
        """
        Wrong material loaded.
        Steps: stop → identify correct material → rebuild toolpath → restart.
        """
        import config as _cfg
        import threading
        mid  = self._agent.machine_id
        job  = self._agent.current_job
        msim = self._msim
        lines: list[str] = []

        # 1. Stop execution
        self._inject("Wrong material loaded — operator reported")
        if self._agent.active_error is not None:
            self._resolve("abort")
        if msim is not None:
            msim.status = "idle"

        lines.append(("machine",
            f"🛑 {mid}: execution stopped — wrong material reported."))
        lines.append(("factory",
            f"  G0 Z{self.Z_SAFE_MM:.0f}   ; retract to Z-safe\n"
            f"  M5         ; spindle stop\n"
            f"  M9         ; coolant off"))

        # 2. Identify the target material
        target_material = p.get("material")
        available       = list(_cfg.MATERIAL_TABLE.keys())
        avail_str       = "  |  ".join(available)

        if target_material and target_material in _cfg.MATERIAL_TABLE:
            lines.append(("factory",
                f"Target material from message: {target_material}"))
        elif job and getattr(job, "material", None):
            target_material = job.material
            lines.append(("factory",
                f"Using job specification material: {target_material}\n"
                f"To override, type: set material <name>"))
        else:
            target_material = _cfg.DEFAULT_MATERIAL
            lines.append(("factory",
                f"No material specified — defaulting to: {target_material}\n"
                f"Available: {avail_str}\n"
                f"Type: set material <name> to override"))

        # 3. Rebuild toolpath in background
        seed = getattr(job, "seed", None) if job else None
        if seed is not None and self._app is not None:
            sfm_range, kc = _cfg.MATERIAL_TABLE.get(target_material, ((300, 500), 2000))
            lines.append(("factory",
                f"Rebuilding toolpath:\n"
                f"  Seed     : {seed}\n"
                f"  Material : {target_material}\n"
                f"  SFM      : {sfm_range}\n"
                f"  Kc       : {kc} MPa"))
            fa_ref  = self._fa
            app_ref = self._app
            def _rebuild() -> None:
                new_job = fa_ref.build_job_from_seed(seed, machine_id=mid)
                if new_job is not None:
                    new_job.material = target_material
                    if msim is not None:
                        msim.queue.appendleft(new_job)
                    app_ref.post_chat(
                        mid, "factory",
                        f"✅ Toolpath rebuilt for {target_material}  "
                        f"({len(new_job.gcode_lines)} lines).  "
                        f"Confirm material loaded and press START.")
                else:
                    app_ref.post_chat(
                        mid, "factory",
                        f"⚠ Toolpath rebuild failed for Seed {seed}.  "
                        f"Check CAM pipeline logs.")
            threading.Thread(target=_rebuild, daemon=True).start()
            lines.append(("factory",
                "Toolpath rebuild running in background — "
                "result posted when complete."))
        else:
            lines.append(("factory",
                "⚠ Cannot auto-regenerate — no seed or app reference.  "
                "Reload program manually with correct material parameters."))

        lines.append(("machine",
            f"Machine idle.  Load correct material ({target_material}) "
            f"and confirm before restarting."))
        return lines

    def _do_unknown(self, p: dict) -> list[str]:
        return [("factory", "Action not recognised — routing to full diagnosis pipeline.")]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _remaining_job_uses_tool(self, tool_id: str) -> bool:
        """
        Check whether the tool is referenced in the remaining (unmachined)
        G-code lines.  Returns True if uncertain (safe default).
        """
        if self._msim is None or not hasattr(self._msim, 'gcode_lines'):
            return True   # unknown → assume yes
        cursor = getattr(self._msim, 'gcode_cursor', 0)
        remaining = self._msim.gcode_lines[cursor:]
        tool_ref  = f"T{tool_id[1:]}"   # "T3" → "T3"
        return any(tool_ref in line for line in remaining)

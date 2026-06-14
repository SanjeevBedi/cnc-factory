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

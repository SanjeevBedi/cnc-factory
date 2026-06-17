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

  build_system_context()  → str   — full capability block for the LLM system prompt
  intent_suggestions()    → list[str]  — flat list for GUI autocomplete
  allowed_intents()       → list[AgentIntent]  — filter by category / safety
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Intent dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentIntent:
    """One capability the machine-agent can execute."""
    intent_id:      str             # snake_case identifier used in JSON responses
    category:       str             # execution | query | tool | program | schedule | safety
    label:          str             # human-readable short name
    description:    str             # what the action does (shown to LLM + operator)
    parameters:     tuple[str, ...] # named parameters the intent accepts (may be empty)
    requires_stop:  bool  = False   # True → spindle must be stopped before execution
    safety_critical: bool = False   # True → requires operator confirmation before dispatch
    reversible:     bool  = True    # False → destructive / cannot be undone
    example:        str   = ""      # one-line JSON example shown to the LLM


# ── Catalogue ─────────────────────────────────────────────────────────────────

INTENT_CATALOGUE: list[AgentIntent] = [

    # ── EXECUTION — control the current machining operation ───────────────────

    AgentIntent(
        intent_id   = "continue",
        category    = "execution",
        label       = "Continue machining",
        description = "Resume or continue the current G-code program without "
                      "parameter changes.  Use when the disturbance is minor and "
                      "within tolerance.",
        parameters  = (),
        example     = '{"intent":"continue"}',
    ),
    AgentIntent(
        intent_id   = "pause",
        category    = "execution",
        label       = "Pause program",
        description = "Suspend execution at the next safe retract position.  "
                      "Spindle remains on.  Use to inspect the workpiece or "
                      "await operator input.",
        parameters  = ("reason",),
        example     = '{"intent":"pause","reason":"dimensional check required"}',
    ),
    AgentIntent(
        intent_id   = "resume",
        category    = "execution",
        label       = "Resume from pause",
        description = "Re-start a paused program from the line where it stopped.",
        parameters  = (),
        example     = '{"intent":"resume"}',
    ),
    AgentIntent(
        intent_id   = "abort",
        category    = "execution",
        label       = "Abort job",
        description = "Stop the program immediately.  Spindle and coolant are "
                      "switched off; axes retract to home.  Part goes to the "
                      "rework queue.",
        parameters  = ("reason",),
        requires_stop  = True,
        reversible     = False,
        safety_critical = True,
        example     = '{"intent":"abort","reason":"tool broken mid-cut"}',
    ),
    AgentIntent(
        intent_id   = "emergency_stop",
        category    = "execution",
        label       = "Emergency stop (E-stop)",
        description = "Immediate full stop: spindle off, feed hold, coolant off, "
                      "all axes halted.  Requires manual reset before any further "
                      "motion.  Use only for safety-critical conditions.",
        parameters  = ("reason",),
        requires_stop   = True,
        reversible      = False,
        safety_critical = True,
        example     = '{"intent":"emergency_stop","reason":"fixture loose"}',
    ),
    AgentIntent(
        intent_id   = "skip_feature",
        category    = "execution",
        label       = "Skip current feature",
        description = "Jump to the next feature (face/pocket/hole) in the program, "
                      "leaving the current one unmachined.  Use when a feature "
                      "cannot be completed with the available tool.",
        parameters  = ("feature_id",),
        example     = '{"intent":"skip_feature","feature_id":"Face 2"}',
    ),

    # ── FEED & SPEED — adjust cutting parameters ──────────────────────────────

    AgentIntent(
        intent_id   = "reduce_feed",
        category    = "execution",
        label       = "Reduce feed rate",
        description = "Apply a percentage feed-rate override to reduce cutting "
                      "forces.  Takes effect on the next motion block.",
        parameters  = ("feed_override_pct",),
        example     = '{"intent":"reduce_feed","feed_override_pct":75}',
    ),
    AgentIntent(
        intent_id   = "increase_feed",
        category    = "execution",
        label       = "Increase feed rate",
        description = "Apply a positive feed-rate override to increase throughput "
                      "when material is softer than specified.",
        parameters  = ("feed_override_pct",),
        example     = '{"intent":"increase_feed","feed_override_pct":115}',
    ),
    AgentIntent(
        intent_id   = "change_rpm",
        category    = "execution",
        label       = "Change spindle RPM",
        description = "Override the programmed spindle speed.  Used for chatter "
                      "avoidance (±10–15% RPM shift) or thermal management.",
        parameters  = ("rpm",),
        example     = '{"intent":"change_rpm","rpm":3400}',
    ),
    AgentIntent(
        intent_id   = "reduce_doc",
        category    = "execution",
        label       = "Reduce depth of cut",
        description = "Reduce the axial depth of cut for the next pass.  "
                      "Requires toolpath regeneration for remaining features.",
        parameters  = ("depth_of_cut_mm",),
        example     = '{"intent":"reduce_doc","depth_of_cut_mm":0.3}',
    ),
    AgentIntent(
        intent_id   = "recalculate",
        category    = "execution",
        label       = "Recalculate feeds & speeds",
        description = "Recompute optimal feed rate, RPM, and depth-of-cut for a "
                      "different material or tool, then continue.  Triggers a "
                      "partial toolpath regeneration for remaining features.",
        parameters  = ("material", "tool_id", "feed_rate_mmpm", "rpm",
                       "depth_of_cut_mm"),
        example     = '{"intent":"recalculate","material":"alloy_steel_4140",'
                      '"tool_id":"T3","feed_rate_mmpm":1800,"rpm":2200,'
                      '"depth_of_cut_mm":0.4}',
    ),
    AgentIntent(
        intent_id   = "add_finish_pass",
        category    = "execution",
        label       = "Add finishing pass",
        description = "Append one light finishing pass (reduced feed, small step-over) "
                      "to the current feature to recover surface finish.",
        parameters  = ("feed_rate_mmpm", "step_over_pct"),
        example     = '{"intent":"add_finish_pass","feed_rate_mmpm":800,'
                      '"step_over_pct":10}',
    ),
    AgentIntent(
        intent_id   = "add_rough_pass",
        category    = "execution",
        label       = "Add roughing pass",
        description = "Insert an additional roughing pass before the current "
                      "operation to remove excess stock caused by an oversize "
                      "blank or previous rework.",
        parameters  = ("depth_of_cut_mm", "feed_rate_mmpm"),
        example     = '{"intent":"add_rough_pass","depth_of_cut_mm":1.5,'
                      '"feed_rate_mmpm":2500}',
    ),
    AgentIntent(
        intent_id   = "change_path",
        category    = "execution",
        label       = "Change toolpath strategy",
        description = "Switch to an alternative toolpath strategy for the current "
                      "feature (e.g. trochoidal, climb vs conventional) to "
                      "reduce chatter or cutting forces.",
        parameters  = ("strategy",),
        example     = '{"intent":"change_path","strategy":"trochoidal"}',
    ),

    # ── TOOL — manage tools in the crib ──────────────────────────────────────

    AgentIntent(
        intent_id   = "change_tool",
        category    = "tool",
        label       = "Change tool",
        description = "Retract to tool-change position, swap to the specified "
                      "tool from the crib, and continue.  Incurs "
                      "TOOL_CHANGE_TIME_S (150 s) downtime.",
        parameters  = ("tool_id",),
        requires_stop = True,
        example     = '{"intent":"change_tool","tool_id":"T4"}',
    ),
    AgentIntent(
        intent_id   = "query_tool_life",
        category    = "tool",
        label       = "Query tool life",
        description = "Return the remaining life percentage, hours, and wear "
                      "status for every tool in the crib.",
        parameters  = ("tool_id",),      # optional — omit for all tools
        example     = '{"intent":"query_tool_life"}',
    ),
    AgentIntent(
        intent_id   = "query_tool_crib",
        category    = "tool",
        label       = "Query full tool crib",
        description = "Return complete details for all tools: ID, diameter, "
                      "insert count, shank length, life remaining, cost, "
                      "and replacement status.",
        parameters  = (),
        example     = '{"intent":"query_tool_crib"}',
    ),
    AgentIntent(
        intent_id   = "set_tool_override",
        category    = "tool",
        label       = "Override tool selection",
        description = "Force a specific tool to be used for the next feature, "
                      "overriding the automatic selection algorithm.",
        parameters  = ("tool_id", "feature_type"),
        example     = '{"intent":"set_tool_override","tool_id":"T6",'
                      '"feature_type":"face"}',
    ),

    # ── PROGRAM — load, inspect, and manage G-code programs ──────────────────

    AgentIntent(
        intent_id   = "load_program",
        category    = "program",
        label       = "Load G-code program",
        description = "Load a G-code program by seed ID or file path into the "
                      "machine buffer, ready for execution.",
        parameters  = ("seed", "filepath"),
        example     = '{"intent":"load_program","seed":1042}',
    ),
    AgentIntent(
        intent_id   = "unload_program",
        category    = "program",
        label       = "Unload program",
        description = "Clear the current program from the machine buffer.  "
                      "The part remains on the table.",
        parameters  = (),
        example     = '{"intent":"unload_program"}',
    ),
    AgentIntent(
        intent_id   = "query_program",
        category    = "program",
        label       = "Query current program",
        description = "Return the currently loaded program: seed ID, line count, "
                      "current cursor position, face labels, and estimated "
                      "remaining time.",
        parameters  = (),
        example     = '{"intent":"query_program"}',
    ),
    AgentIntent(
        intent_id   = "query_execution_history",
        category    = "program",
        label       = "Query execution history",
        description = "Return a list of the last N programs executed: seed, "
                      "material, machining time, cost, sections completed, "
                      "and any errors encountered.",
        parameters  = ("n",),
        example     = '{"intent":"query_execution_history","n":5}',
    ),
    AgentIntent(
        intent_id   = "replan_remaining",
        category    = "program",
        label       = "Replan remaining features",
        description = "Regenerate the toolpath and G-code for all unmachined "
                      "features using the current tool crib.  Completed faces "
                      "(identified by FACE_END sentinels) are preserved.",
        parameters  = ("tool_id",),
        example     = '{"intent":"replan_remaining","tool_id":"T5"}',
    ),

    # ── QUERY — read machine state and production data ────────────────────────

    AgentIntent(
        intent_id   = "query_status",
        category    = "query",
        label       = "Query machine status",
        description = "Return the full machine state snapshot: status, current "
                      "job, section, queue depth, rework count, active error, "
                      "and pending tool-change flag.",
        parameters  = (),
        example     = '{"intent":"query_status"}',
    ),
    AgentIntent(
        intent_id   = "query_cycle_time",
        category    = "query",
        label       = "Query cycle time",
        description = "Return estimated and actual machining time for the current "
                      "or last job: setup, cutting, tool-change, and unclamp "
                      "breakdown.",
        parameters  = ("job_id",),
        example     = '{"intent":"query_cycle_time"}',
    ),
    AgentIntent(
        intent_id   = "query_cost",
        category    = "query",
        label       = "Query job cost",
        description = "Return the cost breakdown for the current or last job: "
                      "machine amortisation, tool consumable, power, and total.",
        parameters  = ("job_id",),
        example     = '{"intent":"query_cost"}',
    ),
    AgentIntent(
        intent_id   = "query_power",
        category    = "query",
        label       = "Query spindle power",
        description = "Return the current estimated spindle power draw and "
                      "comparison against the machine limit.",
        parameters  = (),
        example     = '{"intent":"query_power"}',
    ),
    AgentIntent(
        intent_id   = "query_feed_rpm",
        category    = "query",
        label       = "Query feed & RPM",
        description = "Return the active feed rate (mm/min), spindle speed (RPM), "
                      "depth of cut, step-over, and any active overrides.",
        parameters  = (),
        example     = '{"intent":"query_feed_rpm"}',
    ),
    AgentIntent(
        intent_id   = "query_material",
        category    = "query",
        label       = "Query material",
        description = "Return the material assigned to the current job: name, "
                      "specific cutting force (Kc), and chip-load parameters.",
        parameters  = (),
        example     = '{"intent":"query_material"}',
    ),
    AgentIntent(
        intent_id   = "query_queue",
        category    = "query",
        label       = "Query job queue",
        description = "Return the list of jobs waiting in the machine queue: "
                      "seed IDs, estimated times, materials, and tool requirements.",
        parameters  = (),
        example     = '{"intent":"query_queue"}',
    ),
    AgentIntent(
        intent_id   = "query_error_log",
        category    = "query",
        label       = "Query error log",
        description = "Return the active error (if any) and the last N error "
                      "events: description, section, G-code line, and resolution.",
        parameters  = ("n",),
        example     = '{"intent":"query_error_log","n":3}',
    ),

    # ── SCHEDULE — queue and routing actions ──────────────────────────────────

    AgentIntent(
        intent_id   = "rework",
        category    = "schedule",
        label       = "Move part to rework",
        description = "Mark the current job as rework: stop machining, add to "
                      "the rework queue for manual inspection, and free the "
                      "machine for the next job.",
        parameters  = ("reason",),
        reversible  = False,
        example     = '{"intent":"rework","reason":"dimension out of tolerance"}',
    ),
    AgentIntent(
        intent_id   = "transfer",
        category    = "schedule",
        label       = "Transfer job to another machine",
        description = "Move the current job (or a queued job) to a different "
                      "machine.  The receiving machine must be idle or have "
                      "queue capacity.  Triggers re-routing negotiation.",
        parameters  = ("target_machine_id", "job_id"),
        example     = '{"intent":"transfer","target_machine_id":"M03","job_id":"J-042"}',
    ),
    AgentIntent(
        intent_id   = "reorder_queue",
        category    = "schedule",
        label       = "Reorder job queue",
        description = "Change the processing order of queued jobs by priority, "
                      "estimated time, or material similarity.",
        parameters  = ("order_by",),
        example     = '{"intent":"reorder_queue","order_by":"priority"}',
    ),
    AgentIntent(
        intent_id   = "continue_then_reorder",
        category    = "schedule",
        label       = "Continue current job then reorder",
        description = "Finish the current job without interruption but flag the "
                      "queue for reordering before the next job starts.",
        parameters  = ("reason",),
        example     = '{"intent":"continue_then_reorder","reason":"priority change received"}',
    ),
    AgentIntent(
        intent_id   = "pause_and_inspect",
        category    = "schedule",
        label       = "Pause for quality inspection",
        description = "Pause at the current retract position and notify quality "
                      "control for an in-process dimensional check.",
        parameters  = ("feature_id", "dimension_mm", "tolerance_mm"),
        example     = '{"intent":"pause_and_inspect","feature_id":"Face 1",'
                      '"dimension_mm":45.0,"tolerance_mm":0.05}',
    ),
    AgentIntent(
        intent_id   = "wait_and_resume",
        category    = "schedule",
        label       = "Wait then resume",
        description = "Hold the machine in a safe state for a specified number "
                      "of ticks, then automatically resume.  Use for coolant "
                      "purge, fixture settle, or thermal stabilisation.",
        parameters  = ("wait_ticks", "reason"),
        example     = '{"intent":"wait_and_resume","wait_ticks":6,"reason":"coolant purge"}',
    ),

    # ── SAFETY ────────────────────────────────────────────────────────────────

    AgentIntent(
        intent_id   = "manual_intervention",
        category    = "safety",
        label       = "Request manual intervention",
        description = "Flag the machine as requiring a human operator.  "
                      "Machine pauses and the factory supervisor is notified.  "
                      "No further automatic actions until the operator clears "
                      "the condition.",
        parameters  = ("reason",),
        requires_stop   = True,
        safety_critical = True,
        example     = '{"intent":"manual_intervention",'
                      '"reason":"fixture may be loose — vibration detected"}',
    ),
    AgentIntent(
        intent_id   = "continue_dry",
        category    = "safety",
        label       = "Continue in dry-run mode",
        description = "Re-run the remaining program without coolant.  "
                      "Acceptable only for light finishing cuts in aluminium. "
                      "Not permitted for steel, titanium, or Inconel.",
        parameters  = ("reason",),
        example     = '{"intent":"continue_dry","reason":"coolant line blocked"}',
    ),
    AgentIntent(
        intent_id   = "skip_tool_change",
        category    = "safety",
        label       = "Skip scheduled tool change",
        description = "Bypass the pending tool-change and continue with the worn "
                      "tool.  Permitted only if remaining life > STOP_PCT and the "
                      "operator explicitly accepts increased risk.",
        parameters  = ("tool_id", "accepted_risk"),
        safety_critical = True,
        example     = '{"intent":"skip_tool_change","tool_id":"T2","accepted_risk":true}',
    ),
    AgentIntent(
        intent_id   = "notify_factory",
        category    = "safety",
        label       = "Notify factory supervisor",
        description = "Send an alert to the factory-level supervisor with a "
                      "structured message.  Machine continues unless separately "
                      "instructed to pause.",
        parameters  = ("message", "urgency"),
        example     = '{"intent":"notify_factory","message":"delivery delay on M03",'
                      '"urgency":"normal"}',
    ),
]


# ── Derived lookups ───────────────────────────────────────────────────────────

INTENT_MAP: dict[str, AgentIntent] = {i.intent_id: i for i in INTENT_CATALOGUE}

INTENT_GROUPS: dict[str, list[str]] = {}
for _intent in INTENT_CATALOGUE:
    INTENT_GROUPS.setdefault(_intent.category, []).append(_intent.intent_id)


# ── Public helpers ────────────────────────────────────────────────────────────

def allowed_intents(
    category: Optional[str] = None,
    exclude_safety_critical: bool = False,
    exclude_requires_stop: bool = False,
) -> list[AgentIntent]:
    """Return intents filtered by category and capability flags."""
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
    machine_id: str = "M??",
    include_examples: bool = True,
    include_safety_notes: bool = True,
) -> str:
    """
    Build the capability block that is prepended to every LLM system prompt.

    The block tells the model exactly what the agent can do, what parameters
    each intent accepts, and gives a one-line JSON example for every intent.
    The model is instructed to respond ONLY with intents from this list.
    """
    lines: list[str] = [
        f"=== MACHINE AGENT CAPABILITIES  ({machine_id}) ===",
        "",
        "You are the factory supervisor AI.  The machine agent for "
        f"{machine_id} can execute the following intents.",
        "Your response MUST use 'intent' as the top-level action key and "
        "MUST choose an intent_id from the list below.",
        "Do NOT invent intent names not on this list.",
        "",
    ]

    # Group by category
    category_order = ["execution", "tool", "program", "query", "schedule", "safety"]
    category_label = {
        "execution": "EXECUTION — control the machining operation",
        "tool":      "TOOL — manage tools and the crib",
        "program":   "PROGRAM — load, inspect, and manage G-code",
        "query":     "QUERY — read machine state and production data",
        "schedule":  "SCHEDULE — job queue and routing",
        "safety":    "SAFETY — protective and human-escalation actions",
    }

    for cat in category_order:
        intents_in_cat = [i for i in INTENT_CATALOGUE if i.category == cat]
        if not intents_in_cat:
            continue
        lines.append(f"── {category_label[cat]} ──")
        for intent in intents_in_cat:
            flags = []
            if intent.requires_stop:
                flags.append("spindle-stop required")
            if intent.safety_critical:
                flags.append("⚠ safety-critical")
            if not intent.reversible:
                flags.append("irreversible")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            lines.append(f"  {intent.intent_id:<28s} {intent.label}{flag_str}")
            lines.append(f"    {intent.description}")
            if intent.parameters:
                lines.append(f"    Parameters: {', '.join(intent.parameters)}")
            if include_examples and intent.example:
                lines.append(f"    Example   : {intent.example}")
            lines.append("")

    if include_safety_notes:
        lines += [
            "── SAFETY RULES ──",
            "  1. Never emit emergency_stop or abort without a concrete technical reason.",
            "  2. safety_critical intents (marked ⚠) require 'accepted_risk':true in parameters.",
            "  3. Prefer the least-disruptive intent that resolves the condition.",
            "  4. If the correct intent requires spindle-stop, include "
               "'stop_spindle_first':true.",
            "",
        ]

    lines += [
        "── RESPONSE FORMAT ──",
        '  {"intent"     : "<intent_id>",',
        '   "parameters" : { <key>:<value>, ... },',
        '   "risk_level" : "safe" | "risky" | "catastrophic",',
        '   "reasoning"  : "<one concise sentence>"}',
        "",
        "=== END OF AGENT CAPABILITIES ===",
    ]

    return "\n".join(lines)


def build_suggestions_block(
    disturbance_key: Optional[str] = None,
    possible_actions: Optional[tuple[str, ...]] = None,
) -> str:
    """
    Build a short 'SUGGESTED INTENTS' block for the LLM prompt body.
    Scoped to the disturbance type when possible_actions is provided.
    """
    if possible_actions:
        # Map raw action strings to intent objects where possible
        suggested = [INTENT_MAP[a] for a in possible_actions if a in INTENT_MAP]
    else:
        # Default: non-safety-critical execution intents
        suggested = allowed_intents(
            category="execution", exclude_safety_critical=True)

    if not suggested:
        return ""

    lines = ["SUGGESTED INTENTS FOR THIS DISTURBANCE:"]
    for i in suggested:
        lines.append(f"  {i.intent_id:<28s} — {i.description[:80]}")
    return "\n".join(lines)

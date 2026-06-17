"""
disturbance_engine.py  —  Comprehensive CNC disturbance taxonomy and handling.

Every real-world event that can interrupt or degrade machining is classified
here.  For each type the engine can:
  1. Build a rich, engineering-grounded DisturbanceContext from live agent state.
  2. Generate an exact OpenAI prompt (numbers, not vague words).
  3. Provide rule-based fallback responses when no LLM is available.
  4. Score candidate responses against the active factory policy.

Disturbance categories
──────────────────────
  TOOL      breakage | wear_critical | wrong_type | chatter | deflection
  MATERIAL  harder | softer | dimension_oversize |
            wrong_grade | inclusion
  PROCESS   spindle_overload | coolant_failure | surface_finish_poor |
            dimension_error | power_exceedance | chatter_vibration
  MACHINE   spindle_bearing | axis_fault | fixture_loose | power_fault |
            tool_changer_fault
  SCHEDULE  priority_change | delivery_delay | quality_hold
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import config
from feeds_speeds_engine import compute as _fsc


# ── Taxonomy ──────────────────────────────────────────────────────────────────

class Category(str, Enum):
    TOOL     = "tool"
    MATERIAL = "material"
    PROCESS  = "process"
    MACHINE  = "machine"
    SCHEDULE = "schedule"


@dataclass
class GUIField:
    """Declares one extra input shown by the inject-disturbance dialog."""
    name:    str
    label:   str
    kind:    str          # "text" | "float" | "bool" | "choice"
    default: Any = ""
    choices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DisturbanceSpec:
    """Static descriptor — one per entry in REGISTRY."""
    key:              str
    category:         Category
    label:            str
    short_desc:       str                # shown in the GUI radio button
    possible_actions: tuple[str, ...]
    gui_fields:       tuple[GUIField, ...] = ()
    safety_critical:  bool = False       # True → always abort immediately


# ── Full registry ─────────────────────────────────────────────────────────────

REGISTRY: dict[str, DisturbanceSpec] = {

    # ── TOOL ──────────────────────────────────────────────────────────────────
    "tool_breakage": DisturbanceSpec(
        key="tool_breakage", category=Category.TOOL,
        label="Tool breakage",
        short_desc="Insert broken mid-cut — request tool change from factory agent (50-tick delay)",
        possible_actions=("request_tool_change", "abort"),
        safety_critical=True,
        gui_fields=(
            GUIField("tool_id",  "Broken tool ID",           "text",   "T3"),
            GUIField("no_stock", "No replacement in factory stock",
                     "bool", True),
        ),
    ),
    "tool_wear_critical": DisturbanceSpec(
        key="tool_wear_critical", category=Category.TOOL,
        label="Tool wear — critical (>90 %)",
        short_desc="Change tool before next part reload",
        possible_actions=("change_tool_before_reload", "reduce_feed"),
        gui_fields=(
            GUIField("tool_id",  "Worn tool ID", "text",  "T3"),
            GUIField("wear_pct", "Wear %",       "float", 92.0),
        ),
    ),
    "tool_wrong_type": DisturbanceSpec(
        key="tool_wrong_type", category=Category.TOOL,
        label="Wrong tool loaded",
        short_desc="Stop immediately, unload part, request correct tool (50-tick delay), regen CAM",
        possible_actions=("stop_unload_request_tool",),
        safety_critical=True,
        gui_fields=(
            GUIField("expected_tool", "Expected tool ID", "text", "T3"),
            GUIField("loaded_tool",   "Loaded tool ID",   "text", "T4"),
        ),
    ),
    "tool_chatter": DisturbanceSpec(
        key="tool_chatter", category=Category.TOOL,
        label="Tool chatter / resonance",
        short_desc="Slow machine feedrate by 10%",
        possible_actions=("reduce_feed", "abort"),
        gui_fields=(
            GUIField("frequency_hz", "Chatter frequency Hz (if known)",
                     "float", 0.0),
            GUIField("severity",     "Severity",
                     "choice", "medium",
                     choices=["low", "medium", "high", "severe"]),
        ),
    ),
    "tool_deflection": DisturbanceSpec(
        key="tool_deflection", category=Category.TOOL,
        label="Tool deflection — dimension drift",
        short_desc="Slow machine feedrate by 20%",
        possible_actions=("reduce_feed", "change_tool"),
        gui_fields=(
            GUIField("measured_error_mm", "Measured error mm", "float", 0.05),
        ),
    ),

    # ── MATERIAL ──────────────────────────────────────────────────────────────
    "material_harder": DisturbanceSpec(
        key="material_harder", category=Category.MATERIAL,
        label="Material harder than specified",
        short_desc="Pause machine, redo CAM with actual material, restart machining",
        possible_actions=("pause_redo_cam", "abort"),
        gui_fields=(
            GUIField("original_material", "Specified material",
                     "choice", "aluminium_6061",
                     choices=list(config.MATERIAL_TABLE.keys())),
            GUIField("actual_material",   "Actual material detected",
                     "choice", "mild_steel_1018",
                     choices=list(config.MATERIAL_TABLE.keys())),
            GUIField("evidence",          "Evidence / measurement",
                     "text", "Hardness test: 180 HB"),
        ),
    ),
    "material_softer": DisturbanceSpec(
        key="material_softer", category=Category.MATERIAL,
        label="Material softer than specified",
        short_desc="Pause machine, redo CAM with actual material, restart machining",
        possible_actions=("pause_redo_cam", "abort"),
        gui_fields=(
            GUIField("original_material", "Specified material",
                     "choice", "alloy_steel_4140",
                     choices=list(config.MATERIAL_TABLE.keys())),
            GUIField("actual_material",   "Actual material",
                     "choice", "mild_steel_1018",
                     choices=list(config.MATERIAL_TABLE.keys())),
            GUIField("evidence", "Evidence", "text",
                     "Spec 4140 HT, delivered annealed"),
        ),
    ),
    "material_dimension_oversize": DisturbanceSpec(
        key="material_dimension_oversize", category=Category.MATERIAL,
        label="Stock over-size",
        short_desc="Stop machine, unload part, move to next part — this part returned to factory",
        possible_actions=("stop_unload_next_part",),
        gui_fields=(
            GUIField("axis",          "Oversize axis", "choice", "Z",
                     choices=["X","Y","Z","XY"]),
            GUIField("excess_mm",     "Excess mm",    "float", 5.0),
        ),
    ),
    "material_wrong_grade": DisturbanceSpec(
        key="material_wrong_grade", category=Category.MATERIAL,
        label="Wrong material grade / alloy",
        short_desc="Slow machine feedrate by 10%",
        possible_actions=("reduce_feed", "abort"),
        gui_fields=(
            GUIField("original_material", "Specified", "choice",
                     "aluminium_6061", choices=list(config.MATERIAL_TABLE.keys())),
            GUIField("actual_material",   "Actual",    "choice",
                     "aluminium_cast",   choices=list(config.MATERIAL_TABLE.keys())),
        ),
    ),
    "material_inclusion": DisturbanceSpec(
        key="material_inclusion", category=Category.MATERIAL,
        label="Hard inclusion in workpiece",
        short_desc="Slow machine feedrate by 25%",
        possible_actions=("reduce_feed", "abort"),
        gui_fields=(
            GUIField("depth_mm", "Estimated depth of inclusion mm", "float", 10.0),
        ),
    ),

    # ── PROCESS ───────────────────────────────────────────────────────────────
    "spindle_overload": DisturbanceSpec(
        key="spindle_overload", category=Category.PROCESS,
        label="Spindle overload",
        short_desc="Slow spindle speed by 10%",
        possible_actions=("reduce_spindle_speed", "reduce_feed", "abort"),
        gui_fields=(
            GUIField("overload_pct", "Overload above rated %", "float", 25.0),
            GUIField("measured_kw",  "Measured power kW",      "float", 9.5),
        ),
    ),
    "coolant_failure": DisturbanceSpec(
        key="coolant_failure", category=Category.PROCESS,
        label="Coolant failure",
        short_desc="Stop machining, move to safe Z, request coolant (25-tick delay), restart",
        possible_actions=("stop_request_coolant",),
        safety_critical=False,
        gui_fields=(
            GUIField("coolant_type", "Coolant type",
                     "choice", "flood", choices=["flood","mist","through_tool"]),
            GUIField("partial_loss", "Partial loss (not total)", "bool", False),
        ),
    ),
    "surface_finish_poor": DisturbanceSpec(
        key="surface_finish_poor", category=Category.PROCESS,
        label="Surface finish below spec",
        short_desc="Slow feedrate by 10%",
        possible_actions=("reduce_feed", "rework"),
        gui_fields=(
            GUIField("required_ra",   "Required Ra µm",  "float", 1.6),
            GUIField("measured_ra",   "Measured Ra µm",  "float", 3.8),
        ),
    ),
    "dimension_error": DisturbanceSpec(
        key="dimension_error", category=Category.PROCESS,
        label="Dimension out of tolerance",
        short_desc="Stop machine, unload part and return part to factory",
        possible_actions=("stop_unload_return", "continue"),
        gui_fields=(
            GUIField("nominal_mm",    "Nominal mm",    "float", 50.0),
            GUIField("actual_mm",     "Measured mm",   "float", 50.08),
            GUIField("tolerance_mm",  "Tolerance ± mm","float", 0.05),
            GUIField("feature",       "Feature",       "text",  "Face 2 depth"),
        ),
    ),
    "power_exceedance": DisturbanceSpec(
        key="power_exceedance", category=Category.PROCESS,
        label="Cutting power exceeds machine limit",
        short_desc=f"Machine rated {config.MACHINE_POWER_LIMIT_KW} kW — slow feedrate by 25%",
        possible_actions=("reduce_feed", "abort"),
        gui_fields=(
            GUIField("measured_kw",  "Measured power kW",  "float", 8.2),
        ),
    ),
    "chatter_vibration": DisturbanceSpec(
        key="chatter_vibration", category=Category.PROCESS,
        label="Chatter / regenerative vibration",
        short_desc="Slow machine feedrate by 10%",
        possible_actions=("reduce_feed", "abort"),
        gui_fields=(
            GUIField("frequency_hz", "Frequency Hz (if known)", "float", 0.0),
        ),
    ),

    # ── MACHINE ───────────────────────────────────────────────────────────────
    "spindle_bearing": DisturbanceSpec(
        key="spindle_bearing", category=Category.MACHINE,
        label="Spindle bearing noise / fault",
        short_desc="Stop machine, unload part, return to factory — machine out of order",
        possible_actions=("stop_machine_oos",),
        safety_critical=True,
    ),
    "axis_fault": DisturbanceSpec(
        key="axis_fault", category=Category.MACHINE,
        label="Servo / axis fault",
        short_desc="Stop machine, unload part, return to factory, delay 25 ticks, restart",
        possible_actions=("stop_unload_delay_restart",),
        safety_critical=True,
        gui_fields=(
            GUIField("axis",       "Faulted axis", "choice", "Z",
                     choices=["X","Y","Z"]),
            GUIField("error_mm",   "Following error mm", "float", 0.5),
        ),
    ),
    "fixture_loose": DisturbanceSpec(
        key="fixture_loose", category=Category.MACHINE,
        label="Fixture / workholding loose",
        short_desc="Stop machine, unload part, return to factory, delay 25 ticks, restart",
        possible_actions=("stop_unload_delay_restart",),
        safety_critical=True,
    ),
    "power_fault": DisturbanceSpec(
        key="power_fault", category=Category.MACHINE,
        label="Power / voltage fluctuation",
        short_desc="Stop machine, wait for restart from agent",
        possible_actions=("wait_agent_restart",),
        gui_fields=(
            GUIField("duration_s", "Duration seconds", "float", 2.0),
        ),
    ),
    "tool_changer_fault": DisturbanceSpec(
        key="tool_changer_fault", category=Category.MACHINE,
        label="Automatic tool-changer fault",
        short_desc="Stop machine, wait for restart from agent",
        possible_actions=("wait_agent_restart",),
    ),

    # ── SCHEDULE ──────────────────────────────────────────────────────────────
    "priority_change": DisturbanceSpec(
        key="priority_change", category=Category.SCHEDULE,
        label="Customer priority change",
        short_desc="Move part up in queue",
        possible_actions=("move_part_up_queue", "continue_then_reorder"),
        gui_fields=(
            GUIField("new_priority_job",  "New priority job seed", "text", ""),
            GUIField("reason",            "Reason",                "text",
                     "Customer expedite request"),
        ),
    ),
    "delivery_delay": DisturbanceSpec(
        key="delivery_delay", category=Category.SCHEDULE,
        label="Material / tooling delivery delay",
        short_desc="Move part back in queue; for tooling find closest tool, load in crib, redo CAM",
        possible_actions=("delay_part_back_queue", "load_substitute_tool"),
        gui_fields=(
            GUIField("item",       "Delayed item",    "text",  "T3 x5 pcs"),
            GUIField("delay_hrs",  "Delay hours",     "float", 24.0),
            GUIField("item_type",  "Item type",       "choice", "material",
                     choices=["material", "tooling"]),
        ),
    ),
    "quality_hold": DisturbanceSpec(
        key="quality_hold", category=Category.SCHEDULE,
        label="Quality hold — inspection required",
        short_desc="Stop machine, wait for agent to restart",
        possible_actions=("wait_agent_restart",),
        gui_fields=(
            GUIField("reason", "Hold reason", "text",
                     "Dimensional audit required"),
        ),
    ),
}

# Group by category for the GUI tabs
BY_CATEGORY: dict[Category, list[DisturbanceSpec]] = {c: [] for c in Category}
for _spec in REGISTRY.values():
    BY_CATEGORY[_spec.category].append(_spec)


# ── Rich disturbance context ──────────────────────────────────────────────────

@dataclass
class DisturbanceContext:
    """All engineering data assembled for one disturbance event."""
    # identity
    disturbance_key:  str
    machine_id:       str
    tick:             int
    factory_policy:   str

    # agent snapshot
    machine_status:   str
    current_job:      Optional[str]
    current_section:  Optional[str]
    sections_done:    list[str]     = field(default_factory=list)
    sections_remaining: list[str]  = field(default_factory=list)
    queue_depth:      int           = 0

    # tool state
    active_tool_id:   str           = ""
    active_tool_dia:  float         = 0.0
    active_tool_life: float         = 100.0    # %
    tool_crib:        list[dict]    = field(default_factory=list)

    # material (for MATERIAL category)
    original_material: str  = ""
    actual_material:   str  = ""
    kc_original:       float = 0.0
    kc_actual:         float = 0.0
    kc_ratio:          float = 1.0
    feed_original:     float = 0.0    # mm/min
    feed_required:     float = 0.0    # mm/min after adjustment
    rpm_original:      float = 0.0
    rpm_required:      float = 0.0
    power_original:    float = 0.0
    power_required:    float = 0.0
    feed_reduction_pct:float = 0.0
    tool_adequate:     bool  = True
    tool_grade_note:   str   = ""

    # machine limits
    machine_power_kw:  float = config.MACHINE_POWER_LIMIT_KW
    machine_torque_nm: float = config.MACHINE_TORQUE_LIMIT_NM

    # extra data from dialog (catch-all)
    extra: dict = field(default_factory=dict)

    # factory inventory
    inventory: dict = field(default_factory=dict)


@dataclass
class LLMCandidate:
    action:      str
    risk_level:  str    # safe | risky | catastrophic
    parameters:  dict
    reasoning:   str
    temperature: float
    raw:         str
    score:       float  = 0.0


# ── Engine ────────────────────────────────────────────────────────────────────

class DisturbanceEngine:
    """
    Builds context, generates prompts, provides fallback responses,
    and scores candidates for every disturbance in the REGISTRY.
    """

    # ── context building ─────────────────────────────────────────────────────

    def build_context(
        self,
        agent,           # CncAgent
        spec:  DisturbanceSpec,
        extra: dict,
        factory_inventory: dict,
        tick: int,
        policy: str,
    ) -> DisturbanceContext:
        """Assemble DisturbanceContext from live agent state + dialog extras."""
        state    = agent.get_state()
        all_sec  = (agent.current_job.gcode_lines
                    if agent.current_job else [])
        done     = state.get("completed_sections", [])

        # derive remaining sections from gcode comment labels
        rem = [l.strip("()")
               for l in all_sec
               if l.startswith("(") and not l.startswith("(CNC")
               and l.strip("()") not in done]

        crib = state.get("tool_crib", [])
        # best-life tool currently in spindle → highest remaining life
        active = max(crib, key=lambda t: t.get("remaining_life_pct", 0),
                     default={})

        ctx = DisturbanceContext(
            disturbance_key   = spec.key,
            machine_id        = agent.machine_id,
            tick              = tick,
            factory_policy    = policy,
            machine_status    = state.get("status", "unknown"),
            current_job       = state.get("current_job"),
            current_section   = state.get("current_section"),
            sections_done     = done,
            sections_remaining= rem[:8],     # truncate for prompt
            queue_depth       = state.get("queue_depth", 0),
            active_tool_id    = active.get("tool_id", ""),
            active_tool_dia   = active.get("diameter_mm", 0.0),
            active_tool_life  = active.get("remaining_life_pct", 100.0),
            tool_crib         = crib,
            machine_power_kw  = config.MACHINE_POWER_LIMIT_KW,
            machine_torque_nm = config.MACHINE_TORQUE_LIMIT_NM,
            extra             = extra,
            inventory         = {k: len(v)
                                 for k, v in factory_inventory.items()},
        )

        # ── material-change enrichment ────────────────────────────────────
        if spec.category == Category.MATERIAL and "actual_material" in extra:
            ctx = self._enrich_material(ctx, extra)

        return ctx

    def _enrich_material(self, ctx: DisturbanceContext, extra: dict
                         ) -> DisturbanceContext:
        """Compute engineering implications of a material mismatch."""
        orig = extra.get("original_material", config.DEFAULT_MATERIAL)
        act  = extra.get("actual_material",   orig)

        orig_sfm, kc_orig = config.MATERIAL_TABLE.get(orig, ((800, 1200), 600))
        act_sfm,  kc_act  = config.MATERIAL_TABLE.get(act,  ((800, 1200), 600))

        ratio = kc_act / max(kc_orig, 1.0)

        dia   = ctx.active_tool_dia or config.DEFAULT_TOOL_DIAMETER_MM

        # compute current and required feeds/speeds
        try:
            fs_old = _fsc(orig, dia, 4, axial_depth_mm=3.0, radial_depth_mm=dia*0.4)
            fs_new = _fsc(act,  dia, 4, axial_depth_mm=3.0, radial_depth_mm=dia*0.4)
            feed_old = fs_old.feed_rate_mmpm
            feed_new = fs_new.feed_rate_mmpm
            rpm_old  = fs_old.rpm
            rpm_new  = fs_new.rpm
            pwr_old  = fs_old.power_kw
            pwr_new  = fs_new.power_kw
        except Exception:
            feed_old = 3000.0
            feed_new = 3000.0 / ratio
            rpm_old  = 8000.0
            rpm_new  = rpm_old / ratio
            pwr_old  = pwr_new = 2.0

        feed_reduction = max(0.0, (1.0 - feed_new / max(feed_old, 1.0)) * 100.0)

        # tool adequacy: uncoated HSS/Carbide grade OK for ≤ steel,
        # Ti/Inconel require special coating — simplified heuristic
        hard_materials = {"titanium_ti64", "inconel", "tool_steel_d2"}
        tool_adequate = act not in hard_materials or ratio < 1.5
        tool_grade_note = (
            "Coated carbide grade (AlTiN/TiAlN) mandatory for "
            f"{act.replace('_',' ')} — verify tool spec."
            if not tool_adequate else ""
        )

        ctx.original_material  = orig
        ctx.actual_material    = act
        ctx.kc_original        = float(kc_orig)
        ctx.kc_actual          = float(kc_act)
        ctx.kc_ratio           = round(ratio, 2)
        ctx.feed_original      = round(feed_old, 1)
        ctx.feed_required      = round(feed_new, 1)
        ctx.rpm_original       = round(rpm_old,  1)
        ctx.rpm_required       = round(rpm_new,  1)
        ctx.power_original     = round(pwr_old,  2)
        ctx.power_required     = round(pwr_new,  2)
        ctx.feed_reduction_pct = round(feed_reduction, 1)
        ctx.tool_adequate      = tool_adequate
        ctx.tool_grade_note    = tool_grade_note
        return ctx

    # ── prompt generation ─────────────────────────────────────────────────────

    def build_prompt(self, ctx: DisturbanceContext) -> str:
        """Return the full OpenAI prompt string for this disturbance."""
        spec = REGISTRY.get(ctx.disturbance_key)
        if spec is None:
            return self._generic_prompt(ctx)

        builders = {
            "material_harder":           self._prompt_material_change,
            "material_softer":           self._prompt_material_change,
            "material_wrong_grade":      self._prompt_material_change,
            "tool_breakage":             self._prompt_tool_breakage,
            "tool_wear_critical":        self._prompt_tool_wear,
            "tool_chatter":              self._prompt_chatter,
            "chatter_vibration":         self._prompt_chatter,
            "spindle_overload":          self._prompt_spindle_overload,
            "coolant_failure":           self._prompt_coolant,
            "surface_finish_poor":       self._prompt_surface_finish,
            "dimension_error":           self._prompt_dimension_error,
            "power_exceedance":          self._prompt_power,
            "priority_change":           self._prompt_priority_change,
        }
        builder = builders.get(ctx.disturbance_key, self._generic_prompt)
        return builder(ctx)

    # ── per-type prompt builders ──────────────────────────────────────────────

    def _prompt_material_change(self, ctx: DisturbanceContext) -> str:
        direction = "HARDER" if ctx.kc_ratio > 1.0 else "SOFTER"
        force_chg  = (ctx.kc_ratio - 1.0) * 100.0
        life_red   = min(90.0, (ctx.kc_ratio - 1.0) * 40.0)
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"ALERT: Material mismatch — stock is {direction} than specified.\n\n"
            f"  Specified material  : {ctx.original_material.replace('_',' ')}\n"
            f"  Actual material     : {ctx.actual_material.replace('_',' ')}\n"
            f"  Kc specified        : {ctx.kc_original:.0f} MPa\n"
            f"  Kc actual           : {ctx.kc_actual:.0f} MPa\n"
            f"  Hardness ratio      : {ctx.kc_ratio:.2f}×\n"
            f"  Evidence            : {ctx.extra.get('evidence','—')}\n\n"
            "Engineering implications:\n"
            f"  Cutting force increases by  : {force_chg:+.0f}%\n"
            f"  Tool life expected to drop  : ~{life_red:.0f}%\n"
            f"  Required feed rate          : {ctx.feed_required:.0f} mm/min  "
            f"(was {ctx.feed_original:.0f} mm/min, "
            f"−{ctx.feed_reduction_pct:.0f}%)\n"
            f"  Required spindle speed      : {ctx.rpm_required:.0f} RPM  "
            f"(was {ctx.rpm_original:.0f} RPM)\n"
            f"  Estimated new cutting power : {ctx.power_required:.2f} kW  "
            f"(machine limit {ctx.machine_power_kw:.1f} kW)\n"
            f"  Power within machine limit  : "
            f"{'YES' if ctx.power_required <= ctx.machine_power_kw else 'NO — EXCEEDS LIMIT'}\n"
            + (f"\n  Tool adequacy warning: {ctx.tool_grade_note}\n"
               if ctx.tool_grade_note else "") +
            f"\nCurrent machine status      : {ctx.machine_id} — "
            f"{ctx.machine_status}\n"
            f"Sections completed          : {ctx.sections_done}\n"
            f"Sections remaining          : {ctx.sections_remaining}\n"
            f"Active tool                 : {ctx.active_tool_id}  "
            f"⌀{ctx.active_tool_dia:.0f}mm  "
            f"life={ctx.active_tool_life:.0f}%\n"
            f"Factory policy              : {ctx.factory_policy}\n\n"
            "Available actions:\n"
            "  pause_redo_cam — pause machine, recompute CAM for actual material, restart\n"
            "  abort          — stop job immediately; place in rework queue\n\n"
            "Respond with JSON only (no prose outside the object):\n"
            '{"action":"pause_redo_cam|abort",\n'
            ' "parameters":{"new_material":"<material_key>","feed_rate_mmpm":float,"rpm":float},\n'
            ' "risk_level":"safe|risky|catastrophic",\n'
            ' "reasoning":"<one concise sentence>"}'
        )

    def _prompt_tool_breakage(self, ctx: DisturbanceContext) -> str:
        tool_id   = ctx.extra.get("tool_id", "T3")
        no_stock  = ctx.extra.get("no_stock", True)
        others    = [t for t in ctx.tool_crib
                     if t["tool_id"] != tool_id
                     and t.get("remaining_life_pct", 0) > 15]
        inv_count = ctx.inventory.get(tool_id, 0)
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"CRITICAL: Tool breakage on {ctx.machine_id}.\n\n"
            f"  Broken tool         : {tool_id}\n"
            f"  Factory inventory   : {inv_count} units "
            f"{'— OUT OF STOCK' if inv_count == 0 else '— replacement available'}\n"
            f"  Other usable tools  : "
            f"{[t['tool_id']+' ⌀'+str(t['diameter_mm'])+'mm' for t in others]}\n"
            f"  Job progress        : {len(ctx.sections_done)} sections done, "
            f"{len(ctx.sections_remaining)} remaining\n"
            f"  Factory policy      : {ctx.factory_policy}\n\n"
            "Available actions:\n"
            "  request_tool_change — request replacement from factory agent; "
            "await delivery (50-tick delay); mount and resume\n"
            "  abort               — stop job immediately; place in rework queue\n\n"
            'Respond JSON only:\n'
            '{"action":"request_tool_change|abort","target_machine":"M0x or null",'
            '"parameters":{"tool_id":"T3","delay_ticks":50},'
            '"risk_level":"safe|risky|catastrophic",'
            '"reasoning":"<one sentence>"}'
        )

    def _prompt_tool_wear(self, ctx: DisturbanceContext) -> str:
        wear = ctx.extra.get("wear_pct", 90.0)
        tool = ctx.extra.get("tool_id", "T3")
        inv  = ctx.inventory.get(tool, 0)
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"WARNING: Tool {tool} on {ctx.machine_id} at {wear:.0f}% wear.\n"
            f"Factory inventory for {tool}: {inv} units.\n"
            f"Sections remaining: {ctx.sections_remaining}\n"
            f"Factory policy: {ctx.factory_policy}\n\n"
            "Actions: change_tool_before_reload (change tool before next part load) | "
            "reduce_feed (extend life)\n\n"
            'Respond JSON only:\n'
            '{"action":"change_tool_before_reload|reduce_feed","parameters":{},'
            '"risk_level":"safe|risky|catastrophic","reasoning":"<one sentence>"}'
        )

    def _prompt_chatter(self, ctx: DisturbanceContext) -> str:
        freq = ctx.extra.get("frequency_hz", 0.0)
        sev  = ctx.extra.get("severity", "medium")
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"ALERT: Chatter/vibration on {ctx.machine_id}.\n"
            f"  Severity : {sev}\n"
            f"  Frequency: {freq:.0f} Hz  "
            f"{'(unknown)' if freq == 0 else ''}\n"
            f"  Active tool: {ctx.active_tool_id} ⌀{ctx.active_tool_dia:.0f}mm\n"
            f"  Feed now   : {ctx.feed_original:.0f} mm/min\n"
            f"  Policy     : {ctx.factory_policy}\n\n"
            "Actions: reduce_feed (slow machine by 10%) | abort\n\n"
            'Respond JSON only:\n'
            '{"action":"reduce_feed|abort",'
            '"parameters":{"feed_delta_pct":-10.0},'
            '"risk_level":"safe|risky|catastrophic","reasoning":"<one sentence>"}'
        )

    def _prompt_spindle_overload(self, ctx: DisturbanceContext) -> str:
        overload = ctx.extra.get("overload_pct", 25.0)
        kw       = ctx.extra.get("measured_kw", 0.0)
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"ALERT: Spindle overload on {ctx.machine_id}.\n"
            f"  Overload         : +{overload:.0f}% above rated\n"
            f"  Measured power   : {kw:.1f} kW  "
            f"(limit {config.MACHINE_POWER_LIMIT_KW} kW)\n"
            f"  Active tool      : {ctx.active_tool_id} ⌀{ctx.active_tool_dia:.0f}mm\n"
            f"  Policy           : {ctx.factory_policy}\n\n"
            "Actions: reduce_spindle_speed (slow spindle by 10%) | reduce_feed | abort\n\n"
            'Respond JSON only:\n'
            '{"action":"reduce_spindle_speed|reduce_feed|abort",'
            '"parameters":{"rpm_delta_pct":-10.0},'
            '"risk_level":"safe|risky|catastrophic","reasoning":"<one sentence>"}'
        )

    def _prompt_coolant(self, ctx: DisturbanceContext) -> str:
        partial = ctx.extra.get("partial_loss", False)
        ctype   = ctx.extra.get("coolant_type", "flood")
        mat     = ctx.original_material or config.DEFAULT_MATERIAL
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"ALERT: Coolant {'partial loss' if partial else 'total failure'} "
            f"on {ctx.machine_id}.\n"
            f"  Coolant type   : {ctype}\n"
            f"  Material       : {mat.replace('_',' ')}\n"
            f"  Active tool    : {ctx.active_tool_id} ⌀{ctx.active_tool_dia:.0f}mm\n"
            f"  Sections left  : {len(ctx.sections_remaining)}\n\n"
            "Required action:\n"
            "  stop_request_coolant — stop machining, retract to safe Z-height, "
            "request coolant (25-tick delay), then restart: spindle on, Z down, continue G-code\n\n"
            'Respond JSON only:\n'
            '{"action":"stop_request_coolant","parameters":{"delay_ticks":25},'
            '"risk_level":"safe|risky|catastrophic","reasoning":"<one sentence>"}'
        )

    def _prompt_surface_finish(self, ctx: DisturbanceContext) -> str:
        req = ctx.extra.get("required_ra", 1.6)
        act = ctx.extra.get("measured_ra", 3.8)
        ratio = act / max(req, 0.01)
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"QUALITY ALERT: Surface finish below spec on {ctx.machine_id}.\n"
            f"  Required Ra  : {req:.1f} µm\n"
            f"  Measured Ra  : {act:.1f} µm  ({ratio:.1f}× spec)\n"
            f"  Active tool  : {ctx.active_tool_id}\n"
            f"  Policy       : {ctx.factory_policy}\n\n"
            "Actions: reduce_feed (slow feedrate by 10%) | rework\n\n"
            'Respond JSON only:\n'
            '{"action":"reduce_feed|rework",'
            '"parameters":{"feed_delta_pct":-10.0},'
            '"risk_level":"safe|risky|catastrophic","reasoning":"<one sentence>"}'
        )

    def _prompt_dimension_error(self, ctx: DisturbanceContext) -> str:
        nom = ctx.extra.get("nominal_mm", 50.0)
        act = ctx.extra.get("actual_mm",  50.0)
        tol = ctx.extra.get("tolerance_mm", 0.05)
        err = act - nom
        feat = ctx.extra.get("feature", "unknown feature")
        in_tol = abs(err) <= tol
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"QUALITY ALERT: Dimension error on {ctx.machine_id} — {feat}.\n"
            f"  Nominal          : {nom:.3f} mm\n"
            f"  Measured         : {act:.3f} mm\n"
            f"  Error            : {err:+.3f} mm  "
            f"(tolerance ±{tol:.3f} mm)\n"
            f"  In tolerance     : {'YES' if in_tol else 'NO — OUT OF TOLERANCE'}\n"
            f"  Policy           : {ctx.factory_policy}\n\n"
            "Actions: stop_unload_return (stop machine, unload part, return to factory) | "
            "continue (if in tolerance)\n\n"
            'Respond JSON only:\n'
            '{"action":"stop_unload_return|continue","parameters":{},'
            '"risk_level":"safe|risky|catastrophic","reasoning":"<one sentence>"}'
        )

    def _prompt_power(self, ctx: DisturbanceContext) -> str:
        kw = ctx.extra.get("measured_kw", 0.0)
        over = (kw - config.MACHINE_POWER_LIMIT_KW) / config.MACHINE_POWER_LIMIT_KW * 100
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"ALERT: Cutting power exceeds machine limit on {ctx.machine_id}.\n"
            f"  Measured power : {kw:.1f} kW  "
            f"(limit {config.MACHINE_POWER_LIMIT_KW} kW, "
            f"+{over:.0f}% over)\n"
            f"  Active tool    : {ctx.active_tool_id}\n"
            f"  Policy         : {ctx.factory_policy}\n\n"
            "Actions: reduce_feed (slow feedrate by 25%) | abort\n\n"
            'Respond JSON only:\n'
            '{"action":"reduce_feed|abort",'
            '"parameters":{"feed_delta_pct":-25.0},'
            '"risk_level":"safe|risky|catastrophic","reasoning":"<one sentence>"}'
        )

    def _prompt_priority_change(self, ctx: DisturbanceContext) -> str:
        new_job = ctx.extra.get("new_priority_job", "—")
        reason  = ctx.extra.get("reason", "customer request")
        est_rem = len(ctx.sections_remaining) * 30  # rough seconds
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"SCHEDULE: Priority change requested on {ctx.machine_id}.\n"
            f"  New priority job : {new_job}\n"
            f"  Reason           : {reason}\n"
            f"  Current job      : {ctx.current_job}\n"
            f"  Estimated remaining time : ~{est_rem}s\n"
            f"  Sections remaining       : {ctx.sections_remaining}\n"
            f"  Policy           : {ctx.factory_policy}\n\n"
            "Actions: move_part_up_queue (move priority job to front of queue) | "
            "continue_then_reorder (finish current then reorder)\n\n"
            'Respond JSON only:\n'
            '{"action":"move_part_up_queue|continue_then_reorder",'
            '"parameters":{},"risk_level":"safe|risky|catastrophic",'
            '"reasoning":"<one sentence>"}'
        )

    def _generic_prompt(self, ctx: DisturbanceContext) -> str:
        spec = REGISTRY.get(ctx.disturbance_key)
        label = spec.label if spec else ctx.disturbance_key
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"ALERT: {label} on {ctx.machine_id}.\n"
            f"  Machine status : {ctx.machine_status}\n"
            f"  Current job    : {ctx.current_job}\n"
            f"  Sections done  : {ctx.sections_done}\n"
            f"  Policy         : {ctx.factory_policy}\n"
            f"  Extra data     : {ctx.extra}\n\n"
            'Respond JSON only:\n'
            '{"action":"abort|rework|continue|reduce_feed","parameters":{},'
            '"risk_level":"safe|risky|catastrophic","reasoning":"<one sentence>"}'
        )

    # ── rule-based fallback ────────────────────────────────────────────────────

    def fallback_responses(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        """
        Return up to 3 deterministic candidates when OpenAI is unavailable.
        Each uses engineering data from ctx so numbers are always correct.
        """
        dispatch = {
            "material_harder":            self._fallback_material_harder,
            "material_softer":            self._fallback_material_softer,
            "material_wrong_grade":       self._fallback_material_wrong_grade,
            "material_dimension_oversize":self._fallback_material_oversize,
            "material_inclusion":         self._fallback_material_inclusion,
            "tool_breakage":              self._fallback_tool_breakage,
            "tool_wear_critical":         self._fallback_tool_wear,
            "tool_wrong_type":            self._fallback_tool_wrong_type,
            "tool_deflection":            self._fallback_tool_deflection,
            "tool_chatter":               self._fallback_chatter,
            "chatter_vibration":          self._fallback_chatter,
            "spindle_overload":           self._fallback_spindle_overload,
            "coolant_failure":            self._fallback_coolant,
            "surface_finish_poor":        self._fallback_surface_finish,
            "dimension_error":            self._fallback_dimension_error,
            "power_exceedance":           self._fallback_power,
            "priority_change":            self._fallback_priority,
            "delivery_delay":             self._fallback_delivery_delay,
            "quality_hold":               self._fallback_quality_hold,
            "fixture_loose":              self._fallback_stop_unload_delay_restart,
            "spindle_bearing":            self._fallback_spindle_bearing_oos,
            "axis_fault":                 self._fallback_stop_unload_delay_restart,
            "power_fault":                self._fallback_wait_agent_restart,
            "tool_changer_fault":         self._fallback_wait_agent_restart,
        }
        fn = dispatch.get(ctx.disturbance_key, self._fallback_generic)
        return fn(ctx)

    # material harder
    def _fallback_material_harder(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        ratio = ctx.kc_ratio or 1.0
        candidates: list[LLMCandidate] = []

        if ratio <= 1.0:
            # softer — shouldn't reach here but handle gracefully
            return self._fallback_material_softer(ctx)

        # candidate 1 — pause and redo CAM (unless power hard-exceeds limit)
        if ctx.power_required > ctx.machine_power_kw or not ctx.tool_adequate:
            candidates.append(LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning=(
                    f"Required cutting power {ctx.power_required:.1f} kW exceeds machine "
                    f"limit {ctx.machine_power_kw:.1f} kW or tool grade inadequate. "
                    "Abort and re-plan with correct tool/material."),
                temperature=0.2, raw="",
            ))
        else:
            candidates.append(LLMCandidate(
                action="pause_redo_cam", risk_level="safe",
                parameters={
                    "new_material":   ctx.actual_material,
                    "feed_rate_mmpm": ctx.feed_required,
                    "rpm":            ctx.rpm_required,
                },
                reasoning=(
                    f"Pause machine. Redo CAM for {ctx.actual_material.replace('_',' ')}: "
                    f"feed {ctx.feed_original:.0f}→{ctx.feed_required:.0f} mm/min "
                    f"(−{ctx.feed_reduction_pct:.0f}%), "
                    f"RPM {ctx.rpm_original:.0f}→{ctx.rpm_required:.0f}. Restart."),
                temperature=0.2, raw="",
            ))

        # candidate 2 — conservative reduce_feed fallback if redo-CAM not immediately available
        feed_adj  = ctx.feed_original / ratio
        rpm_adj   = ctx.rpm_original  / ratio
        candidates.append(LLMCandidate(
            action="reduce_feed", risk_level="risky",
            parameters={
                "new_material":       ctx.actual_material,
                "feed_rate_mmpm":     round(feed_adj, 1),
                "rpm":                round(rpm_adj,  1),
                "feed_change_pct":    round((feed_adj / ctx.feed_original - 1) * 100, 1),
            },
            reasoning=(
                f"Proportional feed reduction: feed ×{1/ratio:.2f}, "
                f"RPM ×{1/ratio:.2f}. Use if redo-CAM is not immediately feasible."),
            temperature=0.5, raw="",
        ))

        # candidate 3 — abort as final safety option
        candidates.append(LLMCandidate(
            action="abort", risk_level="safe",
            parameters={},
            reasoning=(
                f"Material mismatch (Kc ratio {ratio:.2f}×). "
                "Abort job if redo-CAM not feasible."),
            temperature=0.8, raw="",
        ))

        return candidates[:3]

    def _fallback_material_softer(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        ratio = ctx.kc_ratio or 1.0
        return [
            LLMCandidate(
                action="pause_redo_cam", risk_level="safe",
                parameters={
                    "new_material":   ctx.actual_material,
                    "feed_rate_mmpm": ctx.feed_required,
                    "rpm":            ctx.rpm_required,
                },
                reasoning=(
                    f"Material softer (Kc ratio {ratio:.2f}×). "
                    "Pause machine, redo CAM with actual material, restart."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning=(
                    "Abort if redo CAM is not immediately feasible."),
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_material_wrong_grade(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -10.0},
                reasoning=(
                    "Wrong material grade: slow feedrate by 10% as a "
                    "conservative precaution."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="risky",
                parameters={"feed_delta_pct": -15.0},
                reasoning=(
                    "Wrong grade: reduce feed 15% for added safety margin."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning=(
                    "Abort if wrong grade poses a significant machining risk."),
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_material_oversize(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        axis    = ctx.extra.get("axis", "unknown")
        excess  = ctx.extra.get("excess_mm", 0.0)
        return [
            LLMCandidate(
                action="stop_unload_next_part", risk_level="safe",
                parameters={"axis": axis, "excess_mm": excess},
                reasoning=(
                    f"Stock over-size by {excess:.1f}mm on {axis} axis. "
                    "Stop machine, unload part, return to factory, load next part."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="stop_unload_next_part", risk_level="safe",
                parameters={"axis": axis, "excess_mm": excess},
                reasoning=(
                    "Over-size stock cannot be safely machined to spec. "
                    "Stop and return part."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort job and flag part for return to factory.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_material_inclusion(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        depth = ctx.extra.get("depth_mm", 10.0)
        return [
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -25.0},
                reasoning=(
                    f"Hard inclusion at ~{depth:.0f}mm depth. "
                    "Slow feedrate by 25% to reduce cutting forces."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="risky",
                parameters={"feed_delta_pct": -30.0},
                reasoning=(
                    "Severe inclusion risk: reduce feed 30% for safer approach."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort if inclusion poses tool breakage risk.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_tool_breakage(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        tool_id  = ctx.extra.get("tool_id", "T3")
        no_stock = ctx.extra.get("no_stock", True)
        return [
            LLMCandidate(
                action="request_tool_change", risk_level="safe",
                parameters={"tool_id": tool_id, "delay_ticks": 50},
                reasoning=(
                    f"Tool {tool_id} broken. "
                    + ("No stock — " if no_stock else "Replacement available — ")
                    + "request tool change from factory agent (50-tick delay)."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="request_tool_change", risk_level="safe",
                parameters={"tool_id": tool_id, "delay_ticks": 50},
                reasoning=(
                    "Request replacement tool from factory and await delivery."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort job if no replacement tool can be sourced.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_tool_wear(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        tool  = ctx.extra.get("tool_id", "T3")
        inv   = ctx.inventory.get(tool, 0)
        wear  = ctx.extra.get("wear_pct", 90.0)
        return [
            LLMCandidate(
                action="change_tool_before_reload",
                risk_level="safe",
                parameters={"tool_id": tool},
                reasoning=(
                    f"Tool {tool} at {wear:.0f}% wear. "
                    + (f"Change before next part reload (stock: {inv})."
                       if inv > 0 else
                       "No stock — reduce feed 15% to extend life.")),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="risky",
                parameters={"feed_delta_pct": -15.0},
                reasoning="Reduce feed 15% to extend remaining tool life.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="change_tool_before_reload", risk_level="safe",
                parameters={"tool_id": tool},
                reasoning=f"Tool at {wear:.0f}% — schedule change before next reload.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_tool_wrong_type(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        expected = ctx.extra.get("expected_tool", "T3")
        loaded   = ctx.extra.get("loaded_tool", "T4")
        return [
            LLMCandidate(
                action="stop_unload_request_tool", risk_level="safe",
                parameters={
                    "expected_tool": expected,
                    "loaded_tool":   loaded,
                    "delay_ticks":   50,
                    "redo_cam":      True,
                },
                reasoning=(
                    f"Wrong tool loaded ({loaded} instead of {expected}). "
                    "Stop immediately, unload part, request correct tool "
                    "(50-tick delay), regenerate CAM, reload and restart."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="stop_unload_request_tool", risk_level="safe",
                parameters={
                    "expected_tool": expected,
                    "loaded_tool":   loaded,
                    "delay_ticks":   50,
                    "redo_cam":      True,
                },
                reasoning=(
                    "Incorrect tool in spindle — safety stop and correct before resuming."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort if correct tool cannot be sourced promptly.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_tool_deflection(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        err = ctx.extra.get("measured_error_mm", 0.05)
        return [
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -20.0},
                reasoning=(
                    f"Tool deflection error {err:.3f}mm. "
                    "Slow feedrate by 20% to reduce deflection forces."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -25.0},
                reasoning=(
                    "Additional feed reduction (25%) for larger deflection."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="change_tool", risk_level="safe",
                parameters={},
                reasoning=(
                    "Change to a stiffer / shorter tool to eliminate deflection."),
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_chatter(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -10.0},
                reasoning=(
                    "Chatter detected. Slow machine feedrate by 10% "
                    "to damp regenerative vibration."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -15.0},
                reasoning="Additional 15% feed reduction if chatter persists.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort if chatter is severe and feed reduction is insufficient.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_spindle_overload(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        over = ctx.extra.get("overload_pct", 25.0)
        return [
            LLMCandidate(
                action="reduce_spindle_speed",
                risk_level="safe",
                parameters={"rpm_delta_pct": -10.0},
                reasoning=(
                    f"Spindle overload +{over:.0f}%. "
                    "Slow spindle speed by 10% to reduce torque."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -15.0},
                reasoning=(
                    "Secondary measure: reduce feed 15% to lower cutting force."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort to prevent spindle damage if overload persists.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_coolant(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [
            LLMCandidate(
                action="stop_request_coolant", risk_level="safe",
                parameters={"delay_ticks": 25},
                reasoning=(
                    "Coolant failure. Stop machining, retract to safe Z-height, "
                    "request coolant (25-tick delay), then restart: spindle on, "
                    "Z down, continue G-code."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="stop_request_coolant", risk_level="safe",
                parameters={"delay_ticks": 25},
                reasoning=(
                    "Stop and request coolant — machine must not cut dry."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort if coolant cannot be restored promptly.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_surface_finish(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        req = ctx.extra.get("required_ra", 1.6)
        act = ctx.extra.get("measured_ra", 3.8)
        return [
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -10.0},
                reasoning=(
                    f"Ra {act:.1f}µm exceeds spec {req:.1f}µm. "
                    "Slow feedrate by 10% to improve surface finish."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -15.0},
                reasoning="Further feed reduction (15%) if finish remains poor.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="rework", risk_level="safe",
                parameters={},
                reasoning=f"Ra {act:.1f}µm too far from spec. Place in rework.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_dimension_error(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        nom = ctx.extra.get("nominal_mm", 50.0)
        act = ctx.extra.get("actual_mm",  50.0)
        tol = ctx.extra.get("tolerance_mm", 0.05)
        err = act - nom
        in_tol = abs(err) <= tol
        return [
            LLMCandidate(
                action="continue" if in_tol else "stop_unload_return",
                risk_level="safe",
                parameters={},
                reasoning=(
                    f"Error {err:+.3f}mm (tol ±{tol:.3f}mm). "
                    + ("Within tolerance — continue." if in_tol else
                       "Out of tolerance — stop machine, unload part, return to factory.")),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="continue" if in_tol else "stop_unload_return",
                risk_level="safe" if in_tol else "risky",
                parameters={},
                reasoning=(
                    "Re-measure and verify tolerance decision." if in_tol else
                    "Part does not meet drawing tolerance — return to factory."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="stop_unload_return", risk_level="safe",
                parameters={},
                reasoning="Stop, unload and return part to factory for inspection.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_power(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        kw   = ctx.extra.get("measured_kw", 0.0)
        over = (kw - config.MACHINE_POWER_LIMIT_KW) / config.MACHINE_POWER_LIMIT_KW * 100
        return [
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -25.0},
                reasoning=(
                    f"Power +{over:.0f}% over limit. "
                    "Slow feedrate by 25% to bring within machine rating."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="risky",
                parameters={"feed_delta_pct": -30.0},
                reasoning="Larger feed reduction (30%) for persistent overpower.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort — power exceedance risk to spindle drive.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_priority(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [
            LLMCandidate(
                action="move_part_up_queue",
                risk_level="safe",
                parameters={},
                reasoning=(
                    "Priority change requested. Move part to front of queue."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="move_part_up_queue", risk_level="safe",
                parameters={},
                reasoning="Expedite requested — reorder queue immediately.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="continue_then_reorder", risk_level="safe",
                parameters={},
                reasoning="Complete current job then reorder queue.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_delivery_delay(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        item      = ctx.extra.get("item", "unknown")
        item_type = ctx.extra.get("item_type", "material")
        delay_hrs = ctx.extra.get("delay_hrs", 24.0)
        if item_type == "tooling":
            return [
                LLMCandidate(
                    action="load_substitute_tool", risk_level="safe",
                    parameters={"item": item, "delay_hrs": delay_hrs},
                    reasoning=(
                        f"Tooling delivery delayed {delay_hrs:.0f}hrs. "
                        "Find closest substitute tool, load in crib, redo CAM, restart."),
                    temperature=0.2, raw="",
                ),
                LLMCandidate(
                    action="load_substitute_tool", risk_level="risky",
                    parameters={"item": item, "delay_hrs": delay_hrs},
                    reasoning="Use substitute tooling and regenerate CAM.",
                    temperature=0.5, raw="",
                ),
                LLMCandidate(
                    action="delay_part_back_queue", risk_level="safe",
                    parameters={"item": item, "delay_hrs": delay_hrs},
                    reasoning=(
                        "Move part to back of queue until correct tooling available."),
                    temperature=0.8, raw="",
                ),
            ]
        return [
            LLMCandidate(
                action="delay_part_back_queue", risk_level="safe",
                parameters={"item": item, "delay_hrs": delay_hrs},
                reasoning=(
                    f"Material delivery delayed {delay_hrs:.0f}hrs. "
                    "Move part to back of queue until factory agent confirms availability."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="delay_part_back_queue", risk_level="safe",
                parameters={"item": item, "delay_hrs": delay_hrs},
                reasoning="Defer affected parts until material is available.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="load_substitute_tool", risk_level="risky",
                parameters={"item": item, "delay_hrs": delay_hrs},
                reasoning="Try substitute material or tool if available.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_quality_hold(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [
            LLMCandidate(
                action="wait_agent_restart", risk_level="safe",
                parameters={},
                reasoning=(
                    "Quality hold placed. Stop machine and wait for "
                    "factory agent to authorise restart."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="wait_agent_restart", risk_level="safe",
                parameters={},
                reasoning="Hold in place until QC inspection complete.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort batch if quality hold cannot be resolved.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_spindle_bearing_oos(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [LLMCandidate(
            action="stop_machine_oos", risk_level="catastrophic",
            parameters={},
            reasoning=(
                "Spindle bearing fault — safety critical. "
                "Stop machine, unload part, return to factory. "
                "Machine OUT OF ORDER until further notice."),
            temperature=0.2, raw="",
        )]

    def _fallback_stop_unload_delay_restart(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [
            LLMCandidate(
                action="stop_unload_delay_restart", risk_level="catastrophic",
                parameters={"delay_ticks": 25},
                reasoning=(
                    f"{ctx.disturbance_key.replace('_',' ')} — safety-critical. "
                    "Stop machine, unload part, return part to factory. "
                    "Delay 25 ticks (supervisor homing), then restart."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="stop_unload_delay_restart", risk_level="catastrophic",
                parameters={"delay_ticks": 25},
                reasoning=(
                    "Safety stop required — unload and delay before restart."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="catastrophic",
                parameters={},
                reasoning="Abort if restart is not possible.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_wait_agent_restart(
            self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [
            LLMCandidate(
                action="wait_agent_restart", risk_level="safe",
                parameters={},
                reasoning=(
                    f"{ctx.disturbance_key.replace('_',' ')} — "
                    "stop machine and wait for restart authorisation from factory agent."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="wait_agent_restart", risk_level="safe",
                parameters={},
                reasoning="Machine halted — awaiting agent restart command.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning="Abort if agent restart is not forthcoming.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_generic(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [
            LLMCandidate(
                action="reduce_feed", risk_level="risky",
                parameters={"feed_delta_pct": -25.0},
                reasoning="Unknown disturbance — conservative feed reduction.",
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="abort",       risk_level="safe",
                parameters={},
                reasoning="Unknown event — abort and await operator.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="rework",      risk_level="safe",
                parameters={},
                reasoning="Place job in rework queue pending investigation.",
                temperature=0.8, raw="",
            ),
        ]

    # ── scoring ───────────────────────────────────────────────────────────────

    # Action desirability table per policy
    _ACTION_SCORE: dict[str, dict[str, float]] = {
        "min_time":      {"continue":1.0,"recalculate":0.9,"reduce_feed":0.7,
                          "reduce_doc":0.6,"add_finish_pass":0.5,"change_tool":0.4,
                          "change_rpm":0.7,"rework":0.2,"abort":0.0,
                          "emergency_stop":0.0,"pause_and_swap":0.6,
                          "finish_section_then_swap":0.8,
                          "continue_then_reorder":0.9,
                          "continue_dry":0.5,
                          # new actions
                          "pause_redo_cam":0.8,
                          "request_tool_change":0.7,
                          "change_tool_before_reload":0.7,
                          "stop_unload_request_tool":0.5,
                          "stop_unload_next_part":0.4,
                          "stop_unload_return":0.3,
                          "stop_request_coolant":0.6,
                          "reduce_spindle_speed":0.7,
                          "stop_machine_oos":0.0,
                          "stop_unload_delay_restart":0.3,
                          "wait_agent_restart":0.2,
                          "move_part_up_queue":0.9,
                          "delay_part_back_queue":0.5,
                          "load_substitute_tool":0.6},
        "min_cost":      {"continue":1.0,"recalculate":0.8,"reduce_feed":0.8,
                          "reduce_doc":0.7,"change_tool":0.5,"add_finish_pass":0.6,
                          "change_rpm":0.7,"rework":0.2,"abort":0.0,
                          "emergency_stop":0.0,"pause_and_swap":0.5,
                          "finish_section_then_swap":0.7,
                          "continue_then_reorder":0.9,
                          "continue_dry":0.5,
                          "pause_redo_cam":0.7,
                          "request_tool_change":0.6,
                          "change_tool_before_reload":0.6,
                          "stop_unload_request_tool":0.4,
                          "stop_unload_next_part":0.3,
                          "stop_unload_return":0.2,
                          "stop_request_coolant":0.5,
                          "reduce_spindle_speed":0.7,
                          "stop_machine_oos":0.0,
                          "stop_unload_delay_restart":0.2,
                          "wait_agent_restart":0.2,
                          "move_part_up_queue":0.8,
                          "delay_part_back_queue":0.4,
                          "load_substitute_tool":0.5},
        "best_finish":   {"add_finish_pass":1.0,"reduce_feed":0.9,"recalculate":0.8,
                          "change_tool":0.7,"continue":0.5,"reduce_doc":0.3,
                          "rework":0.3,"abort":0.1,"emergency_stop":0.0,
                          "change_rpm":0.6,"pause_and_swap":0.4,
                          "finish_section_then_swap":0.6,
                          "continue_then_reorder":0.5,
                          "continue_dry":0.2,
                          "pause_redo_cam":0.8,
                          "request_tool_change":0.6,
                          "change_tool_before_reload":0.7,
                          "stop_unload_request_tool":0.5,
                          "stop_unload_next_part":0.3,
                          "stop_unload_return":0.3,
                          "stop_request_coolant":0.4,
                          "reduce_spindle_speed":0.7,
                          "stop_machine_oos":0.0,
                          "stop_unload_delay_restart":0.2,
                          "wait_agent_restart":0.2,
                          "move_part_up_queue":0.6,
                          "delay_part_back_queue":0.4,
                          "load_substitute_tool":0.6},
        "max_tool_life": {"change_tool":1.0,"reduce_feed":0.9,"reduce_doc":0.9,
                          "recalculate":0.8,"change_rpm":0.8,"continue":0.4,
                          "add_finish_pass":0.6,"rework":0.2,"abort":0.0,
                          "emergency_stop":0.0,"pause_and_swap":0.5,
                          "finish_section_then_swap":0.6,
                          "continue_then_reorder":0.7,
                          "continue_dry":0.3,
                          "pause_redo_cam":0.7,
                          "request_tool_change":0.8,
                          "change_tool_before_reload":0.9,
                          "stop_unload_request_tool":0.6,
                          "stop_unload_next_part":0.3,
                          "stop_unload_return":0.2,
                          "stop_request_coolant":0.4,
                          "reduce_spindle_speed":0.7,
                          "stop_machine_oos":0.0,
                          "stop_unload_delay_restart":0.2,
                          "wait_agent_restart":0.2,
                          "move_part_up_queue":0.7,
                          "delay_part_back_queue":0.4,
                          "load_substitute_tool":0.7},
        "multi_objective":{"recalculate":0.9,"reduce_feed":0.8,"reduce_doc":0.75,
                           "change_tool":0.7,"add_finish_pass":0.7,"continue":0.6,
                           "change_rpm":0.75,"rework":0.25,"abort":0.05,
                           "emergency_stop":0.0,"pause_and_swap":0.55,
                           "finish_section_then_swap":0.7,
                           "continue_then_reorder":0.8,
                           "continue_dry":0.4,
                           "pause_redo_cam":0.8,
                           "request_tool_change":0.7,
                           "change_tool_before_reload":0.75,
                           "stop_unload_request_tool":0.5,
                           "stop_unload_next_part":0.35,
                           "stop_unload_return":0.25,
                           "stop_request_coolant":0.55,
                           "reduce_spindle_speed":0.75,
                           "stop_machine_oos":0.0,
                           "stop_unload_delay_restart":0.25,
                           "wait_agent_restart":0.2,
                           "move_part_up_queue":0.8,
                           "delay_part_back_queue":0.45,
                           "load_substitute_tool":0.6},
    }
    _RISK_PENALTY: dict[str, float] = {
        "safe": 0.0, "risky": -0.15, "catastrophic": -0.50,
    }

    def score(self, candidate: LLMCandidate, policy: str) -> float:
        table = self._ACTION_SCORE.get(policy, self._ACTION_SCORE["min_time"])
        base  = table.get(candidate.action, 0.3)
        pen   = self._RISK_PENALTY.get(candidate.risk_level, 0.0)
        return round(max(0.0, base + pen), 3)


# ── module-level singleton ────────────────────────────────────────────────────

ENGINE = DisturbanceEngine()

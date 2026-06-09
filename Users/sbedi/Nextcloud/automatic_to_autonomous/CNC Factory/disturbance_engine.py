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
  MATERIAL  harder | softer | dimension_oversize | dimension_undersize |
            wrong_grade | inclusion
  PROCESS   spindle_overload | coolant_failure | surface_finish_poor |
            dimension_error | power_exceedance | chatter_vibration
  MACHINE   spindle_bearing | axis_fault | fixture_loose | power_fault |
            tool_changer_fault
  SCHEDULE  priority_change | delivery_delay | batch_change | quality_hold
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
        short_desc="Insert broken mid-cut — spindle halted",
        possible_actions=("abort", "rework", "transfer", "emergency_stop"),
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
        short_desc="Insert wear approaching end of usable life",
        possible_actions=("change_tool", "reduce_feed", "continue"),
        gui_fields=(
            GUIField("tool_id",  "Worn tool ID", "text",  "T3"),
            GUIField("wear_pct", "Wear %",       "float", 92.0),
        ),
    ),
    "tool_wrong_type": DisturbanceSpec(
        key="tool_wrong_type", category=Category.TOOL,
        label="Wrong tool loaded",
        short_desc="Actual tool in spindle differs from program spec",
        possible_actions=("abort", "change_tool", "rework"),
        gui_fields=(
            GUIField("expected_tool", "Expected tool ID", "text", "T3"),
            GUIField("loaded_tool",   "Loaded tool ID",   "text", "T4"),
        ),
    ),
    "tool_chatter": DisturbanceSpec(
        key="tool_chatter", category=Category.TOOL,
        label="Tool chatter / resonance",
        short_desc="High-frequency vibration — audible chatter in cut",
        possible_actions=("change_rpm", "reduce_feed", "reduce_doc",
                          "change_path", "abort"),
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
        short_desc="Long-reach tool bending causes dimensional error",
        possible_actions=("reduce_doc", "reduce_feed", "change_tool",
                          "add_finish_pass"),
        gui_fields=(
            GUIField("measured_error_mm", "Measured error mm", "float", 0.05),
        ),
    ),

    # ── MATERIAL ──────────────────────────────────────────────────────────────
    "material_harder": DisturbanceSpec(
        key="material_harder", category=Category.MATERIAL,
        label="Material harder than specified",
        short_desc="Stock harder/tougher than the job specifies",
        possible_actions=("recalculate", "reduce_feed", "change_tool",
                          "abort"),
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
        short_desc="Stock softer — risk of built-up edge and burring",
        possible_actions=("recalculate", "increase_feed", "reduce_rpm",
                          "continue"),
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
        short_desc="Blank dimensions exceed program stock assumption",
        possible_actions=("add_rough_pass", "abort", "rework"),
        gui_fields=(
            GUIField("axis",          "Oversize axis", "choice", "Z",
                     choices=["X","Y","Z","XY"]),
            GUIField("excess_mm",     "Excess mm",    "float", 5.0),
        ),
    ),
    "material_dimension_undersize": DisturbanceSpec(
        key="material_dimension_undersize", category=Category.MATERIAL,
        label="Stock under-size",
        short_desc="Blank too small — insufficient material for features",
        possible_actions=("rework", "abort", "skip_feature")),
    "material_wrong_grade": DisturbanceSpec(
        key="material_wrong_grade", category=Category.MATERIAL,
        label="Wrong material grade / alloy",
        short_desc="Correct material class but wrong alloy/temper",
        possible_actions=("recalculate", "reduce_feed", "abort"),
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
        short_desc="Localised hard spot — sudden force spike",
        possible_actions=("reduce_doc", "reduce_feed", "abort", "rework"),
        gui_fields=(
            GUIField("depth_mm", "Estimated depth of inclusion mm", "float", 10.0),
        ),
    ),

    # ── PROCESS ───────────────────────────────────────────────────────────────
    "spindle_overload": DisturbanceSpec(
        key="spindle_overload", category=Category.PROCESS,
        label="Spindle overload",
        short_desc="Spindle current / torque exceeds limit",
        possible_actions=("reduce_feed", "reduce_doc", "abort",
                          "change_tool"),
        gui_fields=(
            GUIField("overload_pct", "Overload above rated %", "float", 25.0),
            GUIField("measured_kw",  "Measured power kW",      "float", 9.5),
        ),
    ),
    "coolant_failure": DisturbanceSpec(
        key="coolant_failure", category=Category.PROCESS,
        label="Coolant failure",
        short_desc="Coolant pressure loss or pump fault",
        possible_actions=("abort", "continue_dry", "reduce_feed"),
        safety_critical=False,   # depends on tool/material
        gui_fields=(
            GUIField("coolant_type", "Coolant type",
                     "choice", "flood", choices=["flood","mist","through_tool"]),
            GUIField("partial_loss", "Partial loss (not total)", "bool", False),
        ),
    ),
    "surface_finish_poor": DisturbanceSpec(
        key="surface_finish_poor", category=Category.PROCESS,
        label="Surface finish below spec",
        short_desc="Ra exceeds tolerance — cosmetic / functional fail",
        possible_actions=("add_finish_pass", "reduce_feed",
                          "change_tool", "rework"),
        gui_fields=(
            GUIField("required_ra",   "Required Ra µm",  "float", 1.6),
            GUIField("measured_ra",   "Measured Ra µm",  "float", 3.8),
        ),
    ),
    "dimension_error": DisturbanceSpec(
        key="dimension_error", category=Category.PROCESS,
        label="Dimension out of tolerance",
        short_desc="Measured feature outside drawing tolerance",
        possible_actions=("add_finish_pass", "rework", "continue"),
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
        short_desc=f"Machine rated {config.MACHINE_POWER_LIMIT_KW} kW — "
                   "power spike detected",
        possible_actions=("reduce_feed", "reduce_doc", "abort"),
        gui_fields=(
            GUIField("measured_kw",  "Measured power kW",  "float", 8.2),
        ),
    ),
    "chatter_vibration": DisturbanceSpec(
        key="chatter_vibration", category=Category.PROCESS,
        label="Chatter / regenerative vibration",
        short_desc="Resonance between tool and workpiece causes chatter",
        possible_actions=("change_rpm", "reduce_feed", "reduce_doc",
                          "add_damper", "abort"),
        gui_fields=(
            GUIField("frequency_hz", "Frequency Hz (if known)", "float", 0.0),
        ),
    ),

    # ── MACHINE ───────────────────────────────────────────────────────────────
    "spindle_bearing": DisturbanceSpec(
        key="spindle_bearing", category=Category.MACHINE,
        label="Spindle bearing noise / fault",
        short_desc="Bearing wear or damage — vibration and noise",
        possible_actions=("abort", "reduce_rpm", "emergency_stop"),
        safety_critical=True,
    ),
    "axis_fault": DisturbanceSpec(
        key="axis_fault", category=Category.MACHINE,
        label="Servo / axis fault",
        short_desc="Following error or servo trip on X/Y/Z axis",
        possible_actions=("abort", "emergency_stop"),
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
        short_desc="Part movement detected — safety critical",
        possible_actions=("emergency_stop",),
        safety_critical=True,
    ),
    "power_fault": DisturbanceSpec(
        key="power_fault", category=Category.MACHINE,
        label="Power / voltage fluctuation",
        short_desc="Mains power anomaly — drives tripped",
        possible_actions=("abort", "wait_and_resume"),
        gui_fields=(
            GUIField("duration_s", "Duration seconds", "float", 2.0),
        ),
    ),
    "tool_changer_fault": DisturbanceSpec(
        key="tool_changer_fault", category=Category.MACHINE,
        label="Automatic tool-changer fault",
        short_desc="ATC arm jam or tool not clamped",
        possible_actions=("abort", "manual_intervention", "skip_tool_change"),
    ),

    # ── SCHEDULE ──────────────────────────────────────────────────────────────
    "priority_change": DisturbanceSpec(
        key="priority_change", category=Category.SCHEDULE,
        label="Customer priority change",
        short_desc="Higher-priority order must be expedited",
        possible_actions=("continue_then_reorder", "pause_and_swap",
                          "finish_section_then_swap"),
        gui_fields=(
            GUIField("new_priority_job",  "New priority job seed", "text", ""),
            GUIField("reason",            "Reason",                "text",
                     "Customer expedite request"),
        ),
    ),
    "delivery_delay": DisturbanceSpec(
        key="delivery_delay", category=Category.SCHEDULE,
        label="Material / tooling delivery delay",
        short_desc="Expected stock or tools delayed — affect future jobs",
        possible_actions=("continue", "reorder_queue", "notify_factory"),
        gui_fields=(
            GUIField("item",       "Delayed item",    "text",  "T3 x5 pcs"),
            GUIField("delay_hrs",  "Delay hours",     "float", 24.0),
        ),
    ),
    "quality_hold": DisturbanceSpec(
        key="quality_hold", category=Category.SCHEDULE,
        label="Quality hold — inspection required",
        short_desc="QC has placed a hold on this batch",
        possible_actions=("pause_and_inspect", "rework", "abort"),
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
            "  recalculate   — recompute feeds/speeds for actual material and continue\n"
            "  reduce_feed   — apply a conservative feed-rate reduction and continue\n"
            "  change_tool   — swap to a harder-grade / more robust tool first\n"
            "  abort         — stop job immediately; place in rework queue\n\n"
            "Respond with JSON only (no prose outside the object):\n"
            '{"action":"recalculate|reduce_feed|change_tool|abort",\n'
            ' "parameters":{"feed_rate_mmpm":float,"rpm":float,'
            '"depth_of_cut_mm":float,"tool_id":"T3"},\n'
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
            "Available actions: abort | rework | transfer | emergency_stop\n\n"
            'Respond JSON only:\n'
            '{"action":"abort|rework|transfer","target_machine":"M0x or null",'
            '"parameters":{},"risk_level":"safe|risky|catastrophic",'
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
            "Actions: change_tool (at next section boundary) | "
            "reduce_feed (extend life) | continue (accept risk)\n\n"
            'Respond JSON only:\n'
            '{"action":"change_tool|reduce_feed|continue","parameters":{},'
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
            f"  Spindle now: {ctx.rpm_original:.0f} RPM\n"
            f"  Feed now   : {ctx.feed_original:.0f} mm/min\n"
            f"  Policy     : {ctx.factory_policy}\n\n"
            "Actions: change_rpm | reduce_feed | reduce_doc | change_path | abort\n"
            "Tip: change_rpm ±15% breaks resonance without losing much productivity.\n\n"
            'Respond JSON only:\n'
            '{"action":"change_rpm|reduce_feed|reduce_doc|abort",'
            '"parameters":{"rpm_delta_pct":float,"feed_delta_pct":float},'
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
            "Actions: reduce_feed | reduce_doc | abort | change_tool\n\n"
            'Respond JSON only:\n'
            '{"action":"reduce_feed|reduce_doc|abort",'
            '"parameters":{"feed_delta_pct":float,"doc_delta_pct":float},'
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
            "Rules:\n"
            "  - HSS tools without coolant → tool failure risk very high.\n"
            "  - Carbide tools with short remaining time may continue dry.\n"
            "  - Titanium / Inconel: ALWAYS requires coolant.\n\n"
            "Actions: abort | continue_dry | reduce_feed\n\n"
            'Respond JSON only:\n'
            '{"action":"abort|continue_dry|reduce_feed","parameters":{},'
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
            "Actions: add_finish_pass | reduce_feed | change_tool | rework\n\n"
            'Respond JSON only:\n'
            '{"action":"add_finish_pass|reduce_feed|rework",'
            '"parameters":{"feed_delta_pct":float},'
            '"risk_level":"safe|risky|catastrophic","reasoning":"<one sentence>"}'
        )

    def _prompt_dimension_error(self, ctx: DisturbanceContext) -> str:
        nom = ctx.extra.get("nominal_mm", 50.0)
        act = ctx.extra.get("actual_mm",  50.0)
        tol = ctx.extra.get("tolerance_mm", 0.05)
        err = act - nom
        feat = ctx.extra.get("feature", "unknown feature")
        return (
            "You are the factory supervisor AI for an autonomous CNC machining plant.\n\n"
            f"QUALITY ALERT: Dimension error on {ctx.machine_id} — {feat}.\n"
            f"  Nominal          : {nom:.3f} mm\n"
            f"  Measured         : {act:.3f} mm\n"
            f"  Error            : {err:+.3f} mm  "
            f"(tolerance ±{tol:.3f} mm)\n"
            f"  Over-cut (−)     : {'YES — material already removed' if err < 0 else 'NO'}\n"
            f"  Under-cut (+)    : {'YES — material to remove' if err > 0 else 'NO'}\n"
            f"  Policy           : {ctx.factory_policy}\n\n"
            "Actions: add_finish_pass (if undercut) | rework | continue\n\n"
            'Respond JSON only:\n'
            '{"action":"add_finish_pass|rework|continue","parameters":{},'
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
            "Actions: reduce_feed | reduce_doc | abort\n\n"
            'Respond JSON only:\n'
            '{"action":"reduce_feed|reduce_doc|abort",'
            '"parameters":{"feed_delta_pct":float,"doc_delta_pct":float},'
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
            "Actions: continue_then_reorder | pause_and_swap | "
            "finish_section_then_swap\n\n"
            'Respond JSON only:\n'
            '{"action":"continue_then_reorder|pause_and_swap|finish_section_then_swap",'
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
            "material_harder":      self._fallback_material_harder,
            "material_softer":      self._fallback_material_softer,
            "material_wrong_grade": self._fallback_material_harder,
            "tool_breakage":        self._fallback_tool_breakage,
            "tool_wear_critical":   self._fallback_tool_wear,
            "tool_chatter":         self._fallback_chatter,
            "chatter_vibration":    self._fallback_chatter,
            "spindle_overload":     self._fallback_spindle_overload,
            "coolant_failure":      self._fallback_coolant,
            "surface_finish_poor":  self._fallback_surface_finish,
            "dimension_error":      self._fallback_dimension_error,
            "power_exceedance":     self._fallback_power,
            "priority_change":      self._fallback_priority,
            "fixture_loose":        self._fallback_safety_critical,
            "spindle_bearing":      self._fallback_safety_critical,
            "axis_fault":           self._fallback_safety_critical,
        }
        fn = dispatch.get(ctx.disturbance_key, self._fallback_generic)
        return fn(ctx)

    # material harder/wrong-grade
    def _fallback_material_harder(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        ratio = ctx.kc_ratio or 1.0
        candidates: list[LLMCandidate] = []

        if ratio <= 1.0:
            # softer — shouldn't reach here but handle gracefully
            return self._fallback_material_softer(ctx)

        # candidate 1 (conservative) — abort if power would exceed limit
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
                action="recalculate", risk_level="safe",
                parameters={
                    "feed_rate_mmpm":   ctx.feed_required,
                    "rpm":              ctx.rpm_required,
                    "depth_of_cut_mm":  3.0 / math.sqrt(ratio),
                },
                reasoning=(
                    f"Recomputed feeds/speeds for {ctx.actual_material.replace('_',' ')}: "
                    f"feed {ctx.feed_original:.0f}→{ctx.feed_required:.0f} mm/min "
                    f"(−{ctx.feed_reduction_pct:.0f}%), "
                    f"RPM {ctx.rpm_original:.0f}→{ctx.rpm_required:.0f}. "
                    "Power within machine limits. Continue safely."),
                temperature=0.2, raw="",
            ))

        # candidate 2 (moderate) — proportional reduction
        feed_adj  = ctx.feed_original / ratio
        rpm_adj   = ctx.rpm_original  / ratio
        candidates.append(LLMCandidate(
            action="reduce_feed", risk_level="risky",
            parameters={
                "feed_rate_mmpm":  round(feed_adj, 1),
                "rpm":             round(rpm_adj,  1),
                "override_pct":    round(100.0 / ratio, 1),
            },
            reasoning=(
                f"Simple proportional reduction: feed ×{1/ratio:.2f}, "
                f"RPM ×{1/ratio:.2f}. Practical quick fix; "
                "verify with first cut."),
            temperature=0.5, raw="",
        ))

        # candidate 3 (optimistic / aggressive)
        if ratio < 2.0 and ctx.active_tool_life > 40:
            candidates.append(LLMCandidate(
                action="recalculate", risk_level="risky",
                parameters={
                    "feed_rate_mmpm": ctx.feed_required * 1.1,  # 10% higher
                    "rpm":            ctx.rpm_required,
                    "depth_of_cut_mm": 2.5,
                },
                reasoning=(
                    f"Aggressive recalculation: slightly higher than minimum "
                    f"({ctx.feed_required*1.1:.0f} mm/min). "
                    "Tool has sufficient life for limited trial."),
                temperature=0.8, raw="",
            ))
        else:
            candidates.append(LLMCandidate(
                action="abort", risk_level="safe",
                parameters={},
                reasoning=(
                    f"Kc ratio {ratio:.2f}× is too high for available "
                    f"tooling (life={ctx.active_tool_life:.0f}%). Abort."),
                temperature=0.8, raw="",
            ))

        return candidates[:3]

    def _fallback_material_softer(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        ratio = ctx.kc_ratio or 1.0
        return [
            LLMCandidate(
                action="recalculate", risk_level="safe",
                parameters={
                    "feed_rate_mmpm": ctx.feed_required,
                    "rpm":            ctx.rpm_required,
                },
                reasoning=(
                    f"Material softer (Kc ratio {ratio:.2f}×): "
                    f"increase feed {ctx.feed_original:.0f}→"
                    f"{ctx.feed_required:.0f} mm/min, "
                    "reduce RPM to avoid BUE."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="continue", risk_level="safe",
                parameters={},
                reasoning=(
                    "Small Kc difference — continue with current "
                    "parameters. Monitor surface finish."),
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -10.0},
                reasoning=(
                    "Slightly reduce feed to minimise BUE risk on "
                    "softer material."),
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_tool_breakage(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        no_stock = ctx.extra.get("no_stock", True)
        return [
            LLMCandidate(
                action="rework",          risk_level="safe",
                parameters={},
                reasoning=(
                    "Tool broken; "
                    + ("no replacement in stock. " if no_stock else "")
                    + "Place job in rework queue."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="abort",           risk_level="safe",
                parameters={},
                reasoning="Catastrophic tool failure — abort job immediately.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="emergency_stop",  risk_level="safe",
                parameters={},
                reasoning="Emergency stop; await manual inspection before resuming.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_tool_wear(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        tool  = ctx.extra.get("tool_id", "T3")
        inv   = ctx.inventory.get(tool, 0)
        wear  = ctx.extra.get("wear_pct", 90.0)
        return [
            LLMCandidate(
                action="change_tool" if inv > 0 else "reduce_feed",
                risk_level="safe",
                parameters={},
                reasoning=(
                    f"Tool at {wear:.0f}% wear. "
                    + (f"Replace at next section boundary (stock: {inv})."
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
                action="continue", risk_level="risky",
                parameters={},
                reasoning=f"Tool at {wear:.0f}% — within acceptable range, continue.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_chatter(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        rpm = ctx.rpm_original or 3000.0
        return [
            LLMCandidate(
                action="change_rpm", risk_level="safe",
                parameters={"rpm_delta_pct": -12.0,
                             "feed_delta_pct": -10.0},
                reasoning=(
                    f"Reduce RPM by 12% ({rpm*0.88:.0f} RPM) and feed by 10% "
                    "to break resonance. Most effective first step."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_doc", risk_level="safe",
                parameters={"doc_delta_pct": -25.0},
                reasoning="Reduce depth of cut by 25% to lower dynamic chip load.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="change_rpm", risk_level="risky",
                parameters={"rpm_delta_pct": +15.0},
                reasoning=(
                    f"Try increasing RPM by 15% ({rpm*1.15:.0f} RPM) "
                    "to move above chatter frequency."),
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_spindle_overload(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        over = ctx.extra.get("overload_pct", 25.0)
        feed_cut = min(50.0, over * 1.5)
        doc_cut  = min(40.0, over * 1.0)
        return [
            LLMCandidate(
                action="reduce_feed" if over < 40 else "abort",
                risk_level="safe" if over < 40 else "risky",
                parameters={"feed_delta_pct": -feed_cut,
                             "doc_delta_pct":  -doc_cut},
                reasoning=(
                    f"Spindle overload +{over:.0f}%. "
                    f"Reduce feed by {feed_cut:.0f}%, "
                    f"DOC by {doc_cut:.0f}%."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_doc", risk_level="risky",
                parameters={"doc_delta_pct": -30.0},
                reasoning="Reduce depth of cut 30% to drop cutting force.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort",      risk_level="safe",
                parameters={},
                reasoning="Spindle overload — abort to prevent damage.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_coolant(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        mat     = ctx.original_material or config.DEFAULT_MATERIAL
        hard    = mat in {"titanium_ti64", "inconel", "stainless_304",
                          "alloy_steel_4140", "tool_steel_d2"}
        partial = ctx.extra.get("partial_loss", False)
        short   = len(ctx.sections_remaining) <= 2
        return [
            LLMCandidate(
                action="abort" if hard else ("continue_dry" if short else "abort"),
                risk_level="safe",
                parameters={},
                reasoning=(
                    "Coolant failure. "
                    + ("Hard material requires coolant — abort."
                       if hard else
                       ("Short remaining time — continue dry at reduced feed."
                        if short else "Abort to prevent thermal damage."))),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="risky",
                parameters={"feed_delta_pct": -50.0},
                reasoning="50% feed reduction limits heat generation. "
                          "Monitor spindle temperature.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="abort",       risk_level="safe",
                parameters={},
                reasoning="Abort to protect tool and part from thermal damage.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_surface_finish(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        req = ctx.extra.get("required_ra", 1.6)
        act = ctx.extra.get("measured_ra", 3.8)
        return [
            LLMCandidate(
                action="add_finish_pass", risk_level="safe",
                parameters={"feed_delta_pct": -20.0},
                reasoning=(
                    f"Ra {act:.1f}µm exceeds spec {req:.1f}µm. "
                    "Add finish pass at −20% feed."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -25.0},
                reasoning="Reduce feed 25% for remaining passes to improve Ra.",
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
        overcut = err < -tol
        undercut = err > tol
        return [
            LLMCandidate(
                action="rework" if overcut else ("add_finish_pass" if undercut else "continue"),
                risk_level="safe",
                parameters={},
                reasoning=(
                    f"Error {err:+.3f}mm (tol ±{tol:.3f}mm). "
                    + ("Over-cut: material removed — rework." if overcut else
                       "Under-cut: add finish pass to reach nominal."
                       if undercut else "Within tolerance — continue.")),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="continue", risk_level="risky",
                parameters={},
                reasoning="Continue and re-measure at next checkpoint.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="rework", risk_level="safe",
                parameters={},
                reasoning="Place part in rework for inspection.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_power(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        kw   = ctx.extra.get("measured_kw", 0.0)
        over = (kw - config.MACHINE_POWER_LIMIT_KW) / config.MACHINE_POWER_LIMIT_KW * 100
        fd   = min(40.0, over * 1.2)
        return [
            LLMCandidate(
                action="reduce_feed", risk_level="safe",
                parameters={"feed_delta_pct": -fd,
                             "doc_delta_pct": -fd * 0.5},
                reasoning=(
                    f"Power +{over:.0f}% over limit. "
                    f"Reduce feed {fd:.0f}% to bring within {config.MACHINE_POWER_LIMIT_KW}kW."),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="reduce_doc", risk_level="risky",
                parameters={"doc_delta_pct": -30.0},
                reasoning="Reduce depth of cut 30%.",
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
        est_rem = len(ctx.sections_remaining) * 30
        return [
            LLMCandidate(
                action=("finish_section_then_swap" if est_rem < 120
                        else "pause_and_swap"),
                risk_level="safe",
                parameters={},
                reasoning=(
                    f"~{est_rem}s remaining on current job. "
                    + ("Finish current section then swap."
                       if est_rem < 120 else
                       "Pause immediately and pick up priority job.")),
                temperature=0.2, raw="",
            ),
            LLMCandidate(
                action="continue_then_reorder", risk_level="safe",
                parameters={},
                reasoning="Complete current job then reorder queue.",
                temperature=0.5, raw="",
            ),
            LLMCandidate(
                action="pause_and_swap", risk_level="risky",
                parameters={},
                reasoning="Immediate swap; partial job saved for later.",
                temperature=0.8, raw="",
            ),
        ]

    def _fallback_safety_critical(self, ctx: DisturbanceContext) -> list[LLMCandidate]:
        return [LLMCandidate(
            action="emergency_stop", risk_level="catastrophic",
            parameters={},
            reasoning=(
                f"{ctx.disturbance_key.replace('_',' ')} — "
                "safety-critical event. Emergency stop required."),
            temperature=0.2, raw="",
        )]

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
                          "continue_dry":0.5},
        "min_cost":      {"continue":1.0,"recalculate":0.8,"reduce_feed":0.8,
                          "reduce_doc":0.7,"change_tool":0.5,"add_finish_pass":0.6,
                          "change_rpm":0.7,"rework":0.2,"abort":0.0,
                          "emergency_stop":0.0,"pause_and_swap":0.5,
                          "finish_section_then_swap":0.7,
                          "continue_then_reorder":0.9,
                          "continue_dry":0.5},
        "best_finish":   {"add_finish_pass":1.0,"reduce_feed":0.9,"recalculate":0.8,
                          "change_tool":0.7,"continue":0.5,"reduce_doc":0.3,
                          "rework":0.3,"abort":0.1,"emergency_stop":0.0,
                          "change_rpm":0.6,"pause_and_swap":0.4,
                          "finish_section_then_swap":0.6,
                          "continue_then_reorder":0.5,
                          "continue_dry":0.2},
        "max_tool_life": {"change_tool":1.0,"reduce_feed":0.9,"reduce_doc":0.9,
                          "recalculate":0.8,"change_rpm":0.8,"continue":0.4,
                          "add_finish_pass":0.6,"rework":0.2,"abort":0.0,
                          "emergency_stop":0.0,"pause_and_swap":0.5,
                          "finish_section_then_swap":0.6,
                          "continue_then_reorder":0.7,
                          "continue_dry":0.3},
        "multi_objective":{"recalculate":0.9,"reduce_feed":0.8,"reduce_doc":0.75,
                           "change_tool":0.7,"add_finish_pass":0.7,"continue":0.6,
                           "change_rpm":0.75,"rework":0.25,"abort":0.05,
                           "emergency_stop":0.0,"pause_and_swap":0.55,
                           "finish_section_then_swap":0.7,
                           "continue_then_reorder":0.8,
                           "continue_dry":0.4},
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

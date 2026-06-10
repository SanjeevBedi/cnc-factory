"""
factory_agent.py — Phase 7 of the CNC Factory pipeline.

Responsibilities
----------------
1.  Orchestrate all CNC agents (Phase 6) and the Scheduler (Phase 5).
2.  Error diagnosis via OpenAI LLM (with rule-based fallback when no key).
3.  Tool-crib management — factory dispatches fresh tools to machines.
4.  New-job creation — submit_new_seed() runs the full Phase 1-4 pipeline.
5.  JSON-serialisable factory-state snapshot for dashboards and tests.

LLM flow (unified_plan.tex §7):
    error context → build prompt → call OpenAI (3 temps) → score against policy
    → pick best → dispatch FactoryCommand → CncAgent.handle_factory_response()

Sources: Autonomous_Factory_Sim.docx, unified_plan.tex §7, config.py
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import config
from cnc_agent import (
    CncAgent, ToolRecord,
    build_default_crib,
    _MACHINE_TOOL_DATA,
)
from scheduler import Scheduler, Job, make_job

# optional OpenAI
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None  # type: ignore[assignment,misc]


# ── Data structures -----------------------------------------------------------

@dataclass
class FactoryCommand:
    command_id:        str
    target_machine_id: str
    action:            str          # continue|reduce_feed|abort|rework|replace_tool
    payload:           dict = field(default_factory=dict)
    priority:          str  = "normal"
    tick_issued:       int  = 0


@dataclass
class ErrorContext:
    machine_id:         str
    error_description:  str
    error_section:      str
    error_gcode_line:   str
    machine_status:     str
    tool_life_pct:      float
    sections_completed: list
    queue_depth:        int
    factory_policy:     str
    tick:               int


@dataclass
class LLMResponse:
    response_id:  str
    raw_text:     str
    action:       str
    parameters:   dict  = field(default_factory=dict)
    risk_level:   str   = "safe"   # safe | risky | catastrophic
    policy_score: float = 0.0


@dataclass
class FactoryTickResult:
    tick:             int
    t_real_s:         float
    commands_sent:    list = field(default_factory=list)
    errors_processed: int  = 0
    tools_replaced:   int  = 0
    jobs_created:     int  = 0
    jobs_completed:   int  = 0
    jobs_in_rework:   int  = 0


# ── Policy scoring ------------------------------------------------------------

_POLICY_ACTION_SCORES: dict[str, dict[str, float]] = {
    "min_time": {
        "continue": 1.0, "reduce_feed": 0.5,
        "rework": 0.2, "abort": 0.0, "replace_tool": 0.3,
    },
    "min_cost": {
        "continue": 0.8, "reduce_feed": 0.7,
        "rework": 0.2, "abort": 0.0, "replace_tool": 0.4,
    },
    "best_finish": {
        "continue": 0.5, "reduce_feed": 1.0,
        "rework": 0.3, "abort": 0.0, "replace_tool": 0.6,
    },
    "max_tool_life": {
        "continue": 0.4, "reduce_feed": 1.0,
        "rework": 0.3, "abort": 0.0, "replace_tool": 0.9,
    },
}

_RISK_PENALTY: dict[str, float] = {
    "safe": 0.0, "risky": -0.1, "catastrophic": -0.5,
}


# ── Factory Agent -------------------------------------------------------------

class FactoryAgent:
    """
    Top-level orchestrator: Scheduler + 4 CncAgents + OpenAI LLM.

    Parameters
    ----------
    agents          : list[CncAgent]
    scheduler       : Scheduler
    policy          : "min_time" | "min_cost" | "best_finish" | "max_tool_life"
    openai_api_key  : str | None  — if None or openai not installed, fallback is used
    """

    def __init__(
        self,
        agents:         list[CncAgent],
        scheduler:      Scheduler,
        policy:         str = "min_time",
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.agents    = agents
        self.scheduler = scheduler
        self.policy    = policy
        self.rework_queue: deque = agents[0].rework_queue if agents else deque()

        self._use_llm = (
            openai_api_key is not None
            and bool(openai_api_key.strip())
            and _OPENAI_AVAILABLE
        )
        self._llm_client = (
            _OpenAI(api_key=openai_api_key) if self._use_llm else None  # type: ignore
        )

        self.tool_inventory: dict[str, list[ToolRecord]] = {}
        self._seed_inventory()
        self.command_log: list[FactoryCommand] = []

        self.total_errors_processed = 0
        self.total_tools_replaced   = 0
        self.total_jobs_completed   = 0

    # ── Tick ------------------------------------------------------------------

    def tick(self) -> FactoryTickResult:
        """
        One simulation tick:
          1. Run scheduler tick (assigns jobs to scheduler Machine objects).
          2. Mirror scheduler assignments into CncAgent.part_queue so the
             GUI can access gcode_lines and drive the animation.
          3. Run per-agent tick (error protocol only — NOT execute_job).
          4. Process errors and manage tool cribs.
        """
        sched_result = self.scheduler.tick_once()
        tick_num     = self.scheduler.tick
        t_real_s     = self.scheduler.t_real_s

        ftr = FactoryTickResult(
            tick           = tick_num,
            t_real_s       = t_real_s,
            jobs_completed = len(sched_result.jobs_finished),
        )
        self.total_jobs_completed += len(sched_result.jobs_finished)

        # ── Mirror scheduler → agent queues ─────────────────────────────
        # The scheduler assigns jobs to scheduler.Machine objects.
        # The GUI uses agent.part_queue to read gcode_lines for animation.
        # Bridge the gap: when the scheduler assigns a new job, push it
        # into the matching agent's part_queue (if not already there).
        sched_machines = {m.machine_id: m for m in self.scheduler.machines}
        for agent in self.agents:
            sm = sched_machines.get(agent.machine_id)
            if sm is None or sm.current_job is None:
                continue
            job = sm.current_job
            # Only enqueue if this job isn't already in the agent's queue
            already_queued = any(
                j.job_id == job.job_id for j in agent.part_queue
            )
            if not already_queued and agent.current_job is None:
                agent.part_queue.append(job)
                agent.status = "running"   # mark as active for display

        # ── Per-agent error protocol (skip execute_job — GUI drives G-code)
        for agent in self.agents:
            agent.current_tick = tick_num
            if agent.status == "awaiting_factory" and agent.active_error:
                agent._tick_error_protocol()

        cmds = self._process_errors(tick_num)
        ftr.commands_sent.extend(cmds)
        ftr.errors_processed = len(cmds)
        self.total_errors_processed += len(cmds)

        rcmds = self._manage_tool_cribs(tick_num)
        ftr.commands_sent.extend(rcmds)
        ftr.tools_replaced = len(rcmds)
        self.total_tools_replaced += len(rcmds)

        ftr.jobs_in_rework = len(self.rework_queue)
        self.command_log.extend(ftr.commands_sent)
        return ftr

    def run(self, n_ticks: int) -> list[FactoryTickResult]:
        return [self.tick() for _ in range(n_ticks)]

    # ── Error processing ------------------------------------------------------

    def _process_errors(self, tick_num: int) -> list[FactoryCommand]:
        commands = []
        for agent in self.agents:
            if agent.status == "awaiting_factory" and agent.active_error is not None:
                cmd = self._respond_to_error(agent, tick_num)
                if cmd is not None:
                    agent.handle_factory_response({"action": cmd.action, **cmd.payload})
                    commands.append(cmd)
        return commands

    def _respond_to_error(
        self, agent: CncAgent, tick_num: int
    ) -> Optional[FactoryCommand]:
        ctx = self._build_error_context(agent, tick_num)
        candidates = (
            self._call_openai(ctx) if self._use_llm
            else self._fallback_responses(ctx)
        )
        scored = [self._score_response(r) for r in candidates]

        # Safety gate (unified_plan.tex §7): if ANY candidate identifies
        # catastrophic risk, abort immediately — do not proceed with cutting.
        any_catastrophic = any(r.risk_level == "catastrophic" for r in scored)
        if any_catastrophic:
            action   = "abort"
            payload  = {}
            priority = "emergency"
        else:
            best     = max(scored, key=lambda r: r.policy_score)
            action   = best.action
            payload  = best.parameters
            priority = "urgent" if best.risk_level == "risky" else "normal"

        return FactoryCommand(
            command_id        = str(uuid.uuid4()),
            target_machine_id = agent.machine_id,
            action            = action,
            payload           = payload,
            priority          = priority,
            tick_issued       = tick_num,
        )

    def _build_error_context(self, agent: CncAgent, tick_num: int) -> ErrorContext:
        ev    = agent.active_error
        state = agent.get_state()
        life  = min(
            (t["remaining_life_pct"] for t in state["tool_crib"]),
            default=100.0,
        )
        return ErrorContext(
            machine_id         = agent.machine_id,
            error_description  = ev.description if ev else "",
            error_section      = ev.section      if ev else "",
            error_gcode_line   = ev.gcode_line   if ev else "",
            machine_status     = state["status"],
            tool_life_pct      = life,
            sections_completed = state["completed_sections"],
            queue_depth        = state["queue_depth"],
            factory_policy     = self.policy,
            tick               = tick_num,
        )

    # ── LLM ------------------------------------------------------------------

    def _build_llm_prompt(self, ctx: ErrorContext) -> str:
        return (
            "You are the factory supervisor LLM for an autonomous CNC machining factory.\n"
            f"Active policy: {ctx.factory_policy}\n\n"
            "Respond with a JSON object containing exactly:\n"
            '  "action"     : one of ["continue","reduce_feed","rework","abort"]\n'
            '  "parameters" : dict  (e.g. {"feed_override_pct": 80})\n'
            '  "risk_level" : one of ["safe","risky","catastrophic"]\n'
            '  "reasoning"  : one sentence\n\n'
            f"Machine  : {ctx.machine_id}\n"
            f"Error    : {ctx.error_description}\n"
            f"Section  : {ctx.error_section}\n"
            f"G-code   : {ctx.error_gcode_line}\n"
            f"Tool life: {ctx.tool_life_pct:.1f}%\n"
        )

    def _call_openai(self, ctx: ErrorContext) -> list[LLMResponse]:
        prompt = self._build_llm_prompt(ctx)
        responses: list[LLMResponse] = []
        for temp in list(config.FACTORY_LLM_TEMPS)[: config.FACTORY_N_LLM_CANDIDATES]:
            try:
                comp = self._llm_client.chat.completions.create(  # type: ignore
                    model       = config.OPENAI_MODEL,
                    max_tokens  = config.OPENAI_MAX_TOKENS,
                    temperature = temp,
                    messages    = [
                        {"role": "system",
                         "content": "CNC factory supervisor. Respond only with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = comp.choices[0].message.content or ""
                responses.append(self._parse_llm_text(raw))
            except Exception as exc:
                responses.append(LLMResponse(
                    response_id = str(uuid.uuid4()),
                    raw_text    = f"[API error: {exc}]",
                    action      = "continue",
                    risk_level  = "safe",
                ))
        return responses or self._fallback_responses(ctx)

    def _parse_llm_text(self, raw: str) -> LLMResponse:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            obj = json.loads(clean)
        except json.JSONDecodeError:
            obj = {}
        return LLMResponse(
            response_id = str(uuid.uuid4()),
            raw_text    = raw,
            action      = str(obj.get("action",    "continue")),
            parameters  = dict(obj.get("parameters", {})),
            risk_level  = str(obj.get("risk_level",  "safe")),
        )

    # ── Fallback rules --------------------------------------------------------

    _FALLBACK_RULES: list = [
        (("chatter", "vibration", "resonance"),
         "reduce_feed", "risky",       {"feed_override_pct": 70}),
        (("collision", "crash", "overload"),
         "abort",       "catastrophic", {}),
        (("worn", "broken", "missing tool"),
         "rework",      "risky",        {}),
        (("finish", "surface", "roughness"),
         "reduce_feed", "safe",         {"feed_override_pct": 80}),
        (("coolant", "temperature", "heat"),
         "reduce_feed", "safe",         {"feed_override_pct": 85}),
    ]

    def _fallback_responses(self, ctx: ErrorContext) -> list[LLMResponse]:
        desc   = ctx.error_description.lower()
        p_act  = "continue"
        p_risk = "safe"
        p_parm: dict = {}
        for keywords, action, risk, params in self._FALLBACK_RULES:
            if any(kw in desc for kw in keywords):
                p_act, p_risk, p_parm = action, risk, params
                break
        variants = [
            (p_act,         p_risk,  p_parm),
            ("reduce_feed", "safe",  {"feed_override_pct": 75}),
            ("continue",    "safe",  {}),
        ]
        return [
            LLMResponse(
                response_id = str(uuid.uuid4()),
                raw_text    = f"[fallback] action={act}",
                action      = act,
                parameters  = prm,
                risk_level  = rsk,
            )
            for act, rsk, prm in variants[: config.FACTORY_N_LLM_CANDIDATES]
        ]

    def _score_response(self, resp: LLMResponse) -> LLMResponse:
        table   = _POLICY_ACTION_SCORES.get(self.policy,
                                            _POLICY_ACTION_SCORES["min_time"])
        base    = table.get(resp.action, 0.0)
        penalty = _RISK_PENALTY.get(resp.risk_level, 0.0)
        resp.policy_score = max(0.0, base + penalty)
        return resp

    # ── Tool crib management --------------------------------------------------

    def _seed_inventory(self) -> None:
        raw = _MACHINE_TOOL_DATA.get("M01", [])
        for spec in raw:
            key = spec["tool_id"]
            self.tool_inventory[key] = [
                ToolRecord(
                    tool_id             = spec["tool_id"],
                    n_inserts           = spec["n_inserts"],
                    diameter_mm         = spec["diameter_mm"],
                    shank_length_mm     = spec["shank_length_mm"],
                    max_life_hrs        = spec["max_life_hrs"],
                    remaining_life_hrs  = spec["max_life_hrs"],
                    holder_cost_usd     = spec["holder_cost_usd"],
                    consumable_cost_usd = spec["consumable_cost_usd"],
                    requires_coolant    = True,
                )
                for _ in range(config.FACTORY_TOOL_STOCK_PER_SPEC)
            ]

    def _manage_tool_cribs(self, tick_num: int) -> list[FactoryCommand]:
        """
        WARN_PCT: schedule replacement at next part-load boundary.
          - Queued in _tools_pending_change; the GUI _tick_machine checks
            this set when transitioning idle → setup and inserts a
            tool_change phase first.
        STOP_PCT: block new job start immediately.
          - Handled the same way; the idle→setup guard sees needs_replacement
            and refuses to start until the tool_change countdown finishes.
        In both cases the actual swap + downtime countdown is driven by
        the GUI state machine, not here.
        """
        commands = []
        for agent in self.agents:
            # STOP_PCT tools: flag immediately via FactoryCommand
            for worn in agent.tool_crib.tools_at_stop():
                cmd = self._send_tool_replacement(agent, worn, tick_num,
                                                  urgency="stop")
                if cmd is not None:
                    commands.append(cmd)
            # WARN_PCT tools: schedule for next changeover
            for worn in agent.tool_crib.tools_at_warn():
                cmd = self._send_tool_replacement(agent, worn, tick_num,
                                                  urgency="warn")
                if cmd is not None:
                    commands.append(cmd)
        return commands

    def _send_tool_replacement(
        self,
        agent:      CncAgent,
        worn:       ToolRecord,
        tick_num:   int,
        urgency:    str = "warn",   # "warn" | "stop"
    ) -> Optional[FactoryCommand]:
        """
        Install a replacement tool in the agent's crib.

        1. Try exact tool_id match from inventory.
        2. If stock empty, find nearest-diameter tool across ALL inventory
           entries and use that instead.  Log the substitution.
        3. If truly nothing available, log a critical warning and return None.

        The FactoryCommand payload includes:
          tool_id        : id of the worn tool being replaced
          new_tool_id    : id of the fresh tool installed
          diameter_mm    : diameter of the fresh tool
          substitute     : True when a different-diameter tool was used
          urgency        : "warn" | "stop"
          needs_replanning : True when substitute=True (caller must re-CAM)
        """
        stock = self.tool_inventory.get(worn.tool_id, [])
        substitute = False

        if stock:
            fresh = copy.copy(stock.pop())
        else:
            # No exact match — search all inventory for nearest diameter
            best: Optional[ToolRecord] = None
            best_delta = float("inf")
            for tid, tlist in self.tool_inventory.items():
                if not tlist:
                    continue
                candidate = tlist[-1]   # peek without popping
                delta = abs(candidate.diameter_mm - worn.diameter_mm)
                if delta < best_delta:
                    best_delta = delta
                    best = candidate
                    best_tid = tid
            if best is None:
                # No tools at all in inventory
                return FactoryCommand(
                    command_id        = str(uuid.uuid4()),
                    target_machine_id = agent.machine_id,
                    action            = "tool_out_of_stock",
                    payload           = {"tool_id": worn.tool_id,
                                         "urgency": urgency},
                    priority          = "critical",
                    tick_issued       = tick_num,
                )
            fresh      = copy.copy(self.tool_inventory[best_tid].pop())
            substitute = True

        agent.tool_crib.add_tool(fresh)
        needs_replanning = substitute   # different diameter → re-CAM needed

        return FactoryCommand(
            command_id        = str(uuid.uuid4()),
            target_machine_id = agent.machine_id,
            action            = "replace_tool",
            payload           = {
                "tool_id":         worn.tool_id,
                "new_tool_id":     fresh.tool_id,
                "diameter_mm":     fresh.diameter_mm,
                "substitute":      substitute,
                "urgency":         urgency,
                "needs_replanning": needs_replanning,
            },
            priority          = "critical" if urgency == "stop" else "normal",
            tick_issued       = tick_num,
        )

    def add_to_inventory(self, tool: ToolRecord, quantity: int = 1) -> None:
        key = tool.tool_id
        if key not in self.tool_inventory:
            self.tool_inventory[key] = []
        for _ in range(quantity):
            self.tool_inventory[key].append(copy.copy(tool))

    def inventory_count(self, tool_id: str) -> int:
        return len(self.tool_inventory.get(tool_id, []))

    # ── New job creation ------------------------------------------------------

    def build_resume_job(
        self,
        original_job,
        completed_face_ids: set,
        machine_id: Optional[str] = None,
        progress_cb=None,
    ) -> "Optional[Job]":
        """
        Re-generate G-code for the unmachined faces of an in-progress job
        after a tool substitution (different diameter).  The original job's
        ToolpathResult is reused; only faces NOT in completed_face_ids are
        replanned with the new tool from the machine's crib.

        Parameters
        ----------
        original_job       : the Job currently/recently on the machine
        completed_face_ids : set of face_id ints already fully machined
        machine_id         : machine whose crib supplies the replacement tool

        Returns
        -------
        New Job covering only the remaining faces, or None on error.
        """
        def _cb(phase, detail=""):
            if progress_cb:
                try: progress_cb(original_job.seed, phase, detail)
                except Exception: pass

        try:
            from feeds_speeds_engine import compute as fs_compute
            from toolpath_planner    import plan, ToolpathResult
            from gcode_generator     import generate
            from scheduler           import job_from_gcode
            from timing_model        import assign_material, stamp_toolpath

            if original_job.toolpath_result is None:
                _cb("error", "No toolpath_result on original job — cannot resume")
                return None

            # Select replacement tool from crib
            tool_diameter_mm = config.DEFAULT_TOOL_DIAMETER_MM
            tool_flutes      = config.DEFAULT_TOOL_FLUTES
            tool_id_used     = ""
            if machine_id is not None:
                _agent = next((a for a in self.agents
                               if a.machine_id == machine_id), None)
                if _agent is not None:
                    _tool_rec = _agent.tool_crib.select_for_face(
                        max((min(max(v[0] for v in tf.vertices_2d)
                                - min(v[0] for v in tf.vertices_2d),
                                max(v[1] for v in tf.vertices_2d)
                                - min(v[1] for v in tf.vertices_2d))
                            for tf in original_job.toolpath_result
                                    .__class__.__mro__  # dummy — replaced below
                            ), default=0.0)
                    )
            # Simpler: pick best available tool for any face width
            if machine_id is not None:
                _agent2 = next((a for a in self.agents
                                if a.machine_id == machine_id), None)
                if _agent2:
                    tpr = original_job.toolpath_result
                    face_widths = []
                    for face_tp in tpr.faces:
                        if face_tp.face_id in completed_face_ids:
                            continue
                        # face_tp has no vertices_2d; use safe_region bounds
                        if face_tp.safe_region is not None:
                            b = face_tp.safe_region.bounds  # (minx,miny,maxx,maxy)
                            face_widths.append(min(b[2]-b[0], b[3]-b[1]))
                    rep_width  = max(face_widths) if face_widths else 0.0
                    _tool_rec2 = _agent2.tool_crib.select_for_face(rep_width)
                    if _tool_rec2:
                        tool_diameter_mm = _tool_rec2.diameter_mm
                        tool_flutes      = _tool_rec2.n_inserts
                        tool_id_used     = _tool_rec2.tool_id
                        _cb("fs",
                            f"Resume: crib {machine_id} selected "
                            f"{_tool_rec2.tool_id} Ø{_tool_rec2.diameter_mm:.0f} mm")

            material = original_job.material or assign_material()
            fs = fs_compute(
                material, tool_diameter_mm, tool_flutes,
                tool_type       = config.DEFAULT_TOOL_TYPE,
                axial_depth_mm  = 0.50 * tool_diameter_mm,
                radial_depth_mm = 0.40 * tool_diameter_mm,
            )

            # Build a ToolpathResult containing only remaining faces
            tpr_orig  = original_job.toolpath_result
            remaining = [f for f in tpr_orig.faces
                         if f.face_id not in completed_face_ids]
            if not remaining:
                _cb("fs", "All faces already done — no resume needed")
                return None

            tpr_resume = ToolpathResult(
                seed                   = tpr_orig.seed,
                faces                  = remaining,
                total_path_length_mm   = sum(f.path_length_mm for f in remaining),
                total_estimated_time_s = sum(f.estimated_time_s for f in remaining),
                warnings               = ["RESUME: unmachined faces only"],
            )
            t_mach_s = stamp_toolpath(tpr_resume, fs, t0=0.0)
            gc       = generate(tpr_resume, fs)
            gc.material = material

            job = job_from_gcode(gc)
            job.estimated_time_s  = max(t_mach_s, 1.0)
            job.material          = material
            job.tool_id_used      = tool_id_used
            job.tool_diameter_used = tool_diameter_mm
            job.toolpath_result   = tpr_resume
            job.status            = "cam_ready"
            _cb("gcode",
                f"Resume G-code: {len(gc.lines)} lines  "
                f"{len(remaining)} face(s) remaining  "
                f"Ø{tool_diameter_mm:.0f} mm tool")
            return job

        except Exception as exc:
            _cb("error", f"build_resume_job failed: {exc}")
            return None

    def build_job_from_seed(
        self,
        seed: int,
        machine_id: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
        generate_if_missing: bool = False,
    ) -> Optional[Job]:
        """
        Run the full Phase 1–4 CAM pipeline for *seed* and return a
        **ready-to-queue** Job — WITHOUT submitting it to the scheduler.

        The pipeline progresses through all CAM stages:
          seed → solid → features → feeds/speeds → toolpath → G-code

        After this call the Job has valid gcode_lines, estimated_time_s,
        material, and every other field required by the scheduler.  The
        caller decides when to submit it (by calling enqueue_job).

        Parameters
        ----------
        seed                : integer seed number
        progress_cb         : optional callable(seed, phase, detail)
                              called at each pipeline stage — safe to post
                              events to a PipelineMonitorWindow from this thread
        generate_if_missing : if True and the .npy doesn't exist, call
                              cnc_solid_bridge.generate_solid() first
                              (requires pyocc conda env)

        Returns
        -------
        Job with gcode_lines populated, or None on any error.
        """

        def _cb(phase: str, detail: str = "") -> None:
            if progress_cb is not None:
                try:
                    progress_cb(seed, phase, detail)
                except Exception:
                    pass

        try:
            import cnc_solid_bridge as bridge
            from feature_extractor   import extract_features
            from feeds_speeds_engine import compute as fs_compute
            from toolpath_planner    import plan
            from gcode_generator     import generate
            from scheduler           import job_from_gcode

            # ── Stage: seed ───────────────────────────────────────────────
            _cb("seed", f"Seed {seed} initiated")

            # ── Stage: solid ──────────────────────────────────────────────
            npy = os.path.join(config.SOLID_OUTPUT_DIR,
                               f"solid_faces_seed_{seed}.npy")
            if not os.path.exists(npy):
                if generate_if_missing:
                    _cb("solid", f"Calling Build_Solid.py for seed {seed}…")
                    bridge.generate_solid(seed)
                else:
                    _cb("error", f"No .npy for seed {seed} — skipping")
                    return None

            _cb("solid", "Loading faces from disk…")
            faces = bridge.load_face_polygons(seed)
            meta  = bridge.load_metadata(seed)
            vol   = meta.get("volume") or 0.0
            _cb("solid",
                f"{len(faces)} faces loaded  |  volume = {vol:.0f} mm³")

            # ── Stage: features ───────────────────────────────────────────
            _cb("features", "Extracting edge labels & safe regions…")
            feat  = extract_features(faces, seed=seed, volume=vol)
            _cb("features",
                f"{feat.n_top_faces} top faces  |  "
                f"{getattr(feat, 'n_edges', '?')} edges labelled")

            # ── Stage: feeds & speeds ────────────────────────────────────────────
            from timing_model import assign_material, stamp_toolpath
            material = assign_material()

            # Select the best tool from this machine's crib.
            # Use the narrowest bounding-box side of the widest top face as
            # the representative width; ToolCrib.select_for_face() picks the
            # largest tool whose diameter < face_width / 4.
            tool_diameter_mm = config.DEFAULT_TOOL_DIAMETER_MM
            tool_flutes      = config.DEFAULT_TOOL_FLUTES
            tool_type_sel    = config.DEFAULT_TOOL_TYPE
            tool_id_used     = ""

            if machine_id is not None:
                _agent = next(
                    (a for a in self.agents if a.machine_id == machine_id),
                    None,
                )
                if _agent is not None:
                    face_widths = []
                    for tf in feat.top_faces:
                        xs = [v[0] for v in tf.vertices_2d]
                        ys = [v[1] for v in tf.vertices_2d]
                        if xs and ys:
                            face_widths.append(
                                min(max(xs) - min(xs), max(ys) - min(ys))
                            )
                    rep_width = max(face_widths) if face_widths else 0.0
                    _tool_rec = _agent.tool_crib.select_for_face(rep_width)
                    if _tool_rec is not None:
                        tool_diameter_mm = _tool_rec.diameter_mm
                        tool_flutes      = _tool_rec.n_inserts
                        tool_id_used     = _tool_rec.tool_id
                        _cb("fs",
                            f"Crib {machine_id}: selected "
                            f"{_tool_rec.tool_id} Ø{_tool_rec.diameter_mm:.0f} mm  "
                            f"({_tool_rec.n_inserts} inserts  "
                            f"life={_tool_rec.remaining_life_pct:.0f}%)")
                    else:
                        _cb("fs",
                            f"No usable tool in {machine_id} crib — "
                            f"fallback Ø{tool_diameter_mm:.0f} mm")

            _cb("fs", f"Computing feeds/speeds  ({material})…")
            fs = fs_compute(
                material,
                tool_diameter_mm,
                tool_flutes,
                tool_type       = tool_type_sel,
                axial_depth_mm  = 0.50 * tool_diameter_mm,
                radial_depth_mm = 0.40 * tool_diameter_mm,
            )
            _cb("fs",
                f"RPM = {fs.rpm:.0f}  |  "
                f"feed = {fs.feed_rate_mmpm:.0f} mm/min  |  "
                f"power = {fs.power_kw:.2f} kW  |  "
                f"mat = {material}  |  tool Ø{tool_diameter_mm:.0f} mm")

            # ── Stage: toolpath ───────────────────────────────────────────
            _cb("toolpath", "Planning raster toolpath…")
            tp = plan(feat, fs)
            _cb("toolpath",
                f"{len(tp.faces)} face(s)  |  "
                f"path = {tp.total_path_length_mm:.1f} mm")

            # ── Stage: timing stamp ───────────────────────────────────────
            # Attach t_start/t_end to every Waypoint so the GUI tick-executor
            # knows exactly when each move is due.
            t_mach_s = stamp_toolpath(tp, fs, t0=0.0)
            _cb("toolpath",
                f"Timing stamped: {t_mach_s:.0f} s = {t_mach_s/60:.1f} min "
                f"({material})")

            # ── Stage: G-code ─────────────────────────────────────────────
            _cb("gcode", "Generating and validating G-code…")
            gc = generate(tp, fs)
            gc.material = material          # stored for completed-parts log
            _cb("gcode",
                f"{len(gc.lines)} lines  |  "
                f"est. time = {t_mach_s:.0f} s  ({material})  "
                f"— G-code ready, awaiting queue submission")

            # ── Build Job (NOT yet queued) ─────────────────────────────────
            job = job_from_gcode(gc)
            # Use the stamped machining time (material-aware).
            job.estimated_time_s  = max(t_mach_s, 1.0)
            job.material          = material
            job.tool_id_used      = tool_id_used
            job.tool_diameter_used = tool_diameter_mm
            # Attach the stamped ToolpathResult so the GUI tick-executor can
            # use waypoints_due() to advance the G-code cursor in real sim-time
            # instead of advancing by one line per tick.
            job.toolpath_result   = tp
            # job.status remains "queued" only after enqueue_job() is called;
            # set a sentinel so callers can distinguish CAM-ready from queued.
            job.status = "cam_ready"

            return job

        except Exception as exc:
            _cb("error", str(exc))
            return None

    def enqueue_job(
        self,
        job: Job,
        seed: int,
        progress_cb: Optional[Callable] = None,
    ) -> Job:
        """
        Submit a CAM-ready Job to the scheduler queue.

        This is the second half of the two-phase pipeline:
          Phase 1  →  build_job_from_seed()   (CAM: solid → G-code)
          Phase 2  →  enqueue_job()            (scheduler submission)

        The split means the scheduler queue only receives parts that
        already have fully validated G-code — not parts still being
        processed by the CAM pipeline.

        Parameters
        ----------
        job         : Job returned by build_job_from_seed() (status="cam_ready")
        seed        : original seed number (used only for progress reporting)
        progress_cb : same callback as build_job_from_seed()

        Returns
        -------
        The same Job, now with status="queued" and assigned to the scheduler.
        """

        def _cb(phase: str, detail: str = "") -> None:
            if progress_cb is not None:
                try:
                    progress_cb(seed, phase, detail)
                except Exception:
                    pass

        # ── Stage: scheduler ─────────────────────────────────────────────
        _cb("schedule", "Submitting G-code job to scheduler queue…")
        self.scheduler.submit(job)    # sets job.status = "queued"
        _cb("schedule",
            f"Job {job.job_id[:8]}…  queued  "
            f"(depth = {len(self.scheduler.job_queue)})")

        # ── Stage: machine (assigned on next tick) ────────────────────────
        # The scheduler assigns on the next tick_once(); we report the
        # queue position now and the GUI will see the assignment shortly.
        _cb("machine",
            "Queued — will assign to idle machine on next tick")

        return job

    def submit_new_seed(
        self,
        seed: int,
        progress_cb: Optional[Callable] = None,
        generate_if_missing: bool = False,
    ) -> Optional[Job]:
        """
        Convenience wrapper: run the full Phase 1–4 CAM pipeline for *seed*
        **and** immediately submit the resulting job to the scheduler.

        Internally this calls build_job_from_seed() followed by enqueue_job(),
        keeping the two-phase contract intact.  Prefer calling those methods
        directly when you need to separate G-code generation from queuing.

        Parameters
        ----------
        seed                : integer seed number
        progress_cb         : optional callable(seed, phase, detail)
        generate_if_missing : call Build_Solid.py if .npy is missing

        Returns
        -------
        Job if successful, None on any error.
        """
        job = self.build_job_from_seed(
            seed,
            progress_cb         = progress_cb,
            generate_if_missing = generate_if_missing,
        )
        if job is None:
            return None
        return self.enqueue_job(job, seed, progress_cb=progress_cb)

    # ── State / KPIs ---------------------------------------------------------

    def get_factory_state(self) -> dict:
        return {
            "tick":               self.scheduler.tick,
            "t_real_s":           self.scheduler.t_real_s,
            "policy":             self.policy,
            "use_llm":            self._use_llm,
            "scheduler":          self.scheduler.state_dict(),
            "agents":             [ag.get_state() for ag in self.agents],
            "rework_queue_depth": len(self.rework_queue),
            "tool_inventory":     {tid: len(st)
                                   for tid, st in self.tool_inventory.items()},
            "kpis":               self.get_production_kpis(),
        }

    def get_production_kpis(self) -> dict:
        total_life = sum(t.remaining_life_pct
                         for ag in self.agents
                         for t in ag.tool_crib.tools)
        n_tools = sum(len(ag.tool_crib.tools) for ag in self.agents)
        avg_life = total_life / n_tools if n_tools else 0.0

        return {
            "machines_running":          sum(1 for ag in self.agents
                                             if ag.status == "running"),
            "machines_idle":             sum(1 for ag in self.agents
                                             if ag.status == "idle"),
            "machines_awaiting_factory": sum(1 for ag in self.agents
                                             if ag.status == "awaiting_factory"),
            "machines_stopped":          sum(1 for ag in self.agents
                                             if ag.status in
                                             ("stopped","awaiting_tool_change")),
            "total_jobs_completed":      self.total_jobs_completed,
            "total_errors_processed":    self.total_errors_processed,
            "total_tools_replaced":      self.total_tools_replaced,
            "rework_queue_depth":        len(self.rework_queue),
            "avg_tool_life_pct":         round(avg_life, 1),
            "scheduler_queue_depth":     len(self.scheduler.job_queue),
        }

    def summary(self) -> str:
        kpis = self.get_production_kpis()
        lines = [
            f"Factory — tick {self.scheduler.tick}  "
            f"(t_real = {self.scheduler.t_real_s:.0f} s)  "
            f"policy = {self.policy}  "
            f"LLM = {'openai' if self._use_llm else 'fallback'}",
            f"  Machines : {kpis['machines_running']} running  "
            f"{kpis['machines_idle']} idle  "
            f"{kpis['machines_awaiting_factory']} awaiting  "
            f"{kpis['machines_stopped']} stopped",
            f"  Jobs     : {kpis['total_jobs_completed']} done  "
            f"{kpis['rework_queue_depth']} rework  "
            f"{kpis['scheduler_queue_depth']} queued",
            f"  Errors   : {kpis['total_errors_processed']} processed  "
            f"  Tools replaced: {kpis['total_tools_replaced']}",
            f"  Avg tool life : {kpis['avg_tool_life_pct']:.1f}%",
        ]
        for ag in self.agents:
            state = ag.get_state()
            jid   = state["current_job"] or "—"
            lines.append(
                f"  {ag.machine_id}  {ag.status:<22s}  "
                f"job={jid[:8] if jid != '—' else '—'}"
            )
        return "\n".join(lines)


# ── Builder -------------------------------------------------------------------

def build_factory_agent(
    n_machines:     int  = 4,
    policy:         str  = "min_time",
    openai_api_key: Optional[str] = None,
    dry_run:        bool = True,
) -> FactoryAgent:
    rework: deque = deque()
    agents: list[CncAgent] = []
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

    scheduler = Scheduler(
        n_machines = n_machines,
        policy     = policy,
        rng_seed   = 42,
    )

    return FactoryAgent(
        agents         = agents,
        scheduler      = scheduler,
        policy         = policy,
        openai_api_key = openai_api_key,
    )

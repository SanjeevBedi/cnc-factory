"""
factory_agent.py — Phase 7 of the CNC Factory pipeline.

Responsibilities (per Autonomous_Factory_Sim.docx + unified_plan.tex §7)
------------------------------------------------------------------------
1.  Orchestrate all CNC agents (Phase 6) and the Scheduler (Phase 5).
        - Advance the shared clock tick by tick.
        - Run every CncAgent's tick() and collect state snapshots.
        - Aggregate production KPIs across all machines.

2.  Error diagnosis via OpenAI LLM.
        - When any agent transitions to "awaiting_factory", the factory
          collects structured ErrorContext and submits it to the LLM.
        - Three candidate responses are generated (temperatures 0.2 / 0.5 / 0.8).
        - Each is scored against the active factory policy.
        - The highest-scoring candidate is dispatched back as a FactoryCommand.
        - When no OpenAI key is present the factory falls back to a deterministic
          rule-based analyser so every test runs without a live API key.

3.  Tool-crib management (factory side).
        - Factory maintains a tool inventory (FACTORY_TOOL_STOCK_PER_SPEC copies
          of each spec from M01 by default).
        - When a machine's crib contains a tool flagged needs_replacement, the
          factory sends a replace_tool command and ships a fresh unit.

4.  New-job creation.
        submit_new_seed(seed) runs the full Phase 1–4 pipeline (if the solid
        file exists on disk) and submits the resulting Job to the Scheduler.

5.  Provide a JSON-serialisable factory-state snapshot for dashboards and tests.

Architecture (from unified_plan.tex)
--------------------------------------
    simulate → prompt → transform → CNC LLM → safety check
        → (escalate) → factory LLM → mitigate → resume

Factory LLM produces structured commands:
    {diagnosis, actions: [{target, command, parameters}]}

Structured error context sent to LLM:
    {event_type, description, severity, machine_state, factory_policy,
     sections_completed, sections_remaining}

Sources
-------
Autonomous_Factory_Sim.docx   §error-handling, §tool-crib, §factory-LLM, §policies
unified_plan.tex               §7 Factory Agent System
config.py                      OPENAI_MODEL, FACTORY_N_LLM_CANDIDATES, etc.
cnc_agent.py                   CncAgent, ToolRecord, build_factory()
scheduler.py                   Scheduler, Job, make_job, job_from_gcode
"""

from __future__ import annotations

import copy
import json
import os
import sys
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import config
from cnc_agent import (
    CncAgent, ToolRecord, ToolCrib,
    build_default_crib, build_factory,
    _MACHINE_TOOL_DATA,
)
from scheduler import Scheduler, Job, make_job

# ── Optional OpenAI import ────────────────────────────────────────────────────
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None  # type: ignore[assignment,misc]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FactoryCommand:
    """Command the factory agent sends to a CNC agent."""
    command_id:        str
    target_machine_id: str
    action: str          # continue | reduce_feed | abort | rework | replace_tool | pause
    payload:           dict = field(default_factory=dict)
    priority:          str  = "normal"   # normal | urgent | emergency
    tick_issued:       int  = 0


@dataclass
class ErrorContext:
    """
    Structured context assembled from machine state + error event.
    This is what gets serialised into the LLM prompt.
    """
    machine_id:          str
    error_description:   str
    error_section:       str
    error_gcode_line:    str
    machine_status:      str
    tool_life_pct:       float
    sections_completed:  list
    queue_depth:         int
    factory_policy:      str
    tick:                int


@dataclass
class LLMResponse:
    """One candidate action from the LLM (or fallback)."""
    response_id:   str
    raw_text:      str
    action:        str          # continue | reduce_feed | abort | rework
    parameters:    dict = field(default_factory=dict)
    risk_level:    str  = "safe"   # safe | risky | catastrophic
    policy_score:  float = 0.0


@dataclass
class FactoryTickResult:
    """Summary of one factory-level tick."""
    tick:              int
    t_real_s:          float
    commands_sent:     list = field(default_factory=list)   # FactoryCommand
    errors_processed:  int  = 0
    tools_replaced:    int  = 0
    jobs_created:      int  = 0
    jobs_completed:    int  = 0
    jobs_in_rework:    int  = 0


# ── Policy scoring tables ─────────────────────────────────────────────────────
# Score how well each possible action aligns with each factory policy.
# Higher = better fit.  Used to pick among LLM candidates.

_POLICY_ACTION_SCORES: dict[str, dict[str, float]] = {
    "min_time": {
        "continue":     1.0,
        "reduce_feed":  0.5,
        "rework":       0.2,
        "abort":        0.0,
        "replace_tool": 0.3,
    },
    "min_cost": {
        "continue":     0.8,
        "reduce_feed":  0.7,
        "rework":       0.2,
        "abort":        0.0,
        "replace_tool": 0.4,
    },
    "best_finish": {
        "continue":     0.5,
        "reduce_feed":  1.0,
        "rework":       0.3,
        "abort":        0.0,
        "replace_tool": 0.6,
    },
    "max_tool_life": {
        "continue":     0.4,
        "reduce_feed":  1.0,
        "rework":       0.3,
        "abort":        0.0,
        "replace_tool": 0.9,
    },
}

# Risk penalties applied to the base policy score
_RISK_PENALTY: dict[str, float] = {
    "safe":        0.0,
    "risky":      -0.1,
    "catastrophic":-0.5,
}


# ── Factory Agent ─────────────────────────────────────────────────────────────

class FactoryAgent:
    """
    Top-level orchestrator for the autonomous CNC factory.

    Parameters
    ----------
    agents           : list of CncAgent objects (one per machine)
    scheduler        : shared Scheduler instance
    policy           : factory-wide scheduling + error-response policy
    openai_api_key   : if provided AND openai package is installed, the factory
                       LLM uses GPT; otherwise the built-in rule-based fallback
                       is used so tests pass without a key.
    """

    def __init__(
        self,
        agents:          list[CncAgent],
        scheduler:       Scheduler,
        policy:          str = "min_time",
        openai_api_key:  Optional[str] = None,
    ) -> None:
        self.agents    = agents
        self.scheduler = scheduler
        self.policy    = policy
        self.rework_queue: deque = agents[0].rework_queue if agents else deque()

        # OpenAI client (optional)
        self._use_llm = (
            openai_api_key is not None
            and bool(openai_api_key.strip())
            and _OPENAI_AVAILABLE
        )
        self._llm_client = (
            _OpenAI(api_key=openai_api_key)   # type: ignore[call-arg]
            if self._use_llm else None
        )

        # Tool inventory: factory stock of fresh tools (indexed by tool_id)
        self.tool_inventory: dict[str, list[ToolRecord]] = {}
        self._seed_inventory()

        # Command log
        self.command_log: list[FactoryCommand] = []

        # Statistics
        self.total_errors_processed = 0
        self.total_tools_replaced   = 0
        self.total_jobs_completed   = 0

    # ── Tick ─────────────────────────────────────────────────────────────────

    def tick(self) -> FactoryTickResult:
        """
        Advance factory by one time step.

            1. Advance Scheduler (assigns jobs, progresses machines).
            2. Run each CncAgent tick.
            3. Process any errors (call LLM / fallback → FactoryCommand).
            4. Audit tool cribs → dispatch replacements.
            5. Collect KPIs.
        """
        sched_result = self.scheduler.tick_once()
        tick_num     = self.scheduler.tick
        t_real_s     = self.scheduler.t_real_s

        ftr = FactoryTickResult(
            tick      = tick_num,
            t_real_s  = t_real_s,
            jobs_completed = len(sched_result.jobs_finished),
        )
        self.total_jobs_completed += len(sched_result.jobs_finished)

        # Run agent ticks
        for agent in self.agents:
            agent.tick(tick_num)

        # Process errors
        commands = self._process_errors(tick_num)
        ftr.commands_sent.extend(commands)
        ftr.errors_processed = len(commands)
        self.total_errors_processed += len(commands)

        # Manage tool cribs
        replace_cmds = self._manage_tool_cribs(tick_num)
        ftr.commands_sent.extend(replace_cmds)
        ftr.tools_replaced = len(replace_cmds)
        self.total_tools_replaced += len(replace_cmds)

        ftr.jobs_in_rework = len(self.rework_queue)
        self.command_log.extend(ftr.commands_sent)
        return ftr

    def run(self, n_ticks: int) -> list[FactoryTickResult]:
        """Run for n_ticks steps."""
        return [self.tick() for _ in range(n_ticks)]

    # ── Error processing ─────────────────────────────────────────────────────

    def _process_errors(self, tick_num: int) -> list[FactoryCommand]:
        """Check all agents for active errors and dispatch responses."""
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
        """Build context, call LLM (or fallback), return best FactoryCommand."""
        ctx = self._build_error_context(agent, tick_num)

        candidates = (
            self._call_openai(ctx)
            if self._use_llm
            else self._fallback_responses(ctx)
        )

        scored = [self._score_response(r) for r in candidates]
        best   = max(scored, key=lambda r: r.policy_score)

        # Safety gate: catastrophic risk → override with abort
        if best.risk_level == "catastrophic":
            best.action = "abort"

        cmd = FactoryCommand(
            command_id        = str(uuid.uuid4()),
            target_machine_id = agent.machine_id,
            action            = best.action,
            payload           = best.parameters,
            priority          = "urgent" if best.risk_level != "safe" else "normal",
            tick_issued       = tick_num,
        )
        return cmd

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

    # ── LLM calls ────────────────────────────────────────────────────────────

    def _build_llm_prompt(self, ctx: ErrorContext) -> str:
        return (
            "You are the factory supervisor LLM for an autonomous CNC machining factory.\n"
            f"Active policy: {ctx.factory_policy}\n\n"
            "A CNC machine has reported an error.  Analyse the context below and\n"
            "respond with a JSON object containing exactly these fields:\n"
            '  "action"     : one of ["continue","reduce_feed","rework","abort"]\n'
            '  "parameters" : dict (e.g. {"feed_override_pct": 80})\n'
            '  "risk_level" : one of ["safe","risky","catastrophic"]\n'
            '  "reasoning"  : one sentence explaining the decision\n\n'
            f"Machine      : {ctx.machine_id}\n"
            f"Error        : {ctx.error_description}\n"
            f"Section      : {ctx.error_section}\n"
            f"G-code line  : {ctx.error_gcode_line}\n"
            f"Tool life    : {ctx.tool_life_pct:.1f}%\n"
            f"Sections done: {ctx.sections_completed}\n"
            f"Queue depth  : {ctx.queue_depth}\n"
        )

    def _call_openai(self, ctx: ErrorContext) -> list[LLMResponse]:
        """Generate FACTORY_N_LLM_CANDIDATES responses via OpenAI API."""
        prompt = self._build_llm_prompt(ctx)
        responses: list[LLMResponse] = []
        for temp in config.FACTORY_LLM_TEMPS[: config.FACTORY_N_LLM_CANDIDATES]:
            try:
                completion = self._llm_client.chat.completions.create(  # type: ignore[union-attr]
                    model       = config.OPENAI_MODEL,
                    max_tokens  = config.OPENAI_MAX_TOKENS,
                    temperature = temp,
                    messages    = [
                        {"role": "system",
                         "content": "You are a CNC factory supervisor LLM. "
                                    "Respond only with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = completion.choices[0].message.content or ""
                responses.append(self._parse_llm_text(raw))
            except Exception as exc:
                # API error → fallback entry
                responses.append(LLMResponse(
                    response_id = str(uuid.uuid4()),
                    raw_text    = f"[API error: {exc}]",
                    action      = "continue",
                    risk_level  = "safe",
                ))
        return responses or self._fallback_responses(ctx)

    def _parse_llm_text(self, raw: str) -> LLMResponse:
        """Extract structured fields from LLM output (best-effort JSON parse)."""
        # Strip markdown code fences if present
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            obj = json.loads(clean)
        except json.JSONDecodeError:
            obj = {}
        return LLMResponse(
            response_id = str(uuid.uuid4()),
            raw_text    = raw,
            action      = str(obj.get("action",     "continue")),
            parameters  = dict(obj.get("parameters", {})),
            risk_level  = str(obj.get("risk_level",  "safe")),
        )

    # ── Rule-based fallback ───────────────────────────────────────────────────

    _FALLBACK_RULES: list[tuple[tuple[str, ...], str, str, dict]] = [
        # (keywords, action, risk, parameters)
        (("chatter", "vibration", "resonance"),
         "reduce_feed", "risky", {"feed_override_pct": 70}),
        (("collision", "crash", "overload"),
         "abort", "catastrophic", {}),
        (("worn", "broken", "missing tool"),
         "rework", "risky", {}),
        (("finish", "surface", "roughness"),
         "reduce_feed", "safe", {"feed_override_pct": 80}),
        (("coolant", "temperature", "heat"),
         "reduce_feed", "safe", {"feed_override_pct": 85}),
    ]

    def _fallback_responses(self, ctx: ErrorContext) -> list[LLMResponse]:
        """
        Deterministic rule-based responses used when OpenAI is unavailable.
        Generates FACTORY_N_LLM_CANDIDATES variants (slight action diversity).
        """
        desc = ctx.error_description.lower()
        primary_action = "continue"
        primary_risk   = "safe"
        primary_params: dict = {}

        for keywords, action, risk, params in self._FALLBACK_RULES:
            if any(kw in desc for kw in keywords):
                primary_action = action
                primary_risk   = risk
                primary_params = params
                break

        # Three variants: primary, conservative, fastest
        variants = [
            (primary_action, primary_risk,  primary_params),
            ("reduce_feed",  "safe",         {"feed_override_pct": 75}),
            ("continue",     "safe",         {}),
        ]
        responses = []
        for action, risk, params in variants[: config.FACTORY_N_LLM_CANDIDATES]:
            responses.append(LLMResponse(
                response_id = str(uuid.uuid4()),
                raw_text    = f"[fallback] action={action}",
                action      = action,
                parameters  = params,
                risk_level  = risk,
            ))
        return responses

    # ── Response scoring ──────────────────────────────────────────────────────

    def _score_response(self, resp: LLMResponse) -> LLMResponse:
        """Add policy_score to an LLMResponse in place and return it."""
        table  = _POLICY_ACTION_SCORES.get(self.policy, _POLICY_ACTION_SCORES["min_time"])
        base   = table.get(resp.action, 0.0)
        penalty= _RISK_PENALTY.get(resp.risk_level, 0.0)
        resp.policy_score = max(0.0, base + penalty)
        return resp

    # ── Tool crib management ──────────────────────────────────────────────────

    def _seed_inventory(self) -> None:
        """
        Populate the factory tool inventory from M01's spec.
        FACTORY_TOOL_STOCK_PER_SPEC fresh copies of each tool type.
        """
        raw = _MACHINE_TOOL_DATA.get("M01", [])
        for spec in raw:
            key = spec["tool_id"]
            self.tool_inventory[key] = []
            for _ in range(config.FACTORY_TOOL_STOCK_PER_SPEC):
                self.tool_inventory[key].append(ToolRecord(
                    tool_id             = spec["tool_id"],
                    n_inserts           = spec["n_inserts"],
                    diameter_mm         = spec["diameter_mm"],
                    shank_length_mm     = spec["shank_length_mm"],
                    max_life_hrs        = spec["max_life_hrs"],
                    remaining_life_hrs  = spec["max_life_hrs"],   # brand new
                    holder_cost_usd     = spec["holder_cost_usd"],
                    consumable_cost_usd = spec["consumable_cost_usd"],
                    requires_coolant    = True,
                ))

    def _manage_tool_cribs(self, tick_num: int) -> list[FactoryCommand]:
        """
        For each agent, find tools needing replacement.
        If factory inventory has stock, dispatch a replace_tool command.
        """
        commands = []
        for agent in self.agents:
            for worn in agent.tool_crib.tools_needing_replacement():
                cmd = self._send_tool_replacement(agent, worn, tick_num)
                if cmd is not None:
                    commands.append(cmd)
        return commands

    def _send_tool_replacement(
        self, agent: CncAgent, worn: ToolRecord, tick_num: int
    ) -> Optional[FactoryCommand]:
        """
        Dispatch a fresh tool from factory inventory to the agent.
        Returns a FactoryCommand (and mutates the agent's crib + inventory).
        """
        stock = self.tool_inventory.get(worn.tool_id, [])
        if not stock:
            return None   # out of stock

        fresh = stock.pop()   # take one from inventory
        agent.tool_crib.add_tool(copy.copy(fresh))   # install in machine

        cmd = FactoryCommand(
            command_id        = str(uuid.uuid4()),
            target_machine_id = agent.machine_id,
            action            = "replace_tool",
            payload           = {
                "tool_id":     worn.tool_id,
                "diameter_mm": fresh.diameter_mm,
            },
            priority          = "normal",
            tick_issued       = tick_num,
        )
        return cmd

    def add_to_inventory(self, tool: ToolRecord, quantity: int = 1) -> None:
        """Factory receives a tool delivery (adds to stock)."""
        key = tool.tool_id
        if key not in self.tool_inventory:
            self.tool_inventory[key] = []
        for _ in range(quantity):
            self.tool_inventory[key].append(copy.copy(tool))

    def inventory_count(self, tool_id: str) -> int:
        return len(self.tool_inventory.get(tool_id, []))

    # ── New job creation ──────────────────────────────────────────────────────

    def submit_new_seed(self, seed: int) -> Optional[Job]:
        """
        Run the full Phase 1–4 pipeline for the given seed (if solid file
        exists on disk) and submit the resulting Job to the Scheduler.
        Returns the Job, or None if the solid file is missing.
        """
        npy = os.path.join(config.SOLID_OUTPUT_DIR, f"solid_faces_seed_{seed}.npy")
        if not os.path.exists(npy):
            return None

        try:
            import cnc_solid_bridge as bridge
            from feature_extractor    import extract_features
            from feeds_speeds_engine  import compute as fs_compute
            from toolpath_planner     import plan
            from gcode_generator      import generate
            from scheduler            import job_from_gcode

            faces = bridge.load_face_polygons(seed)
            meta  = bridge.load_metadata(seed)
            feat  = extract_features(faces, seed=seed, volume=meta["volume"])
            fs    = fs_compute(
                config.DEFAULT_MATERIAL, 12.0, 4,
                axial_depth_mm=6.0, radial_depth_mm=4.8,
            )
            tp    = plan(feat, fs)
            gc    = generate(tp, fs)
            job   = job_from_gcode(gc)
            self.scheduler.submit(job)
            return job
        except Exception:
            return None

    # ── State / KPIs ─────────────────────────────────────────────────────────

    def get_factory_state(self) -> dict:
        """JSON-serialisable snapshot of the entire factory."""
        return {
            "tick":               self.scheduler.tick,
            "t_real_s":           self.scheduler.t_real_s,
            "policy":             self.policy,
            "use_llm":            self._use_llm,
            "scheduler":          self.scheduler.state_dict(),
            "agents": [
                agent.get_state() for agent in self.agents
            ],
            "rework_queue_depth": len(self.rework_queue),
            "tool_inventory": {
                tid: len(stock)
                for tid, stock in self.tool_inventory.items()
            },
            "kpis": self.get_production_kpis(),
        }

    def get_production_kpis(self) -> dict:
        """High-level production metrics."""
        total_life = sum(
            t.remaining_life_pct
            for ag in self.agents
            for t in ag.tool_crib.tools
        )
        n_tools = sum(len(ag.tool_crib.tools) for ag in self.agents)
        avg_tool_life = total_life / n_tools if n_tools else 0.0

        running = sum(1 for ag in self.agents if ag.status == "running")
        idle    = sum(1 for ag in self.agents if ag.status == "idle")
        waiting = sum(1 for ag in self.agents if ag.status == "awaiting_factory")
        stopped = sum(1 for ag in self.agents
                      if ag.status in ("stopped", "awaiting_tool_change"))

        return {
            "machines_running":         running,
            "machines_idle":            idle,
            "machines_awaiting_factory":waiting,
            "machines_stopped":         stopped,
            "total_jobs_completed":     self.total_jobs_completed,
            "total_errors_processed":   self.total_errors_processed,
            "total_tools_replaced":     self.total_tools_replaced,
            "rework_queue_depth":       len(self.rework_queue),
            "avg_tool_life_pct":        round(avg_tool_life, 1),
            "scheduler_queue_depth":    len(self.scheduler.job_queue),
        }

    def summary(self) -> str:
        kpis = self.get_production_kpis()
        lines = [
            f"Factory — tick {self.scheduler.tick}  "
            f"(t_real = {self.scheduler.t_real_s:.0f} s)  "
            f"policy = {self.policy}  "
            f"LLM = {'openai' if self._use_llm else 'fallback'}",
            f"  Machines  : {kpis['machines_running']} running  "
            f"{kpis['machines_idle']} idle  "
            f"{kpis['machines_awaiting_factory']} awaiting  "
            f"{kpis['machines_stopped']} stopped",
            f"  Jobs      : {kpis['total_jobs_completed']} done  "
            f"{kpis['rework_queue_depth']} rework  "
            f"{kpis['scheduler_queue_depth']} queued",
            f"  Errors    : {kpis['total_errors_processed']} processed",
            f"  Tool life : avg {kpis['avg_tool_life_pct']:.1f}%  "
            f"  {kpis['total_tools_replaced']} replaced",
        ]
        for ag in self.agents:
            state = ag.get_state()
            jid   = state["current_job"] or "—"
            lines.append(
                f"  {ag.machine_id}  {ag.status:<22s}  "
                f"job={jid[:8] if jid != '—' else '—'}"
            )
        return "\n".join(lines)


# ── Convenience builder ───────────────────────────────────────────────────────

def build_factory_agent(
    n_machines:     int  = 4,
    policy:         str  = "min_time",
    openai_api_key: Optional[str] = None,
    dry_run:        bool = True,
) -> FactoryAgent:
    """
    Create a ready-to-run FactoryAgent with n_machines CncAgents and
    a matching Scheduler (same rework queue shared by all agents).
    """
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
        n_machines  = n_machines,
        policy      = policy,
        rng_seed    = 42,
    )

    return FactoryAgent(
        agents         = agents,
        scheduler      = scheduler,
        policy         = policy,
        openai_api_key = openai_api_key,
    )

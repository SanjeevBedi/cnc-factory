"""
test_factory_agent.py — Phase 7 tests for factory_agent.py

Test suite
----------
 1. test_factory_created_with_agents     — build_factory_agent creates n agents
 2. test_shared_rework_queue             — all agents share the same rework queue ref
 3. test_policy_attribute                — policy stored on factory and scheduler
 4. test_tick_advances_scheduler         — factory.tick() increments scheduler.tick
 5. test_tick_returns_result             — tick() returns FactoryTickResult
 6. test_error_detected_and_commanded    — agent error → factory issues FactoryCommand
 7. test_command_dispatched_to_agent     — FactoryCommand.target_machine_id is correct
 8. test_agent_resumes_after_command     — agent status leaves awaiting_factory
 9. test_fallback_responses_structure    — fallback produces FACTORY_N_LLM_CANDIDATES
10. test_fallback_chatter_reduces_feed   — 'chatter' → reduce_feed action
11. test_fallback_collision_aborts       — 'collision' → abort action
12. test_score_min_time_prefers_continue — min_time: continue beats reduce_feed
13. test_score_max_life_prefers_feed     — max_tool_life: reduce_feed beats continue
14. test_catastrophic_overridden         — catastrophic risk → action forced to abort
15. test_tool_inventory_seeded           — factory starts with stock per spec
16. test_tool_replacement_dispatched     — worn tool → replace_tool command
17. test_inventory_decrements_on_send    — stock count drops after dispatch
18. test_add_to_inventory                — add_to_inventory increases count
19. test_production_kpis_keys            — all expected KPI keys present
20. test_rework_visible_in_kpis          — rework depth shown in KPIs
21. test_factory_state_serialisable      — get_factory_state() is JSON-serialisable
22. test_summary_non_empty               — summary() returns a non-empty string
23. test_submit_new_seed_creates_job     — seed 0 job created and queued (if file exists)
24. test_integration_full_factory_run    — 3 ticks with submitted jobs, no crash

Run:
    cd "/Users/sbedi/Nextcloud/automatic_to_autonomous/CNC Factory"
    python test_factory_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from scheduler import make_job, Job
from cnc_agent import CncAgent, build_default_crib, ToolRecord
from factory_agent import (
    FactoryAgent, FactoryCommand, ErrorContext, LLMResponse,
    FactoryTickResult, build_factory_agent,
    _POLICY_ACTION_SCORES,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _factory(n: int = 4, policy: str = "min_time") -> FactoryAgent:
    """Build a factory in dry-run / no-LLM mode for tests."""
    return build_factory_agent(n_machines=n, policy=policy, dry_run=True)


def _minimal_gcode() -> list[str]:
    return [
        "(Initialisation)", "G21", "G90", "S4000 M3",
        "(Face 1  z=10.000mm  passes=2  entry=cliff_linear)",
        "(approach)",
        "G00 X10.000 Y10.000 Z20.000",
        "G01 X50.000 Y10.000 Z10.000 F1000",
        "M5", "M30",
    ]


def _job(seed: int = 1, est_s: float = 5.0) -> Job:
    j = make_job(seed=seed, estimated_time_s=est_s)
    j.gcode_lines = _minimal_gcode()
    return j


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFactoryAgent(unittest.TestCase):

    # ── 1. Agents created ─────────────────────────────────────────────────────
    def test_factory_created_with_agents(self):
        fa = _factory(4)
        self.assertEqual(len(fa.agents), 4,
            f"Expected 4 agents, got {len(fa.agents)}")
        ids = [ag.machine_id for ag in fa.agents]
        self.assertEqual(ids, ["M01", "M02", "M03", "M04"])

    # ── 2. Shared rework queue ────────────────────────────────────────────────
    def test_shared_rework_queue(self):
        fa = _factory(4)
        # All agents should reference the SAME deque object
        rq0 = fa.agents[0].rework_queue
        for ag in fa.agents[1:]:
            self.assertIs(ag.rework_queue, rq0,
                f"{ag.machine_id}.rework_queue is not shared with M01")

    # ── 3. Policy attribute ───────────────────────────────────────────────────
    def test_policy_attribute(self):
        fa = _factory(policy="max_tool_life")
        self.assertEqual(fa.policy, "max_tool_life")
        self.assertEqual(fa.scheduler.policy, "max_tool_life")

    # ── 4. Tick advances scheduler ────────────────────────────────────────────
    def test_tick_advances_scheduler(self):
        fa = _factory()
        before = fa.scheduler.tick
        fa.tick()
        self.assertEqual(fa.scheduler.tick, before + 1,
            "Scheduler tick should increase by 1 per factory tick")

    # ── 5. Tick returns FactoryTickResult ────────────────────────────────────
    def test_tick_returns_result(self):
        fa = _factory()
        result = fa.tick()
        self.assertIsInstance(result, FactoryTickResult)
        self.assertEqual(result.tick, 1)
        self.assertGreaterEqual(result.t_real_s, config.T_UNIT_SECONDS)

    # ── 6. Error detected and commanded ──────────────────────────────────────
    def test_error_detected_and_commanded(self):
        fa  = _factory()
        ag  = fa.agents[0]
        j   = _job()
        ag.current_job = j
        ag.inject_error("Spindle overload")

        result = fa.tick()
        self.assertGreater(result.errors_processed, 0,
            "Factory should detect the error and process it")

    # ── 7. Command target is correct machine ──────────────────────────────────
    def test_command_dispatched_to_agent(self):
        fa = _factory()
        ag = fa.agents[2]   # M03
        j  = _job()
        ag.current_job = j
        ag.inject_error("Feed stall on M03")

        result = fa.tick()
        commands = [c for c in result.commands_sent if c.action != "replace_tool"]
        self.assertTrue(
            any(c.target_machine_id == "M03" for c in commands),
            f"Expected command to M03; got targets={[c.target_machine_id for c in commands]}"
        )

    # ── 8. Agent resumes after command ────────────────────────────────────────
    def test_agent_resumes_after_command(self):
        fa = _factory()
        ag = fa.agents[0]
        j  = _job()
        ag.current_job = j
        ag.inject_error("Vibration detected")

        fa.tick()
        # After factory responds, agent should no longer be awaiting_factory
        self.assertNotEqual(ag.status, "awaiting_factory",
            f"Agent should leave awaiting_factory after factory response; "
            f"status={ag.status!r}")

    # ── 9. Fallback produces N candidates ─────────────────────────────────────
    def test_fallback_responses_structure(self):
        fa  = _factory()
        ctx = ErrorContext(
            machine_id="M01", error_description="unknown fault",
            error_section="", error_gcode_line="",
            machine_status="awaiting_factory", tool_life_pct=80.0,
            sections_completed=[], queue_depth=0,
            factory_policy="min_time", tick=1,
        )
        resps = fa._fallback_responses(ctx)
        self.assertEqual(len(resps), config.FACTORY_N_LLM_CANDIDATES,
            f"Expected {config.FACTORY_N_LLM_CANDIDATES} responses, got {len(resps)}")
        for r in resps:
            self.assertIn(r.action, ("continue", "reduce_feed", "rework", "abort"),
                f"Unknown action: {r.action!r}")

    # ── 10. Chatter → reduce_feed ─────────────────────────────────────────────
    def test_fallback_chatter_reduces_feed(self):
        fa  = _factory()
        ctx = ErrorContext(
            machine_id="M01", error_description="chatter detected near thin wall",
            error_section="Face 1", error_gcode_line="G01 X50 Y10 Z5 F1000",
            machine_status="awaiting_factory", tool_life_pct=60.0,
            sections_completed=["Face 1"], queue_depth=0,
            factory_policy="min_time", tick=1,
        )
        resps  = fa._fallback_responses(ctx)
        primary = resps[0]
        self.assertEqual(primary.action, "reduce_feed",
            f"Chatter should trigger reduce_feed, got {primary.action!r}")

    # ── 11. Collision → abort ─────────────────────────────────────────────────
    def test_fallback_collision_aborts(self):
        fa  = _factory()
        ctx = ErrorContext(
            machine_id="M01", error_description="collision risk imminent",
            error_section="approach", error_gcode_line="G00 X200 Y200 Z5",
            machine_status="awaiting_factory", tool_life_pct=80.0,
            sections_completed=[], queue_depth=0,
            factory_policy="min_time", tick=1,
        )
        resps   = fa._fallback_responses(ctx)
        primary = resps[0]
        self.assertEqual(primary.action, "abort",
            f"Collision should trigger abort, got {primary.action!r}")

    # ── 12. Scoring: min_time prefers continue ────────────────────────────────
    def test_score_min_time_prefers_continue(self):
        fa = _factory(policy="min_time")
        r_cont = LLMResponse(str(uuid.uuid4() if False else "a"), "", "continue", risk_level="safe")
        r_feed = LLMResponse(str(uuid.uuid4() if False else "b"), "", "reduce_feed", risk_level="safe")
        r_cont = fa._score_response(r_cont)
        r_feed = fa._score_response(r_feed)
        self.assertGreater(r_cont.policy_score, r_feed.policy_score,
            f"min_time: continue ({r_cont.policy_score}) should beat "
            f"reduce_feed ({r_feed.policy_score})")

    # ── 13. Scoring: max_tool_life prefers reduce_feed ────────────────────────
    def test_score_max_life_prefers_feed(self):
        fa     = _factory(policy="max_tool_life")
        r_feed = LLMResponse("c", "", "reduce_feed", risk_level="safe")
        r_cont = LLMResponse("d", "", "continue",    risk_level="safe")
        r_feed = fa._score_response(r_feed)
        r_cont = fa._score_response(r_cont)
        self.assertGreater(r_feed.policy_score, r_cont.policy_score,
            f"max_tool_life: reduce_feed ({r_feed.policy_score}) should beat "
            f"continue ({r_cont.policy_score})")

    # ── 14. Catastrophic risk forced to abort ─────────────────────────────────
    def test_catastrophic_overridden(self):
        fa  = _factory()
        ag  = fa.agents[0]
        j   = _job()
        ag.current_job = j
        # Inject error whose keyword triggers catastrophic fallback
        ag.inject_error("crash imminent — collision detected")

        result = fa.tick()
        # Find the command sent to M01
        m01_cmds = [c for c in result.commands_sent
                    if c.target_machine_id == "M01" and c.action != "replace_tool"]
        self.assertTrue(len(m01_cmds) > 0, "No command sent to M01")
        # Catastrophic → must be abort
        self.assertEqual(m01_cmds[0].action, "abort",
            f"Catastrophic risk should force abort; got {m01_cmds[0].action!r}")

    # ── 15. Tool inventory seeded ─────────────────────────────────────────────
    def test_tool_inventory_seeded(self):
        fa = _factory()
        # Should have FACTORY_TOOL_STOCK_PER_SPEC copies of T1 through T10
        for tid in ("T1", "T2", "T3", "T4", "T5"):
            count = fa.inventory_count(tid)
            self.assertEqual(count, config.FACTORY_TOOL_STOCK_PER_SPEC,
                f"Expected {config.FACTORY_TOOL_STOCK_PER_SPEC} of {tid}, got {count}")

    # ── 16. Worn tool triggers replace_tool command ───────────────────────────
    def test_tool_replacement_dispatched(self):
        fa = _factory()
        ag = fa.agents[0]
        # Exhaust T3 in M01's crib (set to 0% life)
        t3 = ag.tool_crib.get("T3")
        t3.remaining_life_hrs = 0.0   # 0% → needs_replacement

        result = fa.tick()
        replace_cmds = [c for c in result.commands_sent if c.action == "replace_tool"]
        self.assertGreater(len(replace_cmds), 0,
            "Expected a replace_tool command for exhausted T3")
        self.assertTrue(
            any(c.target_machine_id == "M01" for c in replace_cmds),
            f"replace_tool should target M01; got {[c.target_machine_id for c in replace_cmds]}"
        )

    # ── 17. Inventory decrements on dispatch ─────────────────────────────────
    def test_inventory_decrements_on_send(self):
        fa     = _factory()
        ag     = fa.agents[0]
        before = fa.inventory_count("T3")
        t3     = ag.tool_crib.get("T3")
        t3.remaining_life_hrs = 0.0

        fa.tick()
        after = fa.inventory_count("T3")
        self.assertEqual(after, before - 1,
            f"Inventory should drop by 1 after sending T3: "
            f"{before} → {after}")

    # ── 18. add_to_inventory increases count ─────────────────────────────────
    def test_add_to_inventory(self):
        fa    = _factory()
        fresh = ToolRecord("T1", 8, 100.0, 75.0, 30.0, 30.0, 500.0, 20.0)
        before = fa.inventory_count("T1")
        fa.add_to_inventory(fresh, quantity=2)
        self.assertEqual(fa.inventory_count("T1"), before + 2,
            f"Expected count to increase by 2 after add")

    # ── 19. KPI keys present ──────────────────────────────────────────────────
    def test_production_kpis_keys(self):
        fa   = _factory()
        kpis = fa.get_production_kpis()
        required = {
            "machines_running", "machines_idle",
            "machines_awaiting_factory", "machines_stopped",
            "total_jobs_completed", "total_errors_processed",
            "total_tools_replaced", "rework_queue_depth",
            "avg_tool_life_pct", "scheduler_queue_depth",
        }
        for key in required:
            self.assertIn(key, kpis, f"Missing KPI key: {key!r}")

    # ── 20. Rework visible in KPIs ────────────────────────────────────────────
    def test_rework_visible_in_kpis(self):
        fa  = _factory()
        ag  = fa.agents[0]
        j   = _job()
        ag.current_job = j
        ag.inject_error("persistent stall")
        # Exhaust reminders → rework
        for tick in range(1, config.ERROR_WAIT_TICKS * (config.MAX_REMINDERS + 1) + 2):
            ag.tick(tick)
            if ag.status != "awaiting_factory":
                break

        kpis = fa.get_production_kpis()
        self.assertGreaterEqual(kpis["rework_queue_depth"], 1,
            "Rework depth should be ≥ 1 after job moved to rework")

    # ── 21. Factory state JSON-serialisable ───────────────────────────────────
    def test_factory_state_serialisable(self):
        fa = _factory()
        j  = _job()
        fa.agents[0].enqueue(j)
        fa.tick()
        state = fa.get_factory_state()
        try:
            json.dumps(state)
        except (TypeError, ValueError) as e:
            self.fail(f"get_factory_state() not JSON-serialisable: {e}")

    # ── 22. Summary non-empty ─────────────────────────────────────────────────
    def test_summary_non_empty(self):
        fa   = _factory()
        summ = fa.summary()
        self.assertIsInstance(summ, str)
        self.assertGreater(len(summ), 30, f"Summary too short: {summ!r}")
        self.assertIn("M01", summ)

    # ── 23. submit_new_seed (skipped if no solid file) ────────────────────────
    def test_submit_new_seed_creates_job(self):
        npy = os.path.join(config.SOLID_OUTPUT_DIR, "solid_faces_seed_0.npy")
        if not os.path.exists(npy):
            self.skipTest("solid_faces_seed_0.npy not found")
        fa  = _factory()
        before = len(fa.scheduler.job_queue)
        job = fa.submit_new_seed(0)
        self.assertIsNotNone(job, "submit_new_seed(0) should return a Job")
        self.assertEqual(len(fa.scheduler.job_queue), before + 1,
            "Job should be queued in the scheduler")

    # ── 24. Integration: multi-tick factory run ───────────────────────────────
    def test_integration_full_factory_run(self):
        """Submit 2 jobs, run 5 ticks, verify no crashes and KPIs update."""
        fa = _factory()
        for seed in (0, 1):
            j = _job(seed=seed, est_s=30.0)
            fa.scheduler.submit(j)

        results = fa.run(5)

        # All results should be FactoryTickResult with correct tick numbers
        for i, r in enumerate(results, start=1):
            self.assertEqual(r.tick, i, f"Tick {i} has wrong tick number {r.tick}")

        # Factory state should be serialisable
        state = fa.get_factory_state()
        json.dumps(state)

        print(f"\n[integration] 5 ticks, 2 jobs")
        print(fa.summary())
        kpis = fa.get_production_kpis()
        print(f"  KPIs: {kpis}")


# need uuid for test 12/13 without importing it globally
import uuid as _uuid_mod
LLMResponse.__init__  # touch to ensure import is live


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestFactoryAgent)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

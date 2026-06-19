"""
test_factory_agent.py — Phase 7 tests for factory_agent.py  (27 tests)

 1. test_factory_created_with_agents
 2. test_shared_rework_queue
 3. test_policy_attribute
 4. test_tick_advances_scheduler
 5. test_tick_returns_result
 6. test_error_detected_and_commanded
 7. test_command_dispatched_to_agent
 8. test_agent_resumes_after_command
 9. test_fallback_responses_structure
10. test_fallback_chatter_reduces_feed
11. test_fallback_collision_aborts
12. test_score_min_time_prefers_continue
13. test_score_max_life_prefers_feed
14. test_catastrophic_overridden
15. test_tool_inventory_seeded
16. test_tool_replacement_dispatched
17. test_inventory_decrements_on_send
18. test_add_to_inventory
19. test_production_kpis_keys
20. test_rework_visible_in_kpis
21. test_factory_state_serialisable
22. test_summary_non_empty
23. test_submit_new_seed_creates_job
24. test_integration_full_factory_run
25. test_operator_context_appended_to_error
26. test_operator_override_applies_abort
27. test_operator_feedback_visible_in_state
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


# ── helpers -------------------------------------------------------------------

def _factory(n=4, policy="min_time"):
    return build_factory_agent(n_machines=n, policy=policy, dry_run=True)

def _gcode():
    return [
        "(Initialisation)", "G21", "G90", "S4000 M3",
        "(Face 1  z=10.000mm  passes=2  entry=cliff_linear)",
        "(approach)",
        "G00 X10.000 Y10.000 Z20.000",
        "G01 X50.000 Y10.000 Z10.000 F1000",
        "M5", "M30",
    ]

def _job(seed=1, est_s=5.0):
    j = make_job(seed=seed, estimated_time_s=est_s)
    j.gcode_lines = _gcode()
    return j

def _ctx(desc="unknown fault", policy="min_time"):
    return ErrorContext(
        machine_id="M01", error_description=desc,
        error_section="", error_gcode_line="",
        machine_status="awaiting_factory", tool_life_pct=80.0,
        sections_completed=[], queue_depth=0,
        factory_policy=policy, tick=1,
    )


# ── tests ---------------------------------------------------------------------

class TestFactoryAgent(unittest.TestCase):

    # 1
    def test_factory_created_with_agents(self):
        fa = _factory(4)
        self.assertEqual(len(fa.agents), 4)
        self.assertEqual([ag.machine_id for ag in fa.agents],
                         ["M01","M02","M03","M04"])

    # 2
    def test_shared_rework_queue(self):
        fa = _factory(4)
        rq = fa.agents[0].rework_queue
        for ag in fa.agents[1:]:
            self.assertIs(ag.rework_queue, rq,
                f"{ag.machine_id} rework_queue is not shared")

    # 3
    def test_policy_attribute(self):
        fa = _factory(policy="max_tool_life")
        self.assertEqual(fa.policy, "max_tool_life")
        self.assertEqual(fa.scheduler.policy, "max_tool_life")

    # 4
    def test_tick_advances_scheduler(self):
        fa = _factory()
        before = fa.scheduler.tick
        fa.tick()
        self.assertEqual(fa.scheduler.tick, before + 1)

    # 5
    def test_tick_returns_result(self):
        fa = _factory()
        r  = fa.tick()
        self.assertIsInstance(r, FactoryTickResult)
        self.assertEqual(r.tick, 1)
        self.assertGreaterEqual(r.t_real_s, config.T_UNIT_SECONDS)

    # 6
    def test_error_detected_and_commanded(self):
        fa = _factory()
        ag = fa.agents[0]
        ag.current_job = _job()
        ag.inject_error("Spindle overload")
        r = fa.tick()
        self.assertGreater(r.errors_processed, 0)

    # 7
    def test_command_dispatched_to_agent(self):
        fa = _factory()
        ag = fa.agents[2]   # M03
        ag.current_job = _job()
        ag.inject_error("Feed stall on M03")
        r = fa.tick()
        cmds = [c for c in r.commands_sent if c.action != "replace_tool"]
        self.assertTrue(any(c.target_machine_id == "M03" for c in cmds),
            f"No command to M03; targets={[c.target_machine_id for c in cmds]}")

    # 8
    def test_agent_resumes_after_command(self):
        fa = _factory()
        ag = fa.agents[0]
        ag.current_job = _job()
        ag.inject_error("Vibration detected")
        fa.tick()
        self.assertNotEqual(ag.status, "awaiting_factory",
            f"status={ag.status!r} should not be awaiting_factory")

    # 9
    def test_fallback_responses_structure(self):
        fa    = _factory()
        resps = fa._fallback_responses(_ctx())
        self.assertEqual(len(resps), config.FACTORY_N_LLM_CANDIDATES)
        valid = {"continue","reduce_feed","rework","abort"}
        for r in resps:
            self.assertIn(r.action, valid, f"Unknown action {r.action!r}")

    # 10
    def test_fallback_chatter_reduces_feed(self):
        fa = _factory()
        resps = fa._fallback_responses(_ctx("chatter detected near thin wall"))
        self.assertEqual(resps[0].action, "reduce_feed",
            f"Chatter -> reduce_feed expected; got {resps[0].action!r}")

    # 11
    def test_fallback_collision_aborts(self):
        fa = _factory()
        resps = fa._fallback_responses(_ctx("collision risk imminent"))
        self.assertEqual(resps[0].action, "abort",
            f"Collision -> abort expected; got {resps[0].action!r}")

    # 12
    def test_score_min_time_prefers_continue(self):
        fa  = _factory(policy="min_time")
        r_c = fa._score_response(LLMResponse("a","","continue", risk_level="safe"))
        r_f = fa._score_response(LLMResponse("b","","reduce_feed", risk_level="safe"))
        self.assertGreater(r_c.policy_score, r_f.policy_score,
            f"min_time: continue {r_c.policy_score} should > reduce_feed {r_f.policy_score}")

    # 13
    def test_score_max_life_prefers_feed(self):
        fa  = _factory(policy="max_tool_life")
        r_f = fa._score_response(LLMResponse("c","","reduce_feed", risk_level="safe"))
        r_c = fa._score_response(LLMResponse("d","","continue",    risk_level="safe"))
        self.assertGreater(r_f.policy_score, r_c.policy_score,
            f"max_tool_life: reduce_feed {r_f.policy_score} should > continue {r_c.policy_score}")

    # 14
    def test_catastrophic_overridden(self):
        fa = _factory()
        ag = fa.agents[0]
        ag.current_job = _job()
        ag.inject_error("crash imminent — collision detected")
        r  = fa.tick()
        m01 = [c for c in r.commands_sent
               if c.target_machine_id=="M01" and c.action!="replace_tool"]
        self.assertTrue(len(m01) > 0, "No command sent to M01")
        self.assertEqual(m01[0].action, "abort",
            f"Catastrophic -> abort; got {m01[0].action!r}")

    # 15
    def test_tool_inventory_seeded(self):
        fa = _factory()
        for tid in ("T1","T2","T3","T4","T5"):
            self.assertEqual(fa.inventory_count(tid),
                             config.FACTORY_TOOL_STOCK_PER_SPEC,
                f"{tid} expected {config.FACTORY_TOOL_STOCK_PER_SPEC}, "
                f"got {fa.inventory_count(tid)}")

    # 16
    def test_tool_replacement_dispatched(self):
        fa = _factory()
        ag = fa.agents[0]
        ag.tool_crib.get("T3").remaining_life_hrs = 0.0
        r  = fa.tick()
        rcmds = [c for c in r.commands_sent if c.action=="replace_tool"]
        self.assertGreater(len(rcmds), 0, "Expected replace_tool command")
        self.assertTrue(any(c.target_machine_id=="M01" for c in rcmds))

    # 17
    def test_inventory_decrements_on_send(self):
        fa     = _factory()
        before = fa.inventory_count("T3")
        fa.agents[0].tool_crib.get("T3").remaining_life_hrs = 0.0
        fa.tick()
        self.assertEqual(fa.inventory_count("T3"), before - 1,
            f"T3 inventory {before} -> should be {before-1}")

    # 18
    def test_add_to_inventory(self):
        fa    = _factory()
        fresh = ToolRecord("T1",8,100.0,75.0,30.0,30.0,500.0,20.0)
        before = fa.inventory_count("T1")
        fa.add_to_inventory(fresh, quantity=2)
        self.assertEqual(fa.inventory_count("T1"), before + 2)

    # 19
    def test_production_kpis_keys(self):
        fa   = _factory()
        kpis = fa.get_production_kpis()
        for key in (
            "machines_running","machines_idle",
            "machines_awaiting_factory","machines_stopped",
            "total_jobs_completed","total_errors_processed",
            "total_tools_replaced","operator_interventions","rework_queue_depth",
            "avg_tool_life_pct","scheduler_queue_depth",
        ):
            self.assertIn(key, kpis, f"Missing KPI: {key!r}")

    # 20
    def test_rework_visible_in_kpis(self):
        fa = _factory()
        ag = fa.agents[0]
        ag.current_job = _job()
        ag.inject_error("persistent stall")
        # drain reminders
        limit = config.ERROR_WAIT_TICKS * (config.MAX_REMINDERS + 1) + 2
        for t in range(1, limit):
            ag.tick(t)
            if ag.status != "awaiting_factory":
                break
        self.assertGreaterEqual(fa.get_production_kpis()["rework_queue_depth"], 1)

    # 21
    def test_factory_state_serialisable(self):
        fa = _factory()
        fa.agents[0].enqueue(_job())
        fa.tick()
        try:
            json.dumps(fa.get_factory_state())
        except (TypeError, ValueError) as e:
            self.fail(f"get_factory_state not JSON-serialisable: {e}")

    # 22
    def test_summary_non_empty(self):
        fa   = _factory()
        summ = fa.summary()
        self.assertIsInstance(summ, str)
        self.assertGreater(len(summ), 30)
        self.assertIn("M01", summ)

    # 23
    def test_submit_new_seed_creates_job(self):
        import os, config
        npy = os.path.join(config.SOLID_OUTPUT_DIR,"solid_faces_seed_0.npy")
        if not os.path.exists(npy):
            self.skipTest("solid_faces_seed_0.npy not found")
        fa     = _factory()
        before = len(fa.scheduler.job_queue)
        job    = fa.submit_new_seed(0)
        self.assertIsNotNone(job)
        self.assertEqual(len(fa.scheduler.job_queue), before + 1)

    # 24
    def test_integration_full_factory_run(self):
        fa = _factory()
        for s in (1, 2):
            fa.scheduler.submit(_job(seed=s, est_s=30.0))
        results = fa.run(5)
        for i, r in enumerate(results, 1):
            self.assertEqual(r.tick, i)
        json.dumps(fa.get_factory_state())   # must not raise
        print("\n[integration] 5-tick run")
        print(fa.summary())
        print("KPIs:", fa.get_production_kpis())

    # 25
    def test_operator_context_appended_to_error(self):
        fa = _factory()
        ag = fa.agents[0]
        ag.current_job = _job()
        ag.inject_error("tool chatter detected")
        rec = fa.handle_operator_input("M01", "Check vise clamp near jaw 2")
        self.assertFalse(rec["applied"])
        self.assertEqual(rec["context"]["event_type"], "operator_note")
        self.assertEqual(rec["context"]["category"], "tooling")
        self.assertEqual(rec["context"]["location"], "workholding")
        self.assertIn("operator: Check vise clamp near jaw 2",
                      ag.active_error.description)
        self.assertEqual(ag.status, "awaiting_factory")

    # 26
    def test_operator_override_applies_abort(self):
        fa = _factory()
        ag = fa.agents[0]
        ag.current_job = _job()
        ag.inject_error("collision risk imminent")
        rec = fa.handle_operator_input("M01", "Abort the job now")
        self.assertTrue(rec["applied"])
        self.assertEqual(rec["action"], "abort")
        self.assertEqual(ag.status, "stopped")
        self.assertIsNotNone(ag.active_error)
        self.assertTrue(ag.active_error.resolved)

    # 27
    def test_operator_feedback_visible_in_state(self):
        fa = _factory()
        ag = fa.agents[0]
        ag.current_job = _job()
        ag.inject_error("vibration detected")
        fa.handle_operator_input("M01", "Reduce feed to 65%")
        state = fa.get_factory_state()
        self.assertEqual(len(state["operator_feedback_log"]), 1)
        self.assertEqual(state["operator_feedback_log"][0]["action"], "reduce_feed")
        self.assertEqual(
            state["operator_feedback_log"][0]["parameters"]["feed_override_pct"],
            65.0,
        )
        self.assertEqual(
            state["operator_feedback_log"][0]["context"]["event_type"],
            "operator_override",
        )
        self.assertEqual(
            state["operator_context_by_machine"]["M01"]["category"],
            "process_parameters",
        )
        self.assertIsInstance(json.dumps(state), str)

    def test_operator_context_included_in_error_prompt(self):
        fa = _factory()
        ag = fa.agents[0]
        ag.current_job = _job()
        ag.inject_error("chatter near wall")
        fa.handle_operator_input("M01", "Severe chatter near wall")
        ctx = fa._build_error_context(ag, tick_num=3)
        prompt = fa._build_llm_prompt(ctx)
        self.assertEqual(ctx.operator_context["category"], "vibration")
        self.assertEqual(ctx.operator_context["severity"], "high")
        self.assertIn("Operator context:", prompt)
        self.assertIn('"category": "vibration"', prompt)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestFactoryAgent)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

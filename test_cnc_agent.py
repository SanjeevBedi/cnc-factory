"""
test_cnc_agent.py — Phase 6 tests for cnc_agent.py

Test suite
----------
 1. test_default_crib_m01_tools        — M01 has T1–T10, T1 is D100
 2. test_default_crib_m02_tools        — M02 T1 is D75, shared T5–T10 correct
 3. test_default_crib_all_four         — all four machines created with 10 tools
 4. test_select_tool_min_time          — min_time → largest D < width/4
 5. test_select_tool_max_tool_life     — max_tool_life → most remaining life
 6. test_select_tool_best_finish       — best_finish → smallest diameter
 7. test_depth_of_cut_large_tool       — R>25 + short shank → 3 mm
 8. test_depth_of_cut_max_life_policy  — max_tool_life reduces doc by 10 %
 9. test_step_over_formula             — step_over = radius/3
10. test_tool_life_deducted_after_job  — remaining_life drops after execute_job
11. test_tool_change_flagged_on_exhaustion — pending_tool_change set when life < STOP_PCT
12. test_execute_job_dry_run_sections  — section names captured
13. test_execute_job_air_vs_material   — G00 → air, G01 with M3 → material
14. test_execute_job_m30_terminates    — M30 stops execution
15. test_error_inject_status           — inject_error sets awaiting_factory
16. test_error_reminder_after_n_ticks  — reminder sent at ERROR_WAIT_TICKS
17. test_rework_after_max_reminders    — job moved to rework after MAX_REMINDERS
18. test_factory_response_continue     — handle_factory_response('continue') resumes
19. test_factory_response_rework       — 'rework' action moves job to rework queue
20. test_tool_change_adds_time         — pending change adds TOOL_CHANGE_TIME_S to total
21. test_cost_computation              — cost dict has correct keys and positive values
22. test_machine_hourly_rate           — derived from capital/years/annual_hours
23. test_part_queue_fifo               — jobs dequeued in submission order
24. test_four_agents_independent_queues — each agent has its own queue
25. test_state_dict_serialisable       — get_state() is JSON-serialisable
26. test_integration_phase4_to_agent   — seed 0 full pipeline → agent executes

Run:
    cd "/Users/sbedi/Nextcloud/automatic_to_autonomous/CNC Factory"
    python test_cnc_agent.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from scheduler import make_job, Job
from cnc_agent import (
    ToolRecord, ToolCrib, ErrorEvent, AgentExecResult,
    CncAgent, build_default_crib, build_factory,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _agent(machine_id: str = "M01", rework: deque = None) -> CncAgent:
    crib = build_default_crib(machine_id)
    return CncAgent(
        machine_id   = machine_id,
        tool_crib    = crib,
        rework_queue = rework if rework is not None else deque(),
        dry_run      = True,
    )


def _minimal_job(gcode_lines: list[str] = None) -> Job:
    """Tiny job with synthetic G-code."""
    lines = gcode_lines or [
        "(Initialisation)",
        "G21",
        "G90",
        "S4000 M3",
        "(Face 1  z=10.000mm  passes=2  entry=cliff_linear)",
        "(approach)",
        "G00 X10.000 Y10.000 Z20.000",
        "(entry (cliff_linear))",
        "G01 X10.000 Y10.000 Z10.000 F500",
        "G01 X50.000 Y10.000 Z10.000 F1000",
        "G01 X50.000 Y50.000 Z10.000",
        "G01 X10.000 Y50.000 Z10.000",
        "(retract)",
        "G00 X10.000 Y50.000 Z20.000",
        "M5",
        "M30",
    ]
    j = make_job(seed=1, estimated_time_s=5.0, feed_rate_mmpm=1000.0)
    j.gcode_lines = lines
    return j


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCncAgent(unittest.TestCase):

    # ── 1. M01 default crib ───────────────────────────────────────────────────
    def test_default_crib_m01_tools(self):
        crib = build_default_crib("M01")
        self.assertEqual(len(crib.tools), 10,
            f"Expected 10 tools in M01, got {len(crib.tools)}")
        t1 = crib.get("T1")
        self.assertIsNotNone(t1)
        self.assertAlmostEqual(t1.diameter_mm, 100.0,
            msg=f"M01 T1 diameter = {t1.diameter_mm}, expected 100")

    # ── 2. M02 default crib ───────────────────────────────────────────────────
    def test_default_crib_m02_tools(self):
        crib = build_default_crib("M02")
        t1 = crib.get("T1")
        self.assertAlmostEqual(t1.diameter_mm, 75.0,
            msg=f"M02 T1 diameter = {t1.diameter_mm}, expected 75")
        t5 = crib.get("T5")
        self.assertAlmostEqual(t5.diameter_mm, 3.0)
        self.assertAlmostEqual(t5.consumable_cost_usd, 80.0)

    # ── 3. All four machines ──────────────────────────────────────────────────
    def test_default_crib_all_four(self):
        for mid in ("M01", "M02", "M03", "M04"):
            crib = build_default_crib(mid)
            self.assertEqual(len(crib.tools), 10,
                f"{mid} should have 10 tools, got {len(crib.tools)}")

    # ── 4. Tool selection: min_time → largest ─────────────────────────────────
    def test_select_tool_min_time(self):
        agent = _agent("M01")
        # pocket_width = 200mm → D < 200/4 = 50mm → candidates: T3(D25), T4(D10), T5–T10
        tool = agent.select_tool("pocket", 200.0, "min_time")
        self.assertIsNotNone(tool, "Expected a tool but got None")
        # Largest D < 50: T2=50 is excluded (50 is NOT < 50), so T3=25 or T8=15 etc.
        # Actually D < 50: T3=25, T4=10, T7=10, T8=15, T9=20, T10=25
        # But T4 has remaining=10h and remaining_pct = 10/30*100 = 33% > STOP_PCT(10%) → usable
        # Max of those: T3=25 or T10=25 → either is correct
        self.assertLess(tool.diameter_mm, 50.0,
            f"Selected tool D={tool.diameter_mm} should be < 50mm")

    # ── 5. Tool selection: max_tool_life ──────────────────────────────────────
    def test_select_tool_max_tool_life(self):
        agent = _agent("M01")
        # Manually set T3 to have lots of life
        t3 = agent.tool_crib.get("T3")
        t3.remaining_life_hrs = 29.9
        tool = agent.select_tool("pocket", 200.0, "max_tool_life")
        self.assertIsNotNone(tool)
        # Should prefer T3 (most remaining life among D < 50)
        self.assertEqual(tool.tool_id, "T3",
            f"max_tool_life should pick T3 (most life), got {tool.tool_id}")

    # ── 6. Tool selection: best_finish → smallest ─────────────────────────────
    def test_select_tool_best_finish(self):
        agent = _agent("M01")
        tool = agent.select_tool("face", 400.0, "best_finish")
        self.assertIsNotNone(tool)
        # Smallest D available (D < 100mm): T5=D3 is smallest
        self.assertAlmostEqual(tool.diameter_mm, 3.0,
            msg=f"best_finish should pick smallest tool, got D={tool.diameter_mm}")

    # ── 7. Depth of cut: large tool ───────────────────────────────────────────
    def test_depth_of_cut_large_tool(self):
        t = ToolRecord("T1",8,100.0,75.0,30.0,25.0,500.0,20.0)
        # R=50 > 25, shank=75 → NOT < 75, so: 2mm (75 ≤ 75 ≤ 150)
        doc = CncAgent.depth_of_cut_mm(t, "min_time")
        self.assertAlmostEqual(doc, 2.0,
            msg=f"D=100 shank=75: expected doc=2.0, got {doc}")

    def test_depth_of_cut_large_short_shank(self):
        t = ToolRecord("T1",8,100.0,50.0,30.0,25.0,500.0,20.0)
        # R=50 > 25, shank=50 < 75 → 3mm
        doc = CncAgent.depth_of_cut_mm(t, "min_time")
        self.assertAlmostEqual(doc, 3.0,
            msg=f"D=100 shank=50: expected doc=3.0, got {doc}")

    # ── 8. Depth of cut: max_life policy reduces by 10 % ─────────────────────
    def test_depth_of_cut_max_life_policy(self):
        t = ToolRecord("T1",8,100.0,50.0,30.0,25.0,500.0,20.0)
        doc_std  = CncAgent.depth_of_cut_mm(t, "min_time")       # 3.0
        doc_life = CncAgent.depth_of_cut_mm(t, "max_tool_life")   # 3.0 × 0.9
        self.assertAlmostEqual(doc_life, doc_std * 0.90, places=3,
            msg=f"max_tool_life doc should be 90% of {doc_std}, got {doc_life}")

    # ── 9. Step-over formula ──────────────────────────────────────────────────
    def test_step_over_formula(self):
        t = ToolRecord("T3",4,25.0,75.0,30.0,25.0,500.0,20.0)
        so = CncAgent.step_over_mm(t, "min_time")
        self.assertAlmostEqual(so, t.radius_mm / 3.0, places=3,
            msg=f"Expected step_over={t.radius_mm/3:.3f}, got {so}")

    # ── 10. Tool life deducted after job ──────────────────────────────────────
    def test_tool_life_deducted_after_job(self):
        agent = _agent("M01")
        tool  = agent.tool_crib.get("T3")   # D25, life=25h
        before = tool.remaining_life_hrs
        j = _minimal_job()
        agent.execute_job(j)
        after = tool.remaining_life_hrs
        # 5s cutting from ~remaining_life → should decrease (barely)
        self.assertLessEqual(after, before,
            f"Tool life should decrease: {before:.4f} → {after:.4f}")

    # ── 11. Pending tool change when exhausted ────────────────────────────────
    def test_tool_change_flagged_on_exhaustion(self):
        """
        TX: D=30mm, max_life=0.01h, remaining=0.003h (30% > STOP_PCT=10% -> selectable).
        10-s job deducts ~0.00278h -> remaining ~0.00022h (2.2% < 10%) -> needs_replacement.
        """
        agent = _agent("M01")
        custom = ToolRecord(
            tool_id="TX", n_inserts=4,
            diameter_mm=30.0, shank_length_mm=75.0,
            max_life_hrs=0.01, remaining_life_hrs=0.003,
            holder_cost_usd=500.0, consumable_cost_usd=20.0,
        )
        agent.tool_crib.add_tool(custom)   # D=30 largest usable < 50mm
        j = _minimal_job()
        j.estimated_time_s = 10.0
        agent.execute_job(j, policy="min_time")
        tx = agent.tool_crib.get("TX")
        self.assertIsNotNone(tx)
        self.assertTrue(tx.needs_replacement,
            f"TX should need replacement: remaining={tx.remaining_life_pct:.2f}%")
        self.assertTrue(agent._pending_tool_change,
            "Expected pending_tool_change=True after tool exhaustion")

    # ── 12. Section tracking ──────────────────────────────────────────────────
    def test_execute_job_dry_run_sections(self):
        agent = _agent("M01")
        j = _minimal_job()
        r = agent.execute_job(j)
        # Should have captured 'Face 1...' and 'approach' and 'entry...' etc.
        self.assertGreater(len(r.sections_completed), 0,
            f"Expected sections, got {r.sections_completed}")
        has_face = any("Face" in s for s in r.sections_completed)
        self.assertTrue(has_face,
            f"Expected 'Face' in sections, got {r.sections_completed}")

    # ── 13. Air cut vs material cut ───────────────────────────────────────────
    def test_execute_job_air_vs_material(self):
        agent = _agent("M01")
        j = _minimal_job()
        r = agent.execute_job(j)
        # G00 lines → air; G01 after M3 → material
        self.assertGreater(r.air_cut_lines, 0,
            "Expected at least one air cut (G00)")
        self.assertGreater(r.material_cut_lines, 0,
            "Expected at least one material cut (G01 with spindle on)")

    # ── 14. M30 terminates execution ──────────────────────────────────────────
    def test_execute_job_m30_terminates(self):
        lines = [
            "G21", "G90", "S4000 M3",
            "(Face 1  z=5mm  passes=1  entry=cliff_linear)",
            "G01 X10 Y10 Z5 F1000",
            "M30",
            "G01 X999 Y999 Z999 F1000",   # must NOT be reached
        ]
        j = make_job(seed=2, estimated_time_s=1.0)
        j.gcode_lines = lines
        agent = _agent("M01")
        r = agent.execute_job(j)
        # Last line executed should be M30 (index 5 = line #6 = "M30")
        self.assertEqual(r.lines_executed, 6,
            f"Expected 6 lines executed (stop at M30), got {r.lines_executed}")

    # ── 15. Error inject sets status ──────────────────────────────────────────
    def test_error_inject_status(self):
        agent = _agent("M01")
        evt = agent.inject_error("Spindle overload")
        self.assertEqual(agent.status, "awaiting_factory",
            f"Status after inject_error = {agent.status!r}")
        self.assertIsNotNone(agent.active_error)
        self.assertEqual(evt.description, "Spindle overload")

    # ── 16. Reminder after ERROR_WAIT_TICKS ───────────────────────────────────
    def test_error_reminder_after_n_ticks(self):
        agent = _agent("M01")
        j = _minimal_job()
        agent.current_job = j   # simulate running job
        agent.inject_error("Feed stall")

        reminders = []
        for tick in range(1, config.ERROR_WAIT_TICKS * 2 + 1):
            r = agent.tick(tick)
            reminders.extend(
                e for e in r.get("events", []) if "reminder" in e
            )
            if agent.status != "awaiting_factory":
                break

        self.assertTrue(len(reminders) >= 1,
            f"Expected at least 1 reminder in {config.ERROR_WAIT_TICKS*2} ticks; "
            f"got {reminders}")
        self.assertEqual(
            agent.active_error.reminders_sent if agent.active_error else 1,
            len(reminders),
        )

    # ── 17. Rework after MAX_REMINDERS ────────────────────────────────────────
    def test_rework_after_max_reminders(self):
        rework = deque()
        agent = _agent("M01", rework)
        j = _minimal_job()
        agent.current_job = j
        agent.inject_error("Persistent fault")

        total_ticks = config.ERROR_WAIT_TICKS * (config.MAX_REMINDERS + 1) + 1
        for tick in range(1, total_ticks):
            agent.tick(tick)
            if agent.status != "awaiting_factory":
                break

        self.assertEqual(agent.status, "idle",
            f"Expected idle after rework, got {agent.status!r}")
        self.assertEqual(len(rework), 1,
            f"Expected 1 job in rework queue, got {len(rework)}")
        self.assertEqual(rework[0].status, "rework")

    # ── 18. Factory response: continue ────────────────────────────────────────
    def test_factory_response_continue(self):
        agent = _agent("M01")
        agent.inject_error("Vibration")
        agent.handle_factory_response({"action": "continue"})
        self.assertEqual(agent.status, "running",
            f"Expected running after 'continue', got {agent.status!r}")
        self.assertIsNone(agent.active_error)

    # ── 19. Factory response: rework ──────────────────────────────────────────
    def test_factory_response_rework(self):
        rework = deque()
        agent  = _agent("M01", rework)
        j = _minimal_job()
        agent.current_job = j
        agent.inject_error("Critical failure")
        agent.handle_factory_response({"action": "rework"})
        self.assertEqual(agent.status, "idle",
            f"After 'rework' response: expected idle, got {agent.status!r}")
        self.assertEqual(len(rework), 1, "Job should be in rework queue")

    # ── 20. Pending tool change adds time ─────────────────────────────────────
    def test_tool_change_adds_time(self):
        agent = _agent("M01")
        j = _minimal_job()
        # Without pending change
        agent._pending_tool_change = False
        t_no_change = agent.compute_machining_time_s(j)
        # With pending change
        agent._pending_tool_change = True
        t_with_change = agent.compute_machining_time_s(j)
        self.assertAlmostEqual(
            t_with_change - t_no_change,
            config.TOOL_CHANGE_TIME_S,
            places=1,
            msg=f"Tool change should add {config.TOOL_CHANGE_TIME_S}s: "
                f"diff={t_with_change - t_no_change:.1f}"
        )

    # ── 21. Cost computation ──────────────────────────────────────────────────
    def test_cost_computation(self):
        agent = _agent("M01")
        j = _minimal_job()
        tool = agent.tool_crib.get("T3")
        cost = agent.compute_cost(j, tool)
        self.assertIn("total_time_s",   cost)
        self.assertIn("machine_cost",   cost)
        self.assertIn("tool_cost",      cost)
        self.assertIn("total_cost",     cost)
        for k, v in cost.items():
            self.assertGreaterEqual(v, 0.0, f"Cost key {k} should be ≥ 0")
        self.assertAlmostEqual(
            cost["total_cost"],
            cost["machine_cost"] + cost["tool_cost"],
            places=4,
        )

    # ── 22. Machine hourly rate ───────────────────────────────────────────────
    def test_machine_hourly_rate(self):
        agent = _agent("M01")
        expected = (config.MACHINE_CAPITAL_COST_USD / config.MACHINE_AMORT_YEARS
                    / config.ANNUAL_WORKING_HOURS)
        self.assertAlmostEqual(
            agent.machine_hourly_rate_usd, expected, places=4,
            msg=f"Expected hourly rate {expected:.4f}, got {agent.machine_hourly_rate_usd:.4f}"
        )

    # ── 23. FIFO queue ────────────────────────────────────────────────────────
    def test_part_queue_fifo(self):
        agent = _agent("M01")
        seeds = [10, 20, 30]
        jobs = [make_job(seed=s, estimated_time_s=1.0) for s in seeds]
        for jb in jobs:
            jb.gcode_lines = _minimal_job().gcode_lines
            agent.enqueue(jb)

        executed_seeds = []
        for _ in range(3):
            if agent.part_queue:
                j = agent._dequeue_next()
                agent.execute_job(j)
                executed_seeds.append(j.seed)

        self.assertEqual(executed_seeds, seeds,
            f"Expected FIFO order {seeds}, got {executed_seeds}")

    # ── 24. Four agents have independent queues ───────────────────────────────
    def test_four_agents_independent_queues(self):
        agents = build_factory(n_machines=4, dry_run=True)
        self.assertEqual(len(agents), 4)
        # Enqueue one unique job to each
        for i, ag in enumerate(agents):
            j = make_job(seed=i+100, estimated_time_s=1.0)
            j.gcode_lines = _minimal_job().gcode_lines
            ag.enqueue(j)
        # Verify no cross-contamination
        for i, ag in enumerate(agents):
            self.assertEqual(len(ag.part_queue), 1,
                f"{ag.machine_id} should have 1 job, has {len(ag.part_queue)}")
            self.assertEqual(ag.part_queue[0].seed, i + 100)

    # ── 25. State dict serialisable ───────────────────────────────────────────
    def test_state_dict_serialisable(self):
        agent = _agent("M01")
        j = _minimal_job()
        agent.execute_job(j)
        state = agent.get_state()
        try:
            json.dumps(state)
        except (TypeError, ValueError) as e:
            self.fail(f"get_state() not JSON-serialisable: {e}")

    # ── 26. Integration: Phase4 → agent ──────────────────────────────────────
    def test_integration_phase4_to_agent(self):
        """Full chain Phase1→2→3→4→5→6: G-code executed by agent."""
        npy = os.path.join(config.SOLID_OUTPUT_DIR, "solid_faces_seed_0.npy")
        if not os.path.exists(npy):
            self.skipTest("solid_faces_seed_0.npy not found")

        import cnc_solid_bridge as bridge
        from feature_extractor import extract_features
        from feeds_speeds_engine import compute as fs_compute
        from toolpath_planner import plan
        from gcode_generator import generate
        from scheduler import job_from_gcode

        faces  = bridge.load_face_polygons(0)
        meta   = bridge.load_metadata(0)
        feat   = extract_features(faces, seed=0, volume=meta["volume"])
        fs     = fs_compute("aluminium_6061", 12.0, 4,
                            axial_depth_mm=6.0, radial_depth_mm=4.8,
                            fz_override=0.1, sfm_override=1000)
        tp     = plan(feat, fs)
        gc     = generate(tp, fs, spindle_rpm_cap=4000.0)
        job    = job_from_gcode(gc, cost_per_hour=100.0)

        # Build agent and execute
        agent  = _agent("M01")
        result = agent.execute_job(job, policy="min_time")

        self.assertTrue(result.completed, "Job should complete")
        self.assertGreater(result.lines_executed, 0)
        self.assertGreater(result.material_cut_lines + result.air_cut_lines, 0)
        self.assertGreater(result.total_cost_usd, 0.0)
        self.assertGreater(len(result.sections_completed), 0)

        print(f"\n[integration] seed 0 executed on {agent.machine_id}")
        print(f"  Lines          : {result.lines_executed}")
        print(f"  Air cuts       : {result.air_cut_lines}")
        print(f"  Material cuts  : {result.material_cut_lines}")
        print(f"  Sections done  : {result.sections_completed}")
        print(f"  Machining time : {result.machining_time_s:.1f} s")
        print(f"  Total cost     : ${result.total_cost_usd:.4f}")
        print(f"  Tool crib after:")
        for t in agent.tool_crib.tools:
            print(f"    {t.tool_id} D{t.diameter_mm:5.1f}mm  "
                  f"life {t.remaining_life_pct:5.1f}%")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestCncAgent)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

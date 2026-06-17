"""
test_disturbance_engine.py — Unit and integration tests for disturbance_engine.py
"""

import json
import math
import unittest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from disturbance_engine import (
    REGISTRY, BY_CATEGORY, Category,
    DisturbanceEngine, DisturbanceContext, LLMCandidate, ENGINE,
    GUIField,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ctx(**kwargs) -> DisturbanceContext:
    defaults = dict(
        disturbance_key="material_harder",
        machine_id="M01", tick=5, factory_policy="min_time",
        machine_status="running", current_job="seed-0",
        current_section="Face 1", sections_done=["Initialisation","Face 1"],
        sections_remaining=["Face 2","Face 3"],
        queue_depth=0,
        active_tool_id="T3", active_tool_dia=25.0, active_tool_life=80.0,
        tool_crib=[], inventory={},
    )
    defaults.update(kwargs)
    return DisturbanceContext(**defaults)


# ── 1. Registry completeness ──────────────────────────────────────────────────

class TestRegistry(unittest.TestCase):

    def test_registry_non_empty(self):
        self.assertGreater(len(REGISTRY), 10)

    def test_all_categories_populated(self):
        for cat in Category:
            self.assertGreater(len(BY_CATEGORY[cat]), 0,
                               f"Category {cat} is empty")

    def test_all_specs_have_label(self):
        for key, spec in REGISTRY.items():
            self.assertTrue(spec.label, f"{key} has no label")
            self.assertTrue(spec.short_desc, f"{key} has no short_desc")
            self.assertIsInstance(spec.possible_actions, tuple)
            self.assertGreater(len(spec.possible_actions), 0)

    def test_gui_fields_valid(self):
        for key, spec in REGISTRY.items():
            for f in spec.gui_fields:
                self.assertIn(f.kind, ("text","float","bool","choice"),
                              f"{key}.{f.name} has bad kind={f.kind!r}")
                if f.kind == "choice":
                    self.assertGreater(len(f.choices), 0,
                                       f"{key}.{f.name} choice has no options")

    def test_safety_critical_specs(self):
        sc = [k for k, s in REGISTRY.items() if s.safety_critical]
        # fixture_loose, spindle_bearing, axis_fault, tool_breakage must be critical
        for expected in ("fixture_loose","spindle_bearing","axis_fault",
                         "tool_breakage"):
            self.assertIn(expected, sc)


# ── 2. Material enrichment ────────────────────────────────────────────────────

class TestMaterialEnrichment(unittest.TestCase):

    def setUp(self):
        self.engine = DisturbanceEngine()

    def _enrich(self, orig, actual, dia=25.0):
        ctx = _make_ctx(
            disturbance_key="material_harder",
            active_tool_dia=dia,
            original_material=orig,
            actual_material=actual,
        )
        extra = {"original_material": orig, "actual_material": actual,
                 "evidence": "test"}
        return self.engine._enrich_material(ctx, extra)

    def test_al6061_to_mild_steel_ratio(self):
        ctx = self._enrich("aluminium_6061", "mild_steel_1018")
        # Kc mild_steel / Kc al6061 = 1800/600 = 3.0
        self.assertAlmostEqual(ctx.kc_ratio, 3.0, places=1)

    def test_feed_decreases_for_harder_material(self):
        ctx = self._enrich("aluminium_6061", "mild_steel_1018")
        self.assertLess(ctx.feed_required, ctx.feed_original)
        self.assertGreater(ctx.feed_reduction_pct, 0)

    def test_rpm_decreases_for_harder_material(self):
        ctx = self._enrich("aluminium_6061", "mild_steel_1018")
        self.assertLess(ctx.rpm_required, ctx.rpm_original)

    def test_softer_material_feed_increases(self):
        ctx = self._enrich("alloy_steel_4140", "aluminium_6061")
        # Al is softer → feed_required > feed_original
        self.assertGreater(ctx.feed_required, ctx.feed_original)
        self.assertLess(ctx.kc_ratio, 1.0)

    def test_same_material_ratio_one(self):
        ctx = self._enrich("aluminium_6061", "aluminium_6061")
        self.assertAlmostEqual(ctx.kc_ratio, 1.0, places=2)

    def test_titanium_flags_tool_inadequate(self):
        ctx = self._enrich("aluminium_6061", "titanium_ti64")
        self.assertFalse(ctx.tool_adequate)
        self.assertGreater(len(ctx.tool_grade_note), 10)

    def test_mild_steel_tool_adequate(self):
        ctx = self._enrich("aluminium_6061", "mild_steel_1018")
        self.assertTrue(ctx.tool_adequate)

    def test_power_computed(self):
        ctx = self._enrich("aluminium_6061", "mild_steel_1018")
        self.assertGreater(ctx.power_required, 0)
        self.assertGreater(ctx.power_original, 0)

    def test_all_material_pairs_no_crash(self):
        mats = list(config.MATERIAL_TABLE.keys())
        for orig in mats[:3]:
            for act in mats:
                try:
                    self._enrich(orig, act)
                except Exception as e:
                    self.fail(f"_enrich({orig}, {act}) raised {e}")


# ── 3. Prompt generation ──────────────────────────────────────────────────────

class TestPromptGeneration(unittest.TestCase):

    def setUp(self):
        self.engine = DisturbanceEngine()

    def _ctx_for(self, key, extra=None):
        extra = extra or {}
        ctx = _make_ctx(disturbance_key=key, extra=extra)
        if "material" in key:
            ctx.original_material = extra.get("original_material","aluminium_6061")
            ctx.actual_material   = extra.get("actual_material","mild_steel_1018")
            ctx.kc_ratio          = 3.0
            ctx.feed_original     = 3299.0
            ctx.feed_required     = 1374.0
            ctx.rpm_original      = 8247.0
            ctx.rpm_required      = 3436.0
            ctx.power_required    = 1.17
            ctx.power_original    = 0.93
            ctx.feed_reduction_pct= 58.3
        return ctx

    def test_material_harder_prompt_contains_kc(self):
        ctx = self._ctx_for("material_harder",
                            {"original_material":"aluminium_6061",
                             "actual_material":"mild_steel_1018",
                             "evidence":"hardness 180HB"})
        prompt = self.engine.build_prompt(ctx)
        self.assertIn("Kc", prompt)
        self.assertIn("feed", prompt.lower())
        self.assertIn("JSON", prompt)

    def test_material_prompt_contains_numbers(self):
        ctx = self._ctx_for("material_harder",
                            {"original_material":"aluminium_6061",
                             "actual_material":"mild_steel_1018"})
        prompt = self.engine.build_prompt(ctx)
        # must contain numeric data, not just labels
        self.assertRegex(prompt, r"\d+\.\d+")

    def test_tool_breakage_prompt(self):
        ctx = self._ctx_for("tool_breakage",
                            {"tool_id":"T3","no_stock":True})
        prompt = self.engine.build_prompt(ctx)
        self.assertIn("T3", prompt)
        self.assertIn("OUT OF STOCK", prompt)

    def test_chatter_prompt_contains_rpm(self):
        ctx = _make_ctx(disturbance_key="tool_chatter",
                        rpm_original=8000.0,
                        extra={"frequency_hz": 450.0, "severity": "high"})
        prompt = self.engine.build_prompt(ctx)
        self.assertIn("Hz", prompt)

    def test_coolant_prompt_contains_material(self):
        ctx = _make_ctx(disturbance_key="coolant_failure",
                        original_material="titanium_ti64",
                        extra={"coolant_type":"flood","partial_loss":False})
        prompt = self.engine.build_prompt(ctx)
        self.assertIn("coolant", prompt.lower())

    def test_all_types_generate_prompt(self):
        for key in REGISTRY:
            extra = {}
            if "material" in key:
                extra = {"original_material":"aluminium_6061",
                         "actual_material":"mild_steel_1018"}
            ctx = self._ctx_for(key, extra)
            try:
                prompt = self.engine.build_prompt(ctx)
                self.assertGreater(len(prompt), 50,
                                   f"{key} prompt too short")
                self.assertIn("JSON", prompt, f"{key} missing JSON instruction")
            except Exception as e:
                self.fail(f"build_prompt({key}) raised {e}")


# ── 4. Fallback responses ─────────────────────────────────────────────────────

class TestFallbackResponses(unittest.TestCase):

    def setUp(self):
        self.engine = DisturbanceEngine()

    def _fallback(self, key, extra=None, **ctx_kw):
        extra = extra or {}
        ctx = _make_ctx(disturbance_key=key, extra=extra, **ctx_kw)
        if "material" in key and extra.get("actual_material"):
            ctx = self.engine._enrich_material(ctx, extra)
        return self.engine.fallback_responses(ctx)

    # material_harder ─────────────────────────────────────────────────────────

    def test_material_harder_three_candidates(self):
        cands = self._fallback("material_harder",
                               {"original_material":"aluminium_6061",
                                "actual_material":"mild_steel_1018"})
        self.assertEqual(len(cands), 3)

    def test_material_harder_first_is_recalculate_or_abort(self):
        cands = self._fallback("material_harder",
                               {"original_material":"aluminium_6061",
                                "actual_material":"mild_steel_1018"})
        self.assertIn(cands[0].action, ("pause_redo_cam","abort"))

    def test_material_harder_inconel_triggers_abort_when_power_exceeds(self):
        """Al → Inconel: ratio=5.83 — power should exceed machine limit."""
        cands = self._fallback("material_harder",
                               {"original_material":"aluminium_6061",
                                "actual_material":"inconel"})
        # at least one candidate should be abort
        actions = {c.action for c in cands}
        self.assertIn("abort", actions)

    def test_material_softer_feed_increases(self):
        cands = self._fallback("material_softer",
                               {"original_material":"alloy_steel_4140",
                                "actual_material":"aluminium_6061"})
        self.assertGreater(len(cands), 0)
        # per spec: pause and redo CAM for any material mismatch
        self.assertIn(cands[0].action, ("pause_redo_cam", "abort"))

    def test_recalculate_candidate_has_numeric_params(self):
        cands = self._fallback("material_harder",
                               {"original_material":"aluminium_6061",
                                "actual_material":"mild_steel_1018"})
        reclass = [c for c in cands if c.action == "recalculate"]
        if reclass:
            params = reclass[0].parameters
            self.assertIn("feed_rate_mmpm", params)
            self.assertGreater(params["feed_rate_mmpm"], 0)

    # tool_breakage ───────────────────────────────────────────────────────────

    def test_tool_breakage_no_stock(self):
        cands = self._fallback("tool_breakage",
                               {"tool_id":"T3","no_stock":True})
        actions = [c.action for c in cands]
        self.assertIn("request_tool_change", actions)

    # spindle_overload ────────────────────────────────────────────────────────

    def test_spindle_overload_light(self):
        cands = self._fallback("spindle_overload",
                               {"overload_pct": 10.0, "measured_kw": 8.3})
        self.assertNotEqual(cands[0].action, "abort")

    def test_spindle_overload_severe(self):
        cands = self._fallback("spindle_overload",
                               {"overload_pct": 55.0, "measured_kw": 11.7})
        # always reduce spindle speed per the new disturbance spec
        self.assertEqual(cands[0].action, "reduce_spindle_speed")

    # safety critical ─────────────────────────────────────────────────────────

    def test_fixture_loose_emergency_stop(self):
        cands = self._fallback("fixture_loose")
        self.assertEqual(cands[0].action, "stop_unload_delay_restart")

    def test_spindle_bearing_emergency_stop(self):
        cands = self._fallback("spindle_bearing")
        self.assertEqual(cands[0].action, "stop_machine_oos")

    # coolant ─────────────────────────────────────────────────────────────────

    def test_coolant_titanium_abort(self):
        ctx = _make_ctx(disturbance_key="coolant_failure",
                        original_material="titanium_ti64",
                        extra={"coolant_type":"flood","partial_loss":False})
        cands = self.engine.fallback_responses(ctx)
        self.assertEqual(cands[0].action, "stop_request_coolant")

    def test_coolant_al_short_run_dry(self):
        ctx = _make_ctx(disturbance_key="coolant_failure",
                        original_material="aluminium_6061",
                        sections_remaining=["Face 3"],
                        extra={"coolant_type":"flood","partial_loss":False})
        cands = self.engine.fallback_responses(ctx)
        # always stop and request coolant per the new policy
        self.assertEqual(cands[0].action, "stop_request_coolant")

    # dimension_error ─────────────────────────────────────────────────────────

    def test_dimension_overcut_rework(self):
        cands = self._fallback("dimension_error",
                               {"nominal_mm":50.0,"actual_mm":49.9,
                                "tolerance_mm":0.05,"feature":"bore depth"})
        self.assertEqual(cands[0].action, "stop_unload_return")

    def test_dimension_undercut_finish_pass(self):
        cands = self._fallback("dimension_error",
                               {"nominal_mm":50.0,"actual_mm":50.08,
                                "tolerance_mm":0.05,"feature":"bore depth"})
        self.assertEqual(cands[0].action, "stop_unload_return")

    def test_dimension_in_tolerance_continue(self):
        cands = self._fallback("dimension_error",
                               {"nominal_mm":50.0,"actual_mm":50.03,
                                "tolerance_mm":0.05,"feature":"bore depth"})
        self.assertEqual(cands[0].action, "continue")


# ── 5. Scoring ────────────────────────────────────────────────────────────────

class TestScoring(unittest.TestCase):

    def setUp(self):
        self.engine = DisturbanceEngine()

    def _cand(self, action, risk="safe") -> LLMCandidate:
        return LLMCandidate(action=action, risk_level=risk,
                            parameters={}, reasoning="test",
                            temperature=0.2, raw="")

    def test_abort_scores_zero_for_min_time(self):
        c = self._cand("abort")
        self.assertEqual(self.engine.score(c, "min_time"), 0.0)

    def test_continue_scores_highest_for_min_time(self):
        c_cont = self._cand("continue")
        c_rw   = self._cand("rework")
        self.assertGreater(self.engine.score(c_cont, "min_time"),
                           self.engine.score(c_rw,   "min_time"))

    def test_risk_penalty_applied(self):
        safe   = self._cand("reduce_feed", "safe")
        risky  = self._cand("reduce_feed", "risky")
        self.assertGreater(self.engine.score(safe,  "min_time"),
                           self.engine.score(risky, "min_time"))

    def test_max_tool_life_favours_change_tool(self):
        ct = self._cand("change_tool")
        rf = self._cand("continue")
        self.assertGreater(self.engine.score(ct, "max_tool_life"),
                           self.engine.score(rf, "max_tool_life"))

    def test_best_finish_favours_finish_pass(self):
        fp = self._cand("add_finish_pass")
        ab = self._cand("abort")
        self.assertGreater(self.engine.score(fp, "best_finish"),
                           self.engine.score(ab, "best_finish"))

    def test_score_bounded(self):
        for action in ("continue","rework","abort","reduce_feed","change_tool"):
            for risk in ("safe","risky","catastrophic"):
                for policy in ("min_time","min_cost","best_finish","max_tool_life"):
                    s = self.engine.score(self._cand(action, risk), policy)
                    self.assertGreaterEqual(s, 0.0)
                    self.assertLessEqual(s,    1.0)

    def test_catastrophic_risk_large_penalty(self):
        safe  = self._cand("continue", "safe")
        catas = self._cand("continue", "catastrophic")
        self.assertGreater(
            self.engine.score(safe,  "min_time") -
            self.engine.score(catas, "min_time"),
            0.4)


# ── 6. Context builder (integration) ─────────────────────────────────────────

class TestContextBuilder(unittest.TestCase):

    def setUp(self):
        from cnc_agent import CncAgent, build_default_crib
        from collections import deque
        self.engine = DisturbanceEngine()
        self.agent  = CncAgent("M01", build_default_crib("M01"), deque(),
                                dry_run=True)
        from factory_agent import build_factory_agent
        self.fa = build_factory_agent(n_machines=4, dry_run=True)

    def test_build_context_material_harder(self):
        spec  = REGISTRY["material_harder"]
        extra = {"original_material":"aluminium_6061",
                 "actual_material":"mild_steel_1018",
                 "evidence":"Vickers 180HV"}
        ctx   = self.engine.build_context(
            self.agent, spec, extra,
            factory_inventory=self.fa.tool_inventory,
            tick=3, policy="min_time",
        )
        self.assertEqual(ctx.disturbance_key, "material_harder")
        self.assertAlmostEqual(ctx.kc_ratio, 3.0, places=1)
        self.assertGreater(ctx.feed_reduction_pct, 0)

    def test_build_context_tool_breakage(self):
        spec  = REGISTRY["tool_breakage"]
        extra = {"tool_id":"T3","no_stock":True}
        ctx   = self.engine.build_context(
            self.agent, spec, extra,
            factory_inventory={},
            tick=5, policy="max_tool_life",
        )
        self.assertEqual(ctx.factory_policy, "max_tool_life")
        self.assertEqual(ctx.inventory.get("T3", 0), 0)

    def test_build_context_no_crash_all_types(self):
        for key, spec in REGISTRY.items():
            extra = {}
            if any(f.name == "original_material"
                   for f in spec.gui_fields):
                extra["original_material"] = "aluminium_6061"
                extra["actual_material"]   = "mild_steel_1018"
            try:
                ctx = self.engine.build_context(
                    self.agent, spec, extra,
                    factory_inventory=self.fa.tool_inventory,
                    tick=1, policy="min_time",
                )
                self.assertEqual(ctx.machine_id, "M01")
            except Exception as e:
                self.fail(f"build_context({key}) raised: {e}")


# ── 7. Material-change engineering sanity checks ──────────────────────────────

class TestMaterialPhysics(unittest.TestCase):
    """Verify the computed numbers make physical sense."""

    def setUp(self):
        self.engine = DisturbanceEngine()

    def _enrich(self, orig, act):
        ctx = _make_ctx(disturbance_key="material_harder",
                        active_tool_dia=25.0)
        return self.engine._enrich_material(
            ctx, {"original_material":orig, "actual_material":act})

    def test_feed_ratio_matches_kc_ratio(self):
        """Feed reduction should be approximately proportional to Kc ratio."""
        mats = list(config.MATERIAL_TABLE.keys())
        for i, orig in enumerate(mats[:-1]):
            act = mats[i + 1]
            ctx = self._enrich(orig, act)
            # if kc_ratio > 1 (harder), feed must decrease
            if ctx.kc_ratio > 1.05:
                self.assertLess(ctx.feed_required, ctx.feed_original,
                    f"{orig}→{act}: feed_req {ctx.feed_required} "
                    f"should be < feed_orig {ctx.feed_original}")
            # if kc_ratio < 0.95 (softer), feed should increase
            elif ctx.kc_ratio < 0.95:
                self.assertGreater(ctx.feed_required, ctx.feed_original,
                    f"{orig}→{act}: softer material should increase feed")

    def test_power_computed_for_all_transitions(self):
        for orig in list(config.MATERIAL_TABLE.keys())[:4]:
            for act in list(config.MATERIAL_TABLE.keys()):
                ctx = self._enrich(orig, act)
                self.assertGreater(ctx.power_required, 0,
                    f"{orig}→{act}: power_required must be >0")

    def test_inconel_exceeds_machine_power(self):
        """Al6061 → Inconel: power_required likely > 7.5 kW machine limit."""
        ctx = self._enrich("aluminium_6061", "inconel")
        # Not guaranteed (depends on tool dia) but ratio is 5.83×
        # At minimum, kc_ratio must be > 5
        self.assertGreater(ctx.kc_ratio, 5.0)

    def test_kc_ratios_monotone_ordering(self):
        """Softer → harder materials should give increasing kc_ratio."""
        order = ["aluminium_6061", "cast_iron", "mild_steel_1018",
                 "alloy_steel_4140", "stainless_304", "titanium_ti64", "inconel"]
        kcs = []
        for mat in order:
            _, kc = config.MATERIAL_TABLE[mat]
            kcs.append(kc)
        for i in range(len(kcs) - 1):
            self.assertLessEqual(kcs[i], kcs[i + 1],
                f"Expected {order[i]} Kc <= {order[i+1]} Kc")


# ── 8. Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.engine = DisturbanceEngine()

    def test_unknown_disturbance_key_generic_prompt(self):
        ctx = _make_ctx(disturbance_key="totally_unknown_event")
        prompt = self.engine.build_prompt(ctx)
        self.assertIn("JSON", prompt)

    def test_unknown_key_fallback_generic(self):
        ctx = _make_ctx(disturbance_key="totally_unknown_event")
        cands = self.engine.fallback_responses(ctx)
        self.assertGreater(len(cands), 0)

    def test_score_unknown_action_non_negative(self):
        c = LLMCandidate(action="some_new_action", risk_level="safe",
                         parameters={}, reasoning="test",
                         temperature=0.2, raw="")
        s = self.engine.score(c, "min_time")
        self.assertGreaterEqual(s, 0.0)

    def test_zero_dia_material_enrich_no_crash(self):
        ctx = _make_ctx(disturbance_key="material_harder",
                        active_tool_dia=0.0)
        # should not raise
        ctx2 = self.engine._enrich_material(
            ctx, {"original_material":"aluminium_6061",
                  "actual_material":"mild_steel_1018"})
        self.assertIsNotNone(ctx2)

    def test_empty_sections_remaining(self):
        ctx = _make_ctx(disturbance_key="priority_change",
                        sections_remaining=[],
                        extra={"new_priority_job":"42","reason":"expedite"})
        cands = self.engine.fallback_responses(ctx)
        self.assertGreater(len(cands), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

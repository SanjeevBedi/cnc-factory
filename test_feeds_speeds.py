"""
test_feeds_speeds.py — Phase 2 tests for feeds_speeds_engine.py

Test suite
----------
1.  test_sfm_to_mpm_conversion         — unit conversion constant
2.  test_rpm_formula_si                — N = 1000·Vc/(π·D) with known values
3.  test_rpm_formula_imperial_equiv    — cross-check with RPM = SFM×3.82/D_in
4.  test_chip_load_categories          — correct fz range per diameter class
5.  test_feed_rate_formula             — F = N·Z·fz
6.  test_mrr_formula                   — MRR = ae·ap·F
7.  test_physics_force                 — Fc = Kc·fz·ap
8.  test_physics_power                 — P = Fc·Vc/60000
9.  test_physics_torque                — T = Fc·D/2000
10. test_power_check_pass              — small cut within limits
11. test_power_check_fail              — aggressive cut exceeds limit
12. test_all_materials                 — every material in table runs OK
13. test_entry_plunge_fraction         — plunge = 40% of feed
14. test_entry_helix_fraction          — helix  = 80% of feed
15. test_entry_ramp_horizontal         — ramp feed ≈ cos(angle)·feed
16. test_entry_helix_diameter          — helix dia = 1.3×D
17. test_fz_override_warning_low       — fz below minimum triggers warning
18. test_fz_override_warning_high      — fz above maximum triggers warning
19. test_integration_with_phase1       — compute for each top face from seed 0

Run:
    cd "/Users/sbedi/Nextcloud/automatic_to_autonomous/CNC Factory"
    python test_feeds_speeds.py
"""

from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from feeds_speeds_engine import (
    compute,
    compute_for_face,
    list_materials,
    material_info,
    _sfm_to_mpm,
    _rpm,
    _chip_load,
    FeedsSpeedsResult,
)


class TestFeedsSpeedsEngine(unittest.TestCase):

    # ── 1. Unit conversion ────────────────────────────────────────────────────
    def test_sfm_to_mpm_conversion(self):
        """1 SFM = 0.3048 m/min exactly."""
        self.assertAlmostEqual(_sfm_to_mpm(1.0), 0.3048, places=6)
        self.assertAlmostEqual(_sfm_to_mpm(1000.0), 304.8, places=4)

    # ── 2. RPM formula (SI) ───────────────────────────────────────────────────
    def test_rpm_formula_si(self):
        """
        N = 1000·Vc / (π·D)
        Example: Vc = 304.8 m/min (= 1000 SFM), D = 12 mm
        N = 1000 × 304.8 / (π × 12) ≈ 8085 RPM
        """
        vc = _sfm_to_mpm(1000)   # 304.8 m/min
        d  = 12.0
        n  = _rpm(vc, d)
        expected = 1000 * 304.8 / (math.pi * 12)
        self.assertAlmostEqual(n, expected, delta=1.0)
        # Sanity: should be ~ 8085
        self.assertGreater(n, 8000)
        self.assertLess(n, 8200)

    # ── 3. Imperial cross-check ───────────────────────────────────────────────
    def test_rpm_formula_imperial_equiv(self):
        """
        RPM = SFM × 3.82 / D_inches  must agree with SI formula within 0.2%.
        """
        sfm = 1000
        d_mm = 12.0
        d_in = d_mm / 25.4

        n_imperial = sfm * 3.82 / d_in
        n_si       = _rpm(_sfm_to_mpm(sfm), d_mm)

        rel_diff = abs(n_imperial - n_si) / n_si
        self.assertLess(rel_diff, 0.002,
            f"Imperial ({n_imperial:.1f}) vs SI ({n_si:.1f}) differ by "
            f"{rel_diff*100:.2f}% — should be < 0.2%")

    # ── 4. Chip load categories ───────────────────────────────────────────────
    def test_chip_load_categories(self):
        fz_min, fz_max, fz_mid = _chip_load(6.0)    # small
        self.assertAlmostEqual(fz_min, 0.025, places=4)
        self.assertAlmostEqual(fz_max, 0.075, places=4)
        self.assertAlmostEqual(fz_mid, 0.050, places=4)

        fz_min, fz_max, fz_mid = _chip_load(12.0)   # medium boundary
        self.assertAlmostEqual(fz_min, 0.050, places=4)
        self.assertAlmostEqual(fz_max, 0.150, places=4)

        fz_min, fz_max, fz_mid = _chip_load(25.0)   # large boundary
        self.assertAlmostEqual(fz_min, 0.100, places=4)
        self.assertAlmostEqual(fz_max, 0.300, places=4)

    # ── 5. Feed rate formula ──────────────────────────────────────────────────
    def test_feed_rate_formula(self):
        """
        F = N·Z·fz
        Given N=5000, Z=4, fz=0.1 → F = 2000 mm/min exactly.
        """
        result = compute(
            material          = "aluminium_6061",
            tool_diameter_mm  = 12.0,
            tool_flutes       = 4,
            fz_override       = 0.1,
            sfm_override      = None,
        )
        n  = result.rpm
        fz = result.chip_load_mm
        z  = result.tool.flutes
        expected_feed = n * z * fz
        self.assertAlmostEqual(result.feed_rate_mmpm, expected_feed, delta=0.01)

    # ── 6. MRR formula ────────────────────────────────────────────────────────
    def test_mrr_formula(self):
        """MRR = ae·ap·F"""
        result = compute(
            "aluminium_6061", 12.0, 4,
            axial_depth_mm=6.0, radial_depth_mm=4.8, fz_override=0.1
        )
        expected = result.cut.radial_depth_mm * result.cut.axial_depth_mm \
                   * result.feed_rate_mmpm
        self.assertAlmostEqual(result.mrr_mm3pm, expected, delta=0.1)

    # ── 7. Cutting force ──────────────────────────────────────────────────────
    def test_physics_force(self):
        """
        Fc = Kc·fz·ap
        Al 6061: Kc=600 MPa, fz=0.1 mm, ap=6 mm → Fc = 360 N
        """
        result = compute(
            "aluminium_6061", 12.0, 4,
            axial_depth_mm=6.0, radial_depth_mm=4.8,
            fz_override=0.1
        )
        self.assertAlmostEqual(result.chip_area_mm2, 0.1 * 6.0, places=6)
        self.assertAlmostEqual(result.cutting_force_n, 600 * 0.1 * 6.0, delta=0.1)

    # ── 8. Power formula ──────────────────────────────────────────────────────
    def test_physics_power(self):
        """P = Fc·Vc / 60 000  (kW)"""
        result = compute(
            "aluminium_6061", 12.0, 4,
            axial_depth_mm=6.0, radial_depth_mm=4.8,
            fz_override=0.1, sfm_override=1000
        )
        expected_power = result.cutting_force_n * result.cutting_speed_mpm / 60_000.0
        self.assertAlmostEqual(result.power_kw, expected_power, places=6)

    # ── 9. Torque formula ─────────────────────────────────────────────────────
    def test_physics_torque(self):
        """T = Fc·D / 2 000  (Nm)"""
        result = compute(
            "aluminium_6061", 12.0, 4,
            axial_depth_mm=6.0, radial_depth_mm=4.8,
            fz_override=0.1
        )
        expected_torque = result.cutting_force_n * result.tool.diameter_mm / 2_000.0
        self.assertAlmostEqual(result.torque_nm, expected_torque, places=6)

    # ── 10. Power check — pass ────────────────────────────────────────────────
    def test_power_check_pass(self):
        """Light aluminium cut — should be well within 7.5 kW."""
        result = compute(
            "aluminium_6061", 12.0, 4,
            axial_depth_mm=3.0, radial_depth_mm=2.0, fz_override=0.05
        )
        self.assertTrue(result.power_ok,
            f"Expected power OK, got {result.power_kw:.3f} kW")
        self.assertTrue(result.is_feasible)

    # ── 11. Power check — fail ────────────────────────────────────────────────
    def test_power_check_fail(self):
        """
        Force a power overrun:
        Inconel, large tool, aggressive depths, high SFM.
        Kc=3500, fz=0.3 (max), ap=25mm → Fc = 3500×0.3×25 = 26 250 N
        Vc at SFM=150 → 45.7 m/min
        P = 26250 × 45.7 / 60000 ≈ 20.0 kW  >>  7.5 kW limit
        """
        result = compute(
            "inconel", 25.0, 4,
            axial_depth_mm=25.0, radial_depth_mm=12.5,
            fz_override=0.3, sfm_override=150
        )
        self.assertFalse(result.power_ok,
            f"Expected power FAIL, got {result.power_kw:.3f} kW — "
            "limit is 7.5 kW")
        self.assertFalse(result.is_feasible)
        self.assertTrue(any("Power" in w for w in result.warnings),
            "Expected a power-warning in the result.")

    # ── 12. All materials ─────────────────────────────────────────────────────
    def test_all_materials(self):
        """Every material in the table must compute without raising an exception."""
        for mat in list_materials():
            with self.subTest(material=mat):
                result = compute(mat, 12.0, 4)
                self.assertGreater(result.rpm, 0)
                self.assertGreater(result.feed_rate_mmpm, 0)
                self.assertGreater(result.cutting_force_n, 0)
                self.assertGreater(result.power_kw, 0)

    # ── 13. Plunge feed fraction ──────────────────────────────────────────────
    def test_entry_plunge_fraction(self):
        """Plunge feed = 40% of normal feed."""
        result = compute("aluminium_6061", 12.0, 4)
        self.assertAlmostEqual(
            result.plunge_feed_mmpm,
            result.feed_rate_mmpm * 0.40,
            delta=0.01
        )

    # ── 14. Helix feed fraction ───────────────────────────────────────────────
    def test_entry_helix_fraction(self):
        """Helix feed = 80% of normal feed."""
        result = compute("aluminium_6061", 12.0, 4)
        self.assertAlmostEqual(
            result.helix_feed_mmpm,
            result.feed_rate_mmpm * 0.80,
            delta=0.01
        )

    # ── 15. Ramp feed is horizontal component ────────────────────────────────
    def test_entry_ramp_horizontal(self):
        """Ramp feed = feed × cos(angle) at 2° default."""
        result = compute("aluminium_6061", 12.0, 4, ramp_angle_deg=2.0)
        expected = result.feed_rate_mmpm * math.cos(math.radians(2.0))
        self.assertAlmostEqual(result.ramp_feed_mmpm, expected, delta=0.01)

    # ── 16. Helix diameter ────────────────────────────────────────────────────
    def test_entry_helix_diameter(self):
        """Helix diameter = 1.3 × tool diameter."""
        result = compute("aluminium_6061", 12.0, 4)
        self.assertAlmostEqual(result.helix_diameter_mm, 12.0 * 1.3, places=4)

    # ── 17. Low fz warning ────────────────────────────────────────────────────
    def test_fz_override_warning_low(self):
        """fz below minimum should trigger a dust-chips warning."""
        result = compute("aluminium_6061", 12.0, 4, fz_override=0.001)
        self.assertTrue(any("dust" in w.lower() for w in result.warnings),
            f"Expected dust-chips warning, got: {result.warnings}")

    # ── 18. High fz warning ───────────────────────────────────────────────────
    def test_fz_override_warning_high(self):
        """fz above maximum should trigger a chatter/breakage warning."""
        result = compute("aluminium_6061", 12.0, 4, fz_override=1.0)
        self.assertTrue(
            any("chatter" in w.lower() or "breakage" in w.lower()
                for w in result.warnings),
            f"Expected chatter/breakage warning, got: {result.warnings}"
        )

    # ── 19. Integration with Phase 1 ──────────────────────────────────────────
    def test_integration_with_phase1(self):
        """
        Run feeds & speeds for every top face from seed 0.
        Assert: each face produces a valid FeedsSpeedsResult.
        The two largest-area faces must be feasible (not overloaded).
        """
        npy_path = os.path.join(config.SOLID_OUTPUT_DIR, "solid_faces_seed_0.npy")
        if not os.path.exists(npy_path):
            self.skipTest("solid_faces_seed_0.npy not found")

        import cnc_solid_bridge as bridge
        from feature_extractor import extract_features

        faces  = bridge.load_face_polygons(0)
        result = extract_features(faces, seed=0, tool_radius=6.0)

        self.assertGreater(result.n_top_faces, 0)

        # Sort top faces by area descending
        sorted_faces = sorted(result.top_faces,
                              key=lambda f: f.face_area, reverse=True)

        all_results = []
        for tf in sorted_faces:
            r = compute_for_face(tf, material="aluminium_6061",
                                 tool_diameter_mm=12.0, tool_flutes=4)
            self.assertGreater(r.rpm, 0)
            self.assertGreater(r.feed_rate_mmpm, 0)
            self.assertGreater(r.cutting_force_n, 0)
            all_results.append(r)

        # Top 2 faces should be feasible with a 12mm tool in aluminium
        for r in all_results[:2]:
            self.assertTrue(r.is_feasible,
                f"Face should be feasible but got: {r.summary()}")

        print(f"\n[test_integration_phase1] "
              f"{len(all_results)} faces processed")
        print(all_results[0].summary())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestFeedsSpeedsEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

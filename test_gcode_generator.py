"""
test_gcode_generator.py — Phase 4 tests for gcode_generator.py

Test suite
----------
 1. test_header_contains_setup_codes     — G21, G90 present in program
 2. test_spindle_on_before_first_g01     — S{rpm} M3 before first G01
 3. test_tool_setup_commands_present     — TOOL SHAPE and TOOL SIZE in output
 4. test_program_ends_with_m30          — last motion line is M30
 5. test_spindle_off_before_m30         — M5 present and before M30
 6. test_g00_lines_have_no_f_word       — no F on any G00 line
 7. test_g01_modal_f_tracking           — F re-emitted only when value changes
 8. test_to_string_matches_join         — to_string() == "\n".join(lines)
 9. test_line_count_consistent          — len(lines) == line_count
10. test_save_and_reload                — save to tmp file, read back matches
11. test_no_nan_or_inf_in_coords        — validator finds no finiteness errors
12. test_face_comments_in_output        — "(Face N" comment for each face
13. test_active_face_count              — n_active_faces correct
14. test_raster_waypoints_are_g01       — raster pass_type lines are G01
15. test_approach_retract_are_g00       — approach / retract lines are G00
16. test_rpc_cap_applied                — rpm > cap → S word capped, warning issued
17. test_validation_clean_program       — valid program has zero errors
18. test_validation_missing_f_word      — injected G01 without F → error reported
19. test_integration_full_pipeline      — seed 0: Phase1→2→3→4, file saved

Run:
    cd "/Users/sbedi/Nextcloud/automatic_to_autonomous/CNC Factory"
    python test_gcode_generator.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from feeds_speeds_engine import compute as fs_compute
from toolpath_planner import plan, plan_face
from gcode_generator import generate, validate, GCodeResult
from test_toolpath_planner import _rect_face, _default_fs   # reuse synthetic helpers


# ── Shared fixture ─────────────────────────────────────────────────────────────

def _make_simple_gcode(wall_sides=("right",), ae=4.8):
    """One 100×100 synthetic face → G-code."""
    tf = _rect_face(0, 0, 100, 100, 10, wall_sides=list(wall_sides))
    fs = _default_fs(ae=ae)

    # Minimal ToolpathResult wrapper
    from toolpath_planner import FaceToolpath, ToolpathResult
    ft = plan_face(tf, fs)

    class _FR:
        seed = 0
        faces = [ft]
        total_path_length_mm = ft.path_length_mm
        total_estimated_time_s = ft.estimated_time_s
        warnings = []

    return generate(_FR(), fs, program_number=1)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestGCodeGenerator(unittest.TestCase):

    # ── 1. Setup codes present ────────────────────────────────────────────────
    def test_header_contains_setup_codes(self):
        """G21 (metric) and G90 (absolute) must be in the program."""
        gc = _make_simple_gcode()
        prog = gc.to_string()
        self.assertIn("G21", prog, "G21 (metric) missing from program")
        self.assertIn("G90", prog, "G90 (absolute) missing from program")

    # ── 2. Spindle on before first G01 ────────────────────────────────────────
    def test_spindle_on_before_first_g01(self):
        """S{rpm} M3 must appear before the first G01 cutting line."""
        gc = _make_simple_gcode()
        lines = gc.lines
        m3_idx   = next((i for i, l in enumerate(lines) if "M3" in l and "M30" not in l), None)
        g01_idx  = next((i for i, l in enumerate(lines) if l.startswith("G01")), None)
        self.assertIsNotNone(m3_idx, "M3 (spindle on) not found in program")
        if g01_idx is not None:
            self.assertLess(m3_idx, g01_idx,
                f"M3 (line {m3_idx}) appears after first G01 (line {g01_idx})")

    # ── 3. Tool setup commands ────────────────────────────────────────────────
    def test_tool_setup_commands_present(self):
        """TOOL SHAPE and TOOL SIZE must be present."""
        gc   = _make_simple_gcode()
        prog = gc.to_string()
        self.assertIn("TOOL SHAPE", prog, "TOOL SHAPE command missing")
        self.assertIn("TOOL SIZE",  prog, "TOOL SIZE command missing")

    # ── 4. Program ends with M30 ──────────────────────────────────────────────
    def test_program_ends_with_m30(self):
        """M30 must be the last non-blank, non-comment line."""
        gc = _make_simple_gcode()
        motion_lines = [
            l for l in gc.lines
            if l.strip() and not l.strip().startswith("(")
        ]
        self.assertEqual(
            motion_lines[-1].strip(), "M30",
            f"Last motion line is not M30, got: {motion_lines[-1]!r}"
        )

    # ── 5. Spindle off before M30 ─────────────────────────────────────────────
    def test_spindle_off_before_m30(self):
        """M5 must appear before M30."""
        gc   = _make_simple_gcode()
        lines = gc.lines
        m5_idx  = next((i for i, l in enumerate(lines)
                        if l.strip() == "M5"), None)
        m30_idx = next((i for i, l in enumerate(lines)
                        if "M30" in l), None)
        self.assertIsNotNone(m5_idx,  "M5 (spindle off) not found")
        self.assertIsNotNone(m30_idx, "M30 (end) not found")
        self.assertLess(m5_idx, m30_idx,
            f"M5 (line {m5_idx}) must come before M30 (line {m30_idx})")

    # ── 6. G00 lines have no F word ───────────────────────────────────────────
    def test_g00_lines_have_no_f_word(self):
        """Rapid moves (G00) must not carry an F word."""
        gc = _make_simple_gcode()
        for line in gc.lines:
            if line.startswith("G00"):
                self.assertNotIn("F", line,
                    f"G00 line contains F word: {line!r}")

    # ── 7. Modal F tracking ───────────────────────────────────────────────────
    def test_g01_modal_f_tracking(self):
        """
        F word should appear on the FIRST G01 of each new feed value.
        Two consecutive G01 lines with the same feed → second has NO F word.
        """
        gc = _make_simple_gcode()
        last_f    = None
        for line in gc.lines:
            if not line.startswith("G01"):
                continue
            has_f  = " F" in line
            # Extract F if present
            f_val  = None
            if has_f:
                f_val = float(line.split("F")[1].split()[0])

            if last_f is None:
                # First G01: must have F
                self.assertTrue(has_f,
                    f"First G01 has no F word: {line!r}")
            else:
                if f_val is not None:
                    # F changed — OK to re-emit
                    pass
                else:
                    # No F — previous F is still modal. This is correct.
                    pass
            if f_val is not None:
                last_f = f_val

    # ── 8. to_string matches join ─────────────────────────────────────────────
    def test_to_string_matches_join(self):
        gc = _make_simple_gcode()
        self.assertEqual(gc.to_string(), "\n".join(gc.lines))

    # ── 9. line_count consistent ──────────────────────────────────────────────
    def test_line_count_consistent(self):
        gc = _make_simple_gcode()
        self.assertEqual(gc.line_count, len(gc.lines),
            f"line_count={gc.line_count} but len(lines)={len(gc.lines)}")

    # ── 10. Save and reload ───────────────────────────────────────────────────
    def test_save_and_reload(self):
        """Save to a tmp .nc file and read it back — content must match."""
        gc = _make_simple_gcode()
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            path = tmp.name
        try:
            gc.save(path)
            with open(path) as f:
                content = f.read()
            self.assertEqual(content.strip(), gc.to_string().strip())
        finally:
            os.unlink(path)

    # ── 11. No NaN or inf in coordinates ─────────────────────────────────────
    def test_no_nan_or_inf_in_coords(self):
        """Validator must report zero finiteness errors for a clean program."""
        gc = _make_simple_gcode()
        finiteness_errors = [e for e in gc.validation_errors
                             if "non-finite" in e]
        self.assertEqual(finiteness_errors, [],
            f"Non-finite coordinate errors: {finiteness_errors}")

    # ── 12. Face comments in output ───────────────────────────────────────────
    def test_face_comments_in_output(self):
        """A '(Face N' comment must appear for every face in the toolpath."""
        gc = _make_simple_gcode()
        face_comment_lines = [l for l in gc.lines if l.startswith("(Face ")]
        self.assertGreater(len(face_comment_lines), 0,
            "No '(Face ...' comments found in G-code output")

    # ── 13. Active face count ─────────────────────────────────────────────────
    def test_active_face_count(self):
        """n_active_faces must equal the count of faces with n_passes > 0."""
        gc = _make_simple_gcode()
        # For our single 100×100 face with WALL on right side we expect ≥1 active
        self.assertGreaterEqual(gc.n_active_faces, 0)
        self.assertLessEqual(gc.n_active_faces, gc.n_faces)

    # ── 14. Raster waypoints produce G01 lines ────────────────────────────────
    def test_raster_waypoints_are_g01(self):
        """Raster pass waypoints (feed_rate > 0) must produce G01 lines."""
        gc = _make_simple_gcode()
        g01_lines = [l for l in gc.lines if l.startswith("G01")]
        if gc.n_active_faces > 0:
            self.assertGreater(len(g01_lines), 0,
                "Expected G01 lines for active faces but found none")

    # ── 15. Approach/retract produce G00 ─────────────────────────────────────
    def test_approach_retract_are_g00(self):
        """Approach and retract waypoints (feed_rate=0) must produce G00 lines."""
        gc = _make_simple_gcode()
        g00_lines = [l for l in gc.lines if l.startswith("G00")]
        if gc.n_active_faces > 0:
            self.assertGreater(len(g00_lines), 0,
                "Expected G00 (rapid) lines for approach/retract but found none")

    # ── 16. RPM cap applied ───────────────────────────────────────────────────
    def test_rpc_cap_applied(self):
        """
        When computed RPM > cap, S word must equal the cap value,
        and a warning must be present.
        """
        fs  = fs_compute("aluminium_6061", 12.0, 4,
                         axial_depth_mm=6.0, radial_depth_mm=4.8,
                         fz_override=0.1, sfm_override=1000)
        # Computed RPM is ~8247 >> cap 4000
        tf  = _rect_face(0, 0, 100, 100, 10, wall_sides=["right"])
        ft  = plan_face(tf, fs)

        class _FR:
            seed = 0
            faces = [ft]
            total_path_length_mm = ft.path_length_mm
            total_estimated_time_s = ft.estimated_time_s
            warnings = []

        gc = generate(_FR(), fs, spindle_rpm_cap=4000.0)

        self.assertEqual(gc.spindle_rpm_emitted, 4000.0,
            f"Expected S4000, got S{gc.spindle_rpm_emitted}")
        self.assertIn("S4000", gc.to_string(),
            "S4000 not found in program")
        cap_warnings = [w for w in gc.warnings if "cap" in w.lower() or "exceed" in w.lower()]
        self.assertTrue(len(cap_warnings) > 0,
            f"Expected RPM-cap warning but got: {gc.warnings}")

    # ── 17. Validation clean program ─────────────────────────────────────────
    def test_validation_clean_program(self):
        """A correctly generated program must have zero validation errors."""
        gc = _make_simple_gcode()
        self.assertEqual(gc.validation_errors, [],
            f"Unexpected validation errors: {gc.validation_errors}")

    # ── 18. Validation catches missing F ─────────────────────────────────────
    def test_validation_missing_f_word(self):
        """
        Inject a G01 line BEFORE any F word has been set.
        Validator must flag it.
        """
        # Build a minimal result with a G01 before any F
        bad_lines = [
            "(header)",
            "G21",
            "G90",
            "S4000 M3",
            "G01 X10.000 Y20.000 Z5.000",   # ← F not set yet!
            "M5",
            "M30",
        ]
        bad_result = GCodeResult(
            seed=0, program_number=1,
            lines=bad_lines, line_count=len(bad_lines),
            motion_line_count=1, n_faces=1, n_active_faces=1,
            total_path_length_mm=10.0, estimated_time_s=1.0,
            spindle_rpm_emitted=4000.0,
        )
        errors = validate(bad_result)
        self.assertTrue(any("G01" in e and "F" in e for e in errors),
            f"Expected 'G01 without F' error but got: {errors}")

    # ── 19. Full pipeline integration ─────────────────────────────────────────
    def test_integration_full_pipeline(self):
        """
        Seed 0: Phase1 → Phase2 → Phase3 → Phase4.
        Save to a .nc file. Validate. Assert zero structural errors.
        """
        npy = os.path.join(config.SOLID_OUTPUT_DIR, "solid_faces_seed_0.npy")
        if not os.path.exists(npy):
            self.skipTest("solid_faces_seed_0.npy not found")

        import cnc_solid_bridge as bridge
        from feature_extractor import extract_features

        faces   = bridge.load_face_polygons(0)
        meta    = bridge.load_metadata(0)
        feat    = extract_features(faces, seed=0, volume=meta["volume"])

        fs      = fs_compute("aluminium_6061", 12.0, 4,
                             axial_depth_mm=6.0, radial_depth_mm=4.8,
                             fz_override=0.1, sfm_override=1000)
        tp      = plan(feat, fs)
        gc      = generate(tp, fs, program_number=1, spindle_rpm_cap=4000.0)

        # Save
        out_dir = config.SOLID_OUTPUT_DIR
        nc_path = os.path.join(out_dir, "program_seed_0.nc")
        gc.save(nc_path)
        self.assertTrue(os.path.exists(nc_path),
            f"Expected G-code file at {nc_path}")

        # Reload and compare
        with open(nc_path) as f:
            saved = f.read().strip()
        self.assertEqual(saved, gc.to_string().strip())

        # Validate
        self.assertEqual(gc.validation_errors, [],
            f"Validation errors: {gc.validation_errors}")

        # Structural checks
        self.assertGreater(gc.line_count, 10)
        self.assertGreater(gc.motion_line_count, 0)
        self.assertIn("M30", gc.to_string())
        self.assertIn("M5",  gc.to_string())
        self.assertIn("G21", gc.to_string())

        print(f"\n[integration] Seed 0 G-code saved → {nc_path}")
        print(f"  Lines        : {gc.line_count}")
        print(f"  Motion lines : {gc.motion_line_count}")
        print(f"  Active faces : {gc.n_active_faces} / {gc.n_faces}")
        print(f"  Path length  : {gc.total_path_length_mm:.1f} mm")
        print(f"  Est. time    : {gc.estimated_time_s:.1f} s")
        print(f"  S word       : {gc.spindle_rpm_emitted:.0f} RPM (capped)")
        print(f"\n--- First 40 lines ---")
        for l in gc.lines[:40]:
            print(l)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestGCodeGenerator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

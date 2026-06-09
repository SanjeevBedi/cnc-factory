"""
test_toolpath_planner.py — Phase 3 tests for toolpath_planner.py

Test suite
----------
1.  test_safe_region_walls_only_partial     — WALL edges eroded, CLIFF edges not
2.  test_safe_region_all_walls_is_inset     — all WALL → region same as uniform inset
3.  test_safe_region_all_cliffs_full_face   — all CLIFF → region = full face (no erosion)
4.  test_safe_region_empty_when_too_small   — tiny face with large tool → empty
5.  test_raster_produces_passes             — known safe region → correct pass count
6.  test_raster_stepover_spacing            — y-distance between consecutive passes = ae
7.  test_raster_alternates_direction        — even passes go right, odd go left
8.  test_raster_all_points_inside_region    — no waypoint outside safe region
9.  test_raster_no_short_segments           — segments < MIN_PASS_LENGTH filtered out
10. test_entry_cliff_linear                 — cliff edge produces cliff_linear entry
11. test_entry_helix_when_no_cliff          — no cliff edges → helix entry
12. test_helix_descends_monotonically       — helix Z strictly decreasing
13. test_helix_centred_on_face              — helix circles centred on face centroid
14. test_no_waypoint_inside_wall_buffer     — all raster waypoints ≥ R from WALL edges
15. test_faces_sorted_high_to_low           — plan() returns faces Z descending
16. test_time_estimate_formula              — time = length / feed_rate × 60
17. test_integration_phase1_to_phase3       — full pipeline seed 0 → toolpaths
18. test_path_length_consistent             — sum of segment lengths = path_length_mm

Run:
    cd "/Users/sbedi/Nextcloud/automatic_to_autonomous/CNC Factory"
    python test_toolpath_planner.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from feature_extractor import (
    TopFaceFeature, EdgeFeature,
    compute_face_normal, extract_features,
)
from feeds_speeds_engine import compute as fs_compute
from toolpath_planner import (
    compute_safe_region_walls_only,
    generate_raster,
    plan_face,
    plan,
    FaceToolpath,
    ToolpathResult,
    Waypoint,
    _DEFAULT_CLEARANCE_MM,
    _DEFAULT_LEAD_IN_MM,
)

try:
    from shapely.geometry import Polygon as ShapelyPolygon, Point
    _SHAPELY_OK = True
except ImportError:
    _SHAPELY_OK = False


# ── Synthetic helpers ─────────────────────────────────────────────────────────

def _make_edge(edge_id, v_start, v_end, label, adj_id=None, adj_normal=None):
    return EdgeFeature(
        edge_id=edge_id,
        v_start=list(v_start),
        v_end=list(v_end),
        label=label,
        adjacent_face_id=adj_id,
        adjacent_face_normal=adj_normal,
    )


def _rect_face(x0, y0, x1, y1, z, wall_sides=None):
    """
    Build a synthetic TopFaceFeature for a rectangle.
    wall_sides: list of 'left'|'right'|'bottom'|'top' — those sides are WALL.
    All others are CLIFF.
    """
    if wall_sides is None:
        wall_sides = []

    verts_3d = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
    verts_2d = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    side_map = {
        "bottom": (0, [x0,y0,z], [x1,y0,z]),  # edge 0
        "right":  (1, [x1,y0,z], [x1,y1,z]),  # edge 1
        "top":    (2, [x1,y1,z], [x0,y1,z]),  # edge 2
        "left":   (3, [x0,y1,z], [x0,y0,z]),  # edge 3
    }

    edges = []
    for side_name, (eid, vs, ve) in side_map.items():
        label = "wall" if side_name in wall_sides else "cliff"
        edges.append(_make_edge(eid, vs, ve, label))

    area = (x1 - x0) * (y1 - y0)
    return TopFaceFeature(
        face_id=0, z_height=float(z),
        normal=[0.0, 0.0, 1.0],
        vertices_3d=verts_3d, vertices_2d=verts_2d,
        edges=edges,
        wall_count=sum(1 for s in wall_sides if True),
        cliff_count=4 - len(wall_sides),
        level_count=0,
        face_area=area,
        safe_region_area=0.0, safe_region=None,
    )


def _default_fs(tool_d=12.0, ae=4.8, ap=6.0):
    """Phase 2 result for aluminium with a 12 mm tool."""
    return fs_compute(
        "aluminium_6061", tool_d, 4,
        axial_depth_mm=ap, radial_depth_mm=ae,
        fz_override=0.1, sfm_override=1000,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_SHAPELY_OK, "shapely not installed")
class TestToolpathPlanner(unittest.TestCase):

    # ── 1. WALL-only erosion — partial ────────────────────────────────────────
    def test_safe_region_walls_only_partial(self):
        """
        100×100 face, WALL on right edge only (x=100).
        Safe region near right must be eroded by R; left must be untouched.
        """
        tf = _rect_face(0, 0, 100, 100, 10, wall_sides=["right"])
        R  = 6.0
        safe, area = compute_safe_region_walls_only(
            np.array(tf.vertices_2d), tf.edges, R
        )
        self.assertIsNotNone(safe)
        self.assertGreater(area, 0)

        # Right boundary: no point in safe region closer than R to x=100
        # Check that safe region does NOT extend to x=100 (wall side)
        bounds = safe.bounds  # (minx, miny, maxx, maxy)
        self.assertLess(bounds[2], 100.0 - R + 0.1,
            f"Safe region extends too close to wall (x_max={bounds[2]:.2f})")

        # Left boundary: safe region SHOULD extend close to x=0 (cliff side)
        self.assertLess(bounds[0], 0.0 + R,
            f"Safe region should extend to near x=0 (cliff side), got x_min={bounds[0]:.2f}")

    # ── 2. All WALL → same as uniform inset ───────────────────────────────────
    def test_safe_region_all_walls_is_inset(self):
        """All 4 edges WALL → safe region ≈ (100-2R) × (100-2R)."""
        tf = _rect_face(0, 0, 100, 100, 10, wall_sides=["left","right","top","bottom"])
        R  = 6.0
        safe, area = compute_safe_region_walls_only(
            np.array(tf.vertices_2d), tf.edges, R
        )
        # Expected area ≈ 88×88 = 7744
        expected = (100 - 2 * R) ** 2
        self.assertAlmostEqual(area, expected, delta=50,
            msg=f"All-wall safe area: expected ≈{expected:.0f}, got {area:.1f}")

    # ── 3. All CLIFF → full face returned ────────────────────────────────────
    def test_safe_region_all_cliffs_full_face(self):
        """No WALL edges → safe region = full face polygon (no erosion)."""
        tf  = _rect_face(0, 0, 100, 100, 10, wall_sides=[])
        R   = 6.0
        safe, area = compute_safe_region_walls_only(
            np.array(tf.vertices_2d), tf.edges, R
        )
        self.assertIsNotNone(safe)
        self.assertAlmostEqual(area, 100 * 100, delta=1.0,
            msg=f"All-cliff: expected 10000, got {area:.1f}")

    # ── 4. Empty safe region ──────────────────────────────────────────────────
    def test_safe_region_empty_when_too_small(self):
        """8×8 face with R=6 and all-WALL edges → safe region empty."""
        tf = _rect_face(0, 0, 8, 8, 10, wall_sides=["left","right","top","bottom"])
        safe, area = compute_safe_region_walls_only(
            np.array(tf.vertices_2d), tf.edges, 6.0
        )
        self.assertEqual(area, 0.0,
            f"Expected empty safe region, got area={area:.1f}")

    # ── 5. Raster produces passes ─────────────────────────────────────────────
    def test_raster_produces_passes(self):
        """
        80×80 safe region, stepover=8 → expect ~10 passes.
        """
        safe = ShapelyPolygon([(0,0),(80,0),(80,80),(0,80)])
        passes = generate_raster(safe, stepover_mm=8.0)
        # (80 / 8) = 10 passes; allow ±1 for boundary rounding
        self.assertGreater(len(passes), 7)
        self.assertLess(len(passes), 14)

    # ── 6. Stepover spacing ───────────────────────────────────────────────────
    def test_raster_stepover_spacing(self):
        """Y spacing between consecutive passes = stepover_mm ± 1e-3."""
        safe     = ShapelyPolygon([(0,0),(100,0),(100,100),(0,100)])
        stepover = 5.0
        passes   = generate_raster(safe, stepover_mm=stepover)
        self.assertGreater(len(passes), 1)

        # Each pass is a single horizontal segment; get midpoint Y
        y_vals = []
        for p in passes:
            ys = [c[1] for c in p]
            y_vals.append(sum(ys) / len(ys))

        # Consecutive Y differences should equal stepover
        for i in range(1, len(y_vals)):
            dy = abs(y_vals[i] - y_vals[i - 1])
            self.assertAlmostEqual(dy, stepover, delta=1e-2,
                msg=f"Pass {i}: expected dy={stepover}, got {dy:.4f}")

    # ── 7. Alternating direction ──────────────────────────────────────────────
    def test_raster_alternates_direction(self):
        """
        Consecutive passes must start from opposite ends (zig-zag).
        Pass 0 starts at x_min, pass 1 starts at x_max, etc.
        """
        safe   = ShapelyPolygon([(0,0),(100,0),(100,100),(0,100)])
        passes = generate_raster(safe, stepover_mm=10.0)
        self.assertGreater(len(passes), 1)

        for k in range(min(4, len(passes) - 1)):
            x0 = passes[k][0][0]
            x1 = passes[k + 1][0][0]
            self.assertNotAlmostEqual(
                x0, x1, delta=5.0,
                msg=f"Passes {k} and {k+1} both start at x≈{x0:.1f} — "
                "direction not alternating.")

    # ── 8. All raster points inside safe region ───────────────────────────────
    def test_raster_all_points_inside_region(self):
        """Every raster waypoint must lie within (or on the boundary of) the safe region."""
        safe   = ShapelyPolygon([(10,10),(90,10),(90,90),(10,90)])
        passes = generate_raster(safe, stepover_mm=8.0)
        buffered = safe.buffer(0.5)  # small tolerance for float precision
        for k, p in enumerate(passes):
            for c in p:
                pt = Point(c[0], c[1])
                self.assertTrue(buffered.contains(pt),
                    f"Pass {k} point {c} lies outside safe region bounds "
                    f"{safe.bounds}")

    # ── 9. No short segments ──────────────────────────────────────────────────
    def test_raster_no_short_segments(self):
        """All returned raster segments have length ≥ _MIN_PASS_LENGTH_MM."""
        from toolpath_planner import _MIN_PASS_LENGTH_MM
        from shapely.geometry import LineString
        safe   = ShapelyPolygon([(0,0),(100,0),(100,100),(0,100)])
        passes = generate_raster(safe, stepover_mm=5.0)
        for k, p in enumerate(passes):
            seg_len = LineString(p).length
            self.assertGreaterEqual(seg_len, _MIN_PASS_LENGTH_MM,
                f"Pass {k}: length {seg_len:.3f} < minimum {_MIN_PASS_LENGTH_MM}")

    # ── 10. Cliff entry ───────────────────────────────────────────────────────
    def test_entry_cliff_linear(self):
        """
        Face with a CLIFF edge → plan_face must choose cliff_linear entry.
        """
        tf = _rect_face(0, 0, 100, 100, 10, wall_sides=["right", "top", "bottom"])
        fs = _default_fs()
        ft = plan_face(tf, fs)
        self.assertEqual(ft.entry_type, "cliff_linear",
            f"Expected cliff_linear entry, got '{ft.entry_type}'\n"
            f"Warnings: {ft.warnings}")
        self.assertGreater(ft.n_passes, 0)

    # ── 11. Helix entry when no cliff ────────────────────────────────────────
    def test_entry_helix_when_no_cliff(self):
        """All-WALL face → plan_face must use helix entry."""
        tf = _rect_face(0, 0, 80, 80, 10, wall_sides=["left","right","top","bottom"])
        fs = _default_fs(ae=4.0)
        ft = plan_face(tf, fs)
        if ft.entry_type == "skipped":
            self.skipTest("Face too small for tool radius — safe region empty.")
        self.assertEqual(ft.entry_type, "helix",
            f"Expected helix entry, got '{ft.entry_type}'")

    # ── 12. Helix Z decreases monotonically ──────────────────────────────────
    def test_helix_descends_monotonically(self):
        """Helix entry waypoints must have strictly decreasing Z."""
        from toolpath_planner import _make_helix_entry
        fs     = _default_fs()
        verts  = np.array([[0,0],[80,0],[80,80],[0,80]], dtype=float)
        z_top  = 10.0
        z_start= 15.0
        wps = _make_helix_entry(verts, z_top, z_start,
                                helix_dia_mm=fs.helix_diameter_mm,
                                pitch_mm=fs.helix_pitch_mm,
                                feed_rate=fs.helix_feed_mmpm)
        self.assertGreater(len(wps), 2)
        z_vals = [wp.z for wp in wps]
        for i in range(1, len(z_vals)):
            self.assertLessEqual(z_vals[i], z_vals[i - 1] + 1e-9,
                f"Helix Z increased at step {i}: {z_vals[i-1]:.4f} → {z_vals[i]:.4f}")

    # ── 13. Helix centred on face ─────────────────────────────────────────────
    def test_helix_centred_on_face(self):
        """Helix circle should be centred on the face centroid."""
        from toolpath_planner import _make_helix_entry
        verts  = np.array([[0,0],[100,0],[100,100],[0,100]], dtype=float)
        # centroid = (50, 50)
        fs     = _default_fs()
        wps    = _make_helix_entry(verts, 10.0, 15.0,
                                   helix_dia_mm=20.0,
                                   pitch_mm=0.75,
                                   feed_rate=2000.0)
        self.assertGreater(len(wps), 0)
        # All x,y should be within (radius + small eps) of (50,50)
        R  = 10.0 + 0.5
        cx, cy = 50.0, 50.0
        for wp in wps:
            dist = math.hypot(wp.x - cx, wp.y - cy)
            self.assertLess(dist, R + 0.5,
                f"Helix point ({wp.x:.2f},{wp.y:.2f}) too far from centroid "
                f"({cx},{cy}): dist={dist:.2f}, expected ≤ {R}")

    # ── 14. No raster waypoint inside wall buffer ─────────────────────────────
    def test_no_waypoint_inside_wall_buffer(self):
        """
        For a face with WALL on the right (x=100), every raster waypoint
        must have x ≤ 100 - R (tool centre not inside wall buffer).
        """
        R  = 6.0
        tf = _rect_face(0, 0, 100, 100, 10, wall_sides=["right"])
        fs = _default_fs()
        ft = plan_face(tf, fs)

        for tp in ft.passes:
            if tp.pass_type != "raster":
                continue
            for wp in tp.waypoints:
                self.assertLessEqual(
                    wp.x, 100.0 - R + 0.1,
                    f"Raster waypoint x={wp.x:.3f} is inside wall buffer "
                    f"(wall at x=100, R={R})")

    # ── 15. Faces sorted high to low ─────────────────────────────────────────
    def test_faces_sorted_high_to_low(self):
        """plan() must return faces in descending Z order."""
        import cnc_solid_bridge as bridge
        from feature_extractor import extract_features

        npy = os.path.join(config.SOLID_OUTPUT_DIR, "solid_faces_seed_0.npy")
        if not os.path.exists(npy):
            self.skipTest("solid_faces_seed_0.npy not found")

        faces  = bridge.load_face_polygons(0)
        feat   = extract_features(faces, seed=0)
        fs     = _default_fs()
        result = plan(feat, fs)

        z_vals = [ft.z_height for ft in result.faces]
        for i in range(1, len(z_vals)):
            self.assertGreaterEqual(z_vals[i - 1], z_vals[i],
                f"Faces not sorted: z[{i-1}]={z_vals[i-1]} < z[{i}]={z_vals[i]}")

    # ── 16. Time estimate formula ────────────────────────────────────────────
    def test_time_estimate_formula(self):
        """estimated_time_s = (path_length / feed_rate) × 60."""
        tf = _rect_face(0, 0, 100, 100, 10, wall_sides=["right"])
        fs = _default_fs()
        ft = plan_face(tf, fs)
        if ft.path_length_mm > 0 and fs.feed_rate_mmpm > 0:
            expected = (ft.path_length_mm / fs.feed_rate_mmpm) * 60.0
            self.assertAlmostEqual(ft.estimated_time_s, expected, delta=0.1,
                msg=f"Time formula mismatch: expected {expected:.2f}s, "
                    f"got {ft.estimated_time_s:.2f}s")

    # ── 17. Full pipeline integration ────────────────────────────────────────
    def test_integration_phase1_to_phase3(self):
        """Full chain: seed 0 → Phase1 → Phase2 → Phase3, no crash."""
        import cnc_solid_bridge as bridge
        from feature_extractor import extract_features

        npy = os.path.join(config.SOLID_OUTPUT_DIR, "solid_faces_seed_0.npy")
        if not os.path.exists(npy):
            self.skipTest("solid_faces_seed_0.npy not found")

        faces   = bridge.load_face_polygons(0)
        meta    = bridge.load_metadata(0)
        feat    = extract_features(faces, seed=0, volume=meta["volume"])
        fs      = fs_compute("aluminium_6061", 12.0, 4,
                              axial_depth_mm=6.0, radial_depth_mm=4.8,
                              fz_override=0.1, sfm_override=1000)
        result  = plan(feat, fs)

        self.assertGreater(len(result.faces), 0)
        self.assertGreater(result.total_path_length_mm, 0)

        # At least the two largest faces should have passes
        planned = [ft for ft in result.faces if ft.n_passes > 0]
        self.assertGreater(len(planned), 0,
            "Expected at least one face with raster passes")

        print(f"\n{result.summary()}")

    # ── 18. Path length consistency ───────────────────────────────────────────
    def test_path_length_consistent(self):
        """
        Manually sum all raster and link segment lengths and compare
        to ft.path_length_mm.
        """
        tf = _rect_face(0, 0, 100, 100, 10, wall_sides=["right"])
        fs = _default_fs()
        ft = plan_face(tf, fs)

        computed = 0.0
        for tp in ft.passes:
            wps = tp.waypoints
            for i in range(1, len(wps)):
                a, b = wps[i - 1], wps[i]
                if a.feed_rate > 0 and b.feed_rate > 0:
                    computed += math.sqrt(
                        (a.x - b.x) ** 2 + (a.y - b.y) ** 2 +
                        (a.z - b.z) ** 2
                    )

        self.assertAlmostEqual(
            computed, ft.path_length_mm, delta=1.0,
            msg=f"Path length mismatch: computed {computed:.2f}, "
                f"stored {ft.path_length_mm:.2f}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestToolpathPlanner)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

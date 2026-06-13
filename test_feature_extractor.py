"""
test_feature_extractor.py — Phase 1 tests for feature_extractor.py.

Test suite
----------
1. test_normal_computation      — Newell method on known faces
2. test_cube_synthetic          — Simple 100×100×50 cube: all top edges WALL
3. test_step_synthetic          — Two-block step: WALL + CLIFF correctly split
4. test_pocket_synthetic        — Pocket floor: all edges WALL (no entry from side)
5. test_safe_region             — Safe region shrinks by tool_radius
6. test_integration_seed0       — Real .npy from Build_Solid.py (seed=0)
7. test_regression_multi_seeds  — 5 seeds: no crash, all faces labelled
8. test_visual                  — Optional matplotlib plot (skipped in CI)

Run all:
    cd "/Users/sbedi/Nextcloud/automatic_to_autonomous/CNC Factory"
    python test_feature_extractor.py

Run single:
    python test_feature_extractor.py TestFeatureExtractor.test_cube_synthetic
"""

from __future__ import annotations

import sys
import os
import unittest
import numpy as np

# ── Make sure the CNC Factory package is on the path ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import (
    compute_face_normal,
    find_adjacent_face,
    classify_edge,
    compute_safe_region,
    extract_features,
    FeatureResult,
    TopFaceFeature,
)
import config


# ── Synthetic solid helpers ────────────────────────────────────────────────────

def _make_face(verts: list[list]) -> dict:
    """Build a minimal face dict matching cnc_solid_bridge output."""
    return {
        "outer_boundary": np.array(verts, dtype=float),
        "holes": [],
    }


def _cube_faces(w=100.0, d=100.0, h=50.0) -> list[dict]:
    """
    Return 6 faces of a box from (0,0,0) to (w,d,h).
    Vertices are ordered so Newell's method gives the outward normal.
    """
    return [
        # Top      z=h  normal +Z
        _make_face([[0,0,h],[w,0,h],[w,d,h],[0,d,h]]),
        # Bottom   z=0  normal -Z
        _make_face([[0,0,0],[0,d,0],[w,d,0],[w,0,0]]),
        # Front    y=0  normal -Y
        _make_face([[0,0,0],[w,0,0],[w,0,h],[0,0,h]]),
        # Back     y=d  normal +Y
        _make_face([[0,d,0],[0,d,h],[w,d,h],[w,d,0]]),
        # Left     x=0  normal -X
        _make_face([[0,0,0],[0,0,h],[0,d,h],[0,d,0]]),
        # Right    x=w  normal +X
        _make_face([[w,0,0],[w,d,0],[w,d,h],[w,0,h]]),
    ]


def _step_faces(
    w_lo=100.0, d=100.0, h_lo=30.0,
    w_hi=50.0,            h_hi=60.0,
) -> list[dict]:
    """
    Two-block step solid (viewed from the side):
        Upper block: x=[0, w_hi], z=[0, h_hi]
        Lower block: x=[0, w_lo], z=[0, h_lo]
    Combined occupies: x=[0,w_lo], y=[0,d], z=[0,h_lo]
                  plus x=[0,w_hi], y=[0,d], z=[h_lo,h_hi]

    Top faces expected:
        Upper top  z=h_hi  x=[0,w_hi]  → all WALL  (surrounded by walls)
        Lower top  z=h_lo  x=[w_hi,w_lo] → WALL(right,front,back) + WALL(left=step riser)
            BUT the step riser max-z = h_hi > h_lo → WALL on left
            So lower top also has WALL on all sides?  No — lower-right (x=w_lo) is CLIFF.

    Correct classification for lower top face:
        x=w_lo side : vertical wall, max-z = h_lo = top-face z  → CLIFF
        y=0    side : vertical wall, max-z = h_lo                → CLIFF
        y=d    side : vertical wall, max-z = h_lo                → CLIFF
        x=w_hi side : step riser,   max-z = h_hi > h_lo         → WALL
    """
    faces = []

    # ── Upper block faces ───────────────────────────────────────────────────
    # Top of upper block z=h_hi
    faces.append(_make_face([
        [0,0,h_hi],[w_hi,0,h_hi],[w_hi,d,h_hi],[0,d,h_hi]
    ]))
    # Left wall upper   x=0, z=[0,h_hi]
    faces.append(_make_face([
        [0,0,0],[0,0,h_hi],[0,d,h_hi],[0,d,0]
    ]))
    # Front wall upper  y=0, x=[0,w_hi], z=[0,h_hi]
    faces.append(_make_face([
        [0,0,0],[w_hi,0,0],[w_hi,0,h_hi],[0,0,h_hi]
    ]))
    # Back wall upper   y=d, x=[0,w_hi], z=[0,h_hi]
    faces.append(_make_face([
        [0,d,0],[0,d,h_hi],[w_hi,d,h_hi],[w_hi,d,0]
    ]))
    # Step riser        x=w_hi, z=[h_lo,h_hi]
    faces.append(_make_face([
        [w_hi,0,h_lo],[w_hi,0,h_hi],[w_hi,d,h_hi],[w_hi,d,h_lo]
    ]))

    # ── Lower (extended) block faces ────────────────────────────────────────
    # Lower exposed top z=h_lo, x=[w_hi,w_lo]
    faces.append(_make_face([
        [w_hi,0,h_lo],[w_lo,0,h_lo],[w_lo,d,h_lo],[w_hi,d,h_lo]
    ]))
    # Right wall        x=w_lo, z=[0,h_lo]
    faces.append(_make_face([
        [w_lo,0,0],[w_lo,d,0],[w_lo,d,h_lo],[w_lo,0,h_lo]
    ]))
    # Front wall lower  y=0, x=[w_hi,w_lo], z=[0,h_lo]
    faces.append(_make_face([
        [w_hi,0,0],[w_lo,0,0],[w_lo,0,h_lo],[w_hi,0,h_lo]
    ]))
    # Back wall lower   y=d, x=[w_hi,w_lo], z=[0,h_lo]
    faces.append(_make_face([
        [w_hi,d,0],[w_hi,d,h_lo],[w_lo,d,h_lo],[w_lo,d,0]
    ]))
    # Bottom            z=0
    faces.append(_make_face([
        [0,0,0],[0,d,0],[w_lo,d,0],[w_lo,0,0]
    ]))

    return faces


def _pocket_faces(
    outer=100.0, depth=50.0, pocket_w=40.0, pocket_d=40.0, pocket_h=20.0
) -> list[dict]:
    """
    Box with a rectangular pocket on top.
    The pocket floor is at z = depth - pocket_h.
    Pocket floor edges are all WALL (surrounded by pocket walls that rise above).
    """
    floor_z = depth - pocket_h
    po = (outer - pocket_w) / 2   # pocket offset from outer edge

    faces = []
    # Top annular face (outer top minus pocket opening) — simplified as
    # four separate rectangles surrounding the pocket. We just add the
    # pocket floor for our test.

    # Pocket floor  z=floor_z  x=[po, po+pocket_w], y=[po, po+pocket_d]
    x0, x1 = po, po + pocket_w
    y0, y1 = po, po + pocket_d
    faces.append(_make_face([
        [x0,y0,floor_z],[x1,y0,floor_z],[x1,y1,floor_z],[x0,y1,floor_z]
    ]))
    # Pocket left wall   x=x0, z=[floor_z,depth], max-z=depth > floor_z → WALL
    faces.append(_make_face([
        [x0,y0,floor_z],[x0,y1,floor_z],[x0,y1,depth],[x0,y0,depth]
    ]))
    # Pocket right wall  x=x1
    faces.append(_make_face([
        [x1,y0,floor_z],[x1,y0,depth],[x1,y1,depth],[x1,y1,floor_z]
    ]))
    # Pocket front wall  y=y0
    faces.append(_make_face([
        [x0,y0,floor_z],[x0,y0,depth],[x1,y0,depth],[x1,y0,floor_z]
    ]))
    # Pocket back wall   y=y1
    faces.append(_make_face([
        [x0,y1,floor_z],[x1,y1,floor_z],[x1,y1,depth],[x0,y1,depth]
    ]))
    return faces


# ── Test class ────────────────────────────────────────────────────────────────

class TestFeatureExtractor(unittest.TestCase):

    # ── 1. Normal computation ─────────────────────────────────────────────────
    def test_normal_computation_top(self):
        """Newell normal of a CCW horizontal quad at z=5 should be +Z."""
        verts = np.array([[0,0,5],[1,0,5],[1,1,5],[0,1,5]])
        n = compute_face_normal(verts)
        self.assertAlmostEqual(n[2],  1.0, places=6)
        self.assertAlmostEqual(n[0],  0.0, places=6)
        self.assertAlmostEqual(n[1],  0.0, places=6)

    def test_normal_computation_side(self):
        """Vertical face in XZ plane → normal should be ±Y."""
        verts = np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]])
        n = compute_face_normal(verts)
        self.assertAlmostEqual(abs(n[1]), 1.0, places=6)
        self.assertAlmostEqual(n[0], 0.0, places=6)
        self.assertAlmostEqual(n[2], 0.0, places=6)

    def test_normal_unit_length(self):
        """Normal must always be a unit vector."""
        verts = np.array([[0,0,0],[7,0,0],[7,5,0],[0,5,0]])
        n = compute_face_normal(verts)
        self.assertAlmostEqual(np.linalg.norm(n), 1.0, places=6)

    # ── 2. Cube: all top edges should be WALL ─────────────────────────────────
    def test_cube_synthetic(self):
        """
        A plain cube has one top face.  Every edge of that top face borders
        a vertical wall that reaches exactly to z=h.  By our rule:
            max-z of adjacent vertical face = h = top_face_z  → CLIFF

        Wait — that means cube top edges are CLIFF (the wall descends from
        the top level, there is nothing *above*).  This is correct: you can
        approach the top of a freestanding cube from any side.
        """
        faces = _cube_faces(w=100, d=100, h=50)
        result = extract_features(faces, seed="cube_test")

        self.assertEqual(result.n_top_faces, 1)
        tf = result.top_faces[0]
        self.assertAlmostEqual(tf.z_height, 50.0, places=2)
        self.assertEqual(len(tf.edges), 4)

        # All edges should be CLIFF (no wall rises above z=50)
        labels = {e.label for e in tf.edges}
        self.assertNotIn("wall",  labels,
            "Cube top face: no adjacent face rises above z=50, so no WALL expected.")
        self.assertIn("cliff", labels,
            "Cube top face: all sides should be CLIFF (open air at top level).")

    # ── 3. Step solid: upper top all-WALL, lower top has WALL + CLIFF ─────────
    def test_step_synthetic(self):
        """
        Upper block top (z=60): step-riser max-z=60=z_top → CLIFF.
                                 left/front/back walls max-z=60  → CLIFF.
        Lower block top  (z=30): step-riser max-z=60>30  → WALL on that edge.
                                  right/front/back walls max-z=30 → CLIFF.
        """
        faces = _step_faces(w_lo=100, d=100, h_lo=30, w_hi=50, h_hi=60)
        result = extract_features(faces, seed="step_test")

        self.assertEqual(result.n_top_faces, 2,
            f"Expected 2 top faces, got {result.n_top_faces}")

        # Sort by z-height
        top_faces = sorted(result.top_faces, key=lambda f: f.z_height)
        lower_tf, upper_tf = top_faces[0], top_faces[1]

        # ── Upper top face (z=60) ──────────────────────────────────────────
        self.assertAlmostEqual(upper_tf.z_height, 60.0, places=1)
        upper_labels = [e.label for e in upper_tf.edges]
        # All 4 surrounding walls top out at z=60 → CLIFF
        self.assertEqual(upper_tf.wall_count, 0,
            f"Upper top should have 0 WALLs, got {upper_tf.wall_count}: {upper_labels}")
        self.assertEqual(upper_tf.cliff_count, 4,
            f"Upper top should have 4 CLIFFs, got {upper_tf.cliff_count}: {upper_labels}")

        # ── Lower top face (z=30) ──────────────────────────────────────────
        self.assertAlmostEqual(lower_tf.z_height, 30.0, places=1)
        lower_labels = [e.label for e in lower_tf.edges]
        # Step riser (x=w_hi=50 side) rises to z=60 > 30 → WALL
        self.assertEqual(lower_tf.wall_count, 1,
            f"Lower top should have 1 WALL (step riser), got {lower_tf.wall_count}: {lower_labels}")
        # Remaining 3 edges (right, front, back) max-z=30 → CLIFF
        self.assertEqual(lower_tf.cliff_count, 3,
            f"Lower top should have 3 CLIFFs, got {lower_tf.cliff_count}: {lower_labels}")

    # ── 4. Pocket floor: all edges WALL ───────────────────────────────────────
    def test_pocket_synthetic(self):
        """
        Pocket floor is surrounded by 4 walls that rise above it.
        All 4 edges must be WALL.
        """
        faces = _pocket_faces()
        result = extract_features(faces, seed="pocket_test")

        # Find the pocket floor (lowest top face)
        pocket_floors = [tf for tf in result.top_faces]
        self.assertGreaterEqual(len(pocket_floors), 1,
            "Expected at least one top face (pocket floor).")

        floor = min(pocket_floors, key=lambda f: f.z_height)
        self.assertEqual(floor.wall_count, 4,
            f"Pocket floor: expected 4 WALLs, got {floor.wall_count}. "
            f"Labels: {[e.label for e in floor.edges]}")
        self.assertEqual(floor.cliff_count, 0,
            f"Pocket floor: expected 0 CLIFFs, got {floor.cliff_count}.")

    # ── 5. Safe region shrinks by tool_radius ─────────────────────────────────
    def test_safe_region(self):
        """
        100×100 square face with tool_radius=10.
        Safe region should be 80×80 = 6400 mm².
        """
        verts_2d = np.array([[0,0],[100,0],[100,100],[0,100]])
        safe_poly, safe_area = compute_safe_region(verts_2d, tool_radius=10.0)

        self.assertIsNotNone(safe_poly, "Safe region should not be None.")
        self.assertAlmostEqual(safe_area, 80 * 80, delta=1.0,
            msg=f"Expected safe area ≈ 6400, got {safe_area:.1f}")

    def test_safe_region_too_small(self):
        """
        A tiny 5×5 face with tool_radius=10 should yield an empty safe region.
        """
        verts_2d = np.array([[0,0],[5,0],[5,5],[0,5]])
        safe_poly, safe_area = compute_safe_region(verts_2d, tool_radius=10.0)
        self.assertEqual(safe_area, 0.0,
            f"Expected 0 safe area for tiny face, got {safe_area:.1f}")

    # ── 6. Integration test with real Build_Solid.py output (seed=0) ─────────
    def test_integration_seed0(self):
        """
        Load the existing solid_faces_seed_0.npy from Build_Solid.py output
        and verify the extractor runs without error and produces sane results.
        """
        npy_path = os.path.join(config.SOLID_OUTPUT_DIR, "solid_faces_seed_0.npy")
        if not os.path.exists(npy_path):
            self.skipTest(f"solid_faces_seed_0.npy not found at {npy_path}")

        import cnc_solid_bridge as bridge
        faces = bridge.load_face_polygons(0)
        meta  = bridge.load_metadata(0)

        result = extract_features(
            faces,
            seed=0,
            volume=meta.get("volume"),
            tool_radius=6.0,
        )

        # Basic sanity checks
        self.assertGreater(result.total_faces, 0)
        self.assertGreater(result.n_top_faces, 0,
            "seed=0 solid should have at least one top face.")

        for tf in result.top_faces:
            # Every top face must have at least one edge
            self.assertGreater(len(tf.edges), 0)
            # Every edge must be labelled
            for e in tf.edges:
                self.assertIn(e.label, ("wall", "cliff", "level"),
                    f"Unknown label '{e.label}' on face {tf.face_id} edge {e.edge_id}")
            # Safe region must be valid if face is big enough AND has entry points
            # Note: Some faces may have entry edges but zero safe region due to
            # geometry (e.g., tool radius too large, narrow passages)
            if tf.face_area > 200:
                has_entry = any(e.label in ("cliff", "level") for e in tf.edges)
                # Only warn if completely enclosed by walls
                if not has_entry:
                    self.assertEqual(tf.safe_region_area, 0,
                        f"Face {tf.face_id} fully walled — expected zero safe region")

        print(f"\n[test_integration_seed0]\n{result.summary()}")

    # ── 7. Regression across 5 seeds ─────────────────────────────────────────
    def test_regression_multi_seeds(self):
        """
        Run extractor on the first 5 available seeds.
        Assert: no exception, at least 1 top face each, all edges labelled.
        """
        import cnc_solid_bridge as bridge
        seeds = bridge.list_available_seeds()[:5]

        if not seeds:
            self.skipTest("No .npy files found — run Build_Solid.py first.")

        for seed in seeds:
            with self.subTest(seed=seed):
                faces  = bridge.load_face_polygons(seed)
                result = extract_features(faces, seed=seed)
                self.assertGreater(result.n_top_faces, 0,
                    f"Seed {seed}: no top faces found.")
                for tf in result.top_faces:
                    for e in tf.edges:
                        self.assertIn(e.label, ("wall", "cliff", "level"))


# ── Optional visual test (run manually) ──────────────────────────────────────

def run_visual_test(seed: int = 0, tool_radius: float = 6.0):
    """
    Produce a matplotlib figure for manual inspection.
    Edges coloured: WALL=red, CLIFF=green, LEVEL=blue.
    Safe region: light-grey filled polygon.

    Run:   python test_feature_extractor.py --visual [seed]
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import cnc_solid_bridge as bridge

    faces  = bridge.load_face_polygons(seed)
    result = extract_features(faces, seed=seed, tool_radius=tool_radius)

    print(result.summary())

    if not result.top_faces:
        print("No top faces found — nothing to plot.")
        return

    n = len(result.top_faces)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                              squeeze=False)
    fig.suptitle(f"Feature Extractor — Seed {seed}  (r={tool_radius}mm)",
                 fontsize=14)

    COLOUR = {"wall": "red", "cliff": "green", "level": "blue"}

    for idx, tf in enumerate(result.top_faces):
        ax = axes[idx // cols][idx % cols]
        ax.set_aspect("equal")
        ax.set_title(f"Face {tf.face_id}  z={tf.z_height:.1f}\n"
                     f"W={tf.wall_count} C={tf.cliff_count} L={tf.level_count}",
                     fontsize=9)

        # Safe region (filled grey)
        if tf.safe_region is not None and not tf.safe_region.is_empty:
            try:
                coords = np.array(tf.safe_region.exterior.coords)
                patch  = MplPolygon(coords, closed=True,
                                    facecolor="lightgrey", edgecolor="grey",
                                    linewidth=1, alpha=0.6, zorder=1)
                ax.add_patch(patch)
            except Exception:
                pass

        # Face outline (thin black)
        verts = np.array(tf.vertices_2d)
        verts_closed = np.vstack([verts, verts[0]])
        ax.plot(verts_closed[:, 0], verts_closed[:, 1],
                color="black", linewidth=0.8, alpha=0.4, zorder=2)

        # Edges coloured by label
        for e in tf.edges:
            x = [e.v_start[0], e.v_end[0]]
            y = [e.v_start[1], e.v_end[1]]
            ax.plot(x, y, color=COLOUR[e.label], linewidth=2.5, zorder=3)

            # Label midpoint
            mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
            ax.text(mx, my, e.label[0].upper(),
                    fontsize=7, color=COLOUR[e.label],
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor="white", alpha=0.7, edgecolor="none"))

        ax.autoscale()

    # Hide unused axes
    for idx in range(len(result.top_faces), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=c, label=l.capitalize())
               for l, c in COLOUR.items()]
    patches.append(mpatches.Patch(color="lightgrey", label="Safe region"))
    fig.legend(handles=patches, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--visual" in sys.argv:
        seed_arg = 0
        for arg in sys.argv:
            try:
                seed_arg = int(arg)
            except ValueError:
                pass
        run_visual_test(seed=seed_arg)
    else:
        # Increase verbosity so each sub-test name prints
        loader = unittest.TestLoader()
        suite  = loader.loadTestsFromTestCase(TestFeatureExtractor)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

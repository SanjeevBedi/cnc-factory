"""
toolpath_planner.py — Phase 3 of the CNC Factory pipeline.

Responsibilities
----------------
1. Compute a WALL-only eroded safe region per top face:
       S_i = P_i  minus  (union of R-wide forbidden zones along WALL edges)
   CLIFF / LEVEL edges are left at full boundary — the tool may approach
   freely from those directions.

2. Generate zig-zag raster toolpaths inside S_i:
       - Passes run along the raster_angle direction (default 0 °, i.e. along X)
       - Stepover = ae from Phase 2 FeedsSpeedsResult (or explicit override)
       - Alternating direction on each pass (zig-zag / bi-directional)

3. Select entry strategy per face:
       - CLIFF entry  : linear lead-in from outside along a CLIFF edge
       - Pocket entry : helix spiral down (no reachable CLIFF edge)

4. Build ordered 3-D waypoint lists:
       [x, y, z_clearance]  — rapid position above entry
       [x, y, z_height]     — plunge / lead-in to face height
       ... raster passes ...
       [x, y, z_clearance]  — retract after last pass

5. Sort faces highest-Z → lowest-Z (machine top features first).

6. Estimate machining time from path lengths and feed rates.

Sources
-------
toolpath_tikz.tex  — safe region definition, raster strategy, entry rule
unified_plan.tex   — face order, algorithm skeleton
enhanced_plan.tex  — entry strategies (cliff / helix)

NO OpenCASCADE.  Pure numpy + shapely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from shapely.geometry import (
        LineString, MultiLineString, MultiPolygon,
        Point, Polygon as ShapelyPolygon,
    )
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    _SHAPELY_OK = True
except ImportError:
    _SHAPELY_OK = False
    print("[toolpath_planner] WARNING: shapely not installed.")

import config
from feature_extractor import TopFaceFeature, EdgeFeature
from feeds_speeds_engine import FeedsSpeedsResult


# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_CLEARANCE_MM  = 5.0    # Z clearance above face for rapid moves
_DEFAULT_LEAD_IN_MM    = 10.0   # linear lead-in distance past face boundary
_DEFAULT_RASTER_ANGLE  = 0.0    # degrees — raster along X
_MIN_PASS_LENGTH_MM    = 0.5    # discard raster segments shorter than this
_HELIX_STEPS_PER_REV   = 36     # angular resolution of helix circle (10° steps)
_RAPID_FEED            = 0.0    # 0 = rapid G0 (used as sentinel in G-code)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    x:         float
    y:         float
    z:         float
    feed_rate: float  # mm/min  (0.0 = rapid G0)
    move_type: str    # 'rapid' | 'linear' | 'helix'
    # Timing fields — set by timing_model.stamp_waypoints() after planning.
    # t_start and t_end are decimal simulated seconds from job start.
    # Both default to 0.0 until stamped; _done tracks execution state.
    t_start:   float = 0.0
    t_end:     float = 0.0
    _done:     bool  = False


@dataclass
class ToolpathPass:
    waypoints: list    # list[Waypoint]
    pass_type: str     # 'approach' | 'entry' | 'raster' | 'link' | 'retract'


@dataclass
class FaceToolpath:
    face_id:          int
    z_height:         float
    entry_type:       str          # 'cliff_linear' | 'helix' | 'skipped'
    entry_edge_label: str          # label of edge used for entry
    n_passes:         int
    stepover_mm:      float
    path_length_mm:   float
    estimated_time_s: float
    safe_region:      object       # Shapely Polygon | None
    passes:           list         # list[ToolpathPass]
    warnings:         list = field(default_factory=list)


@dataclass
class ToolpathResult:
    seed:                  object
    faces:                 list    # list[FaceToolpath], sorted high→low Z
    total_path_length_mm:  float
    total_estimated_time_s: float
    warnings:              list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Toolpath Summary — Seed {self.seed}",
            f"  Faces planned       : {len(self.faces)}",
            f"  Total path length   : {self.total_path_length_mm:.1f} mm",
            f"  Estimated time      : {self.total_estimated_time_s:.1f} s "
            f"({self.total_estimated_time_s/60:.2f} min)",
        ]
        for ft in self.faces:
            lines.append(
                f"  Face {ft.face_id:3d}  z={ft.z_height:6.2f}  "
                f"passes={ft.n_passes:3d}  "
                f"step={ft.stepover_mm:.2f}  "
                f"len={ft.path_length_mm:7.1f} mm  "
                f"entry={ft.entry_type}"
            )
        return "\n".join(lines)


# ── Safe region: erode WALL edges only ───────────────────────────────────────

def compute_safe_region_walls_only(
    face_verts_2d: np.ndarray,
    edges:         list,
    tool_radius:   float,
) -> tuple[object, float]:
    """
    Build the safe machining region for the tool CENTRE.

    Strategy (from toolpath_tikz.tex):
        S_i = P_i  minus  union(forbidden_zones on WALL edges)

    A WALL edge gets a forbidden zone = a rectangle of width 2·R centred on
    the edge line, clipped to the inside of P_i.
    CLIFF / LEVEL edges are untouched — the tool may reach the full boundary
    (and may overhang slightly on cliff sides to clean the step).

    Returns (shapely_polygon_or_None, area_float).
    """
    if not _SHAPELY_OK:
        return None, 0.0
    if len(face_verts_2d) < 3:
        return None, 0.0

    try:
        face_poly = ShapelyPolygon(face_verts_2d)
        if not face_poly.is_valid:
            face_poly = make_valid(face_poly)
        if face_poly.is_empty or face_poly.area < 1e-6:
            return None, 0.0

        safe = face_poly

        for edge in edges:
            if edge.label != "wall":
                continue
            p1 = np.array(edge.v_start[:2])
            p2 = np.array(edge.v_end[:2])
            if np.allclose(p1, p2, atol=1e-8):
                continue
            wall_line   = LineString([p1, p2])
            # Buffer both sides — the side outside the polygon is ignored
            # because we intersect (implicitly through difference) with face_poly
            forbidden   = wall_line.buffer(tool_radius, cap_style=2)
            safe        = safe.difference(forbidden)
            if safe.is_empty:
                return None, 0.0

        if not safe.is_valid:
            safe = make_valid(safe)
        if safe.is_empty or not hasattr(safe, 'area') or safe.area < 1e-6:
            return None, 0.0

        return safe, float(safe.area)

    except Exception as exc:
        print(f"[toolpath_planner] safe_region_walls_only failed: {exc}")
        return None, 0.0


# ── Raster generation ─────────────────────────────────────────────────────────

def _rotate_poly(poly: object, angle_rad: float) -> object:
    """Rotate a Shapely geometry about the origin."""
    from shapely.affinity import rotate
    return rotate(poly, math.degrees(angle_rad), origin=(0, 0), use_radians=False)


def generate_raster(
    safe_region:  object,
    stepover_mm:  float,
    angle_deg:    float = 0.0,
) -> list:
    """
    Generate zig-zag raster line segments inside safe_region.

    The raster lines run parallel to angle_deg (0 = along X axis).
    Passes are spaced stepover_mm apart along the perpendicular direction.
    Alternating passes run in opposite directions (zig-zag).

    Returns
    -------
    list of list of (x, y) tuples — each inner list is one pass (≥2 points).
    """
    if not _SHAPELY_OK or safe_region is None or safe_region.is_empty:
        return []
    if stepover_mm <= 0:
        raise ValueError(f"stepover_mm must be > 0, got {stepover_mm}")

    angle_rad = math.radians(angle_deg)

    # Rotate geometry so raster lines become horizontal (along X)
    rotated = _rotate_poly(safe_region, -angle_rad)

    minx, miny, maxx, maxy = rotated.bounds
    width  = maxx - minx
    height = maxy - miny
    if width < 1e-6 or height < 1e-6:
        return []

    # First pass y-start: centre the raster within the bounding box
    n_passes  = max(1, int(math.floor(height / stepover_mm)) + 1)
    y_start   = miny + (height - (n_passes - 1) * stepover_mm) / 2.0

    passes_rotated = []
    go_positive    = True  # zig-zag direction

    for k in range(n_passes):
        y = y_start + k * stepover_mm
        scan = LineString([(minx - 1.0, y), (maxx + 1.0, y)])

        try:
            clipped = scan.intersection(rotated)
        except Exception:
            continue

        if clipped.is_empty:
            continue

        # Collect individual LineString segments
        if clipped.geom_type == "LineString":
            segments = [clipped]
        elif clipped.geom_type == "MultiLineString":
            segments = list(clipped.geoms)
        elif clipped.geom_type == "GeometryCollection":
            segments = [g for g in clipped.geoms
                        if g.geom_type in ("LineString", "MultiLineString")]
        else:
            continue

        for seg in segments:
            if seg.geom_type == "MultiLineString":
                sub = list(seg.geoms)
            else:
                sub = [seg]
            for s in sub:
                coords = list(s.coords)
                if len(coords) < 2:
                    continue
                length = s.length
                if length < _MIN_PASS_LENGTH_MM:
                    continue
                # Apply zig-zag direction
                if not go_positive:
                    coords = coords[::-1]
                passes_rotated.append(coords)
                go_positive = not go_positive

    # Rotate all coordinates back to original frame
    from shapely.affinity import rotate as shapely_rotate

    all_passes = []
    for pass_coords in passes_rotated:
        line = LineString(pass_coords)
        line_back = shapely_rotate(line, angle_deg, origin=(0, 0))
        all_passes.append(list(line_back.coords))

    return all_passes


# ── Entry strategy helpers ────────────────────────────────────────────────────

def _find_cliff_entry(
    edges:          list,
    safe_region:    object,
    raster_passes:  list,
    tool_radius:    float,
    lead_in_mm:     float,
) -> tuple[Optional[str], Optional[list]]:
    """
    Attempt to plan a linear cliff entry onto the first raster pass.

    The entry approaches the safe region boundary along the direction
    perpendicular to the raster (i.e. along the same scan line), from the
    side of the first-pass start point where a CLIFF edge exists.

    Returns (entry_type, entry_waypoints_2d)  or (None, None) if not possible.
    entry_waypoints_2d = [(x_outside, y), (x_start, y)]
    """
    if not raster_passes:
        return None, None

    first_pass_start = raster_passes[0][0]  # (x, y)
    fx, fy = first_pass_start

    # Build a simple representation of cliff edges
    cliff_edges = [e for e in edges if e.label == "cliff"]
    if not cliff_edges:
        return None, None

    # Try four approach directions; pick the first that starts from a cliff side
    # and whose approach line exits the face polygon
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        # Check if any cliff edge has a component in this approach direction
        for ce in cliff_edges:
            ex = (ce.v_start[0] + ce.v_end[0]) / 2.0
            ey = (ce.v_start[1] + ce.v_end[1]) / 2.0
            # Edge midpoint should be on the approach side of the face
            if dx * (ex - fx) > 0 or dy * (ey - fy) > 0:
                # Approach from the outside: start lead_in_mm before face entry
                x_out = fx + dx * lead_in_mm
                y_out = fy + dy * lead_in_mm
                return "cliff_linear", [(x_out, y_out), (fx, fy)]

    # Fallback: approach from the first pass start direction unconstrained
    first_pass_end = raster_passes[0][-1]
    vx = first_pass_start[0] - first_pass_end[0]
    vy = first_pass_start[1] - first_pass_end[1]
    mag = math.hypot(vx, vy)
    if mag < 1e-9:
        return None, None
    vx, vy = vx / mag, vy / mag
    x_out = fx + vx * lead_in_mm
    y_out = fy + vy * lead_in_mm
    return "cliff_linear", [(x_out, y_out), (fx, fy)]


def _make_helix_entry(
    face_verts_2d: np.ndarray,
    z_top:         float,
    z_start:       float,
    helix_dia_mm:  float,
    pitch_mm:      float,
    feed_rate:     float,
) -> list:
    """
    Generate 3-D helix waypoints descending from z_start to z_top.
    The helix is centred at the centroid of the face polygon.
    Returns list of Waypoint.
    """
    if not _SHAPELY_OK:
        return []

    poly    = ShapelyPolygon(face_verts_2d)
    cx, cy  = poly.centroid.x, poly.centroid.y
    radius  = helix_dia_mm / 2.0
    depth   = z_start - z_top
    if depth <= 0 or pitch_mm <= 0:
        return []

    n_revs   = depth / pitch_mm
    n_points = max(8, int(math.ceil(n_revs * _HELIX_STEPS_PER_REV)))
    waypoints = []

    for i in range(n_points + 1):
        t    = i / n_points                         # 0 → 1
        ang  = t * n_revs * 2.0 * math.pi           # cumulative angle
        z_pt = z_start - t * depth
        x_pt = cx + radius * math.cos(ang)
        y_pt = cy + radius * math.sin(ang)
        waypoints.append(Waypoint(x_pt, y_pt, z_pt, feed_rate, "helix"))

    return waypoints


# ── Per-face planner ──────────────────────────────────────────────────────────

def plan_face(
    top_face:      TopFaceFeature,
    fs_result:     FeedsSpeedsResult,
    clearance_mm:  float = _DEFAULT_CLEARANCE_MM,
    lead_in_mm:    float = _DEFAULT_LEAD_IN_MM,
    raster_angle:  float = _DEFAULT_RASTER_ANGLE,
    stepover_mm:   Optional[float] = None,
) -> FaceToolpath:
    """
    Plan the complete toolpath for one top face.

    Parameters
    ----------
    top_face      : TopFaceFeature from Phase 1
    fs_result     : FeedsSpeedsResult from Phase 2
    clearance_mm  : Z height above face for rapid moves
    lead_in_mm    : linear lead-in distance outside face boundary
    raster_angle  : raster direction in degrees (0 = along X)
    stepover_mm   : override radial stepover; defaults to fs_result.cut.radial_depth_mm

    Returns
    -------
    FaceToolpath
    """
    warnings: list[str] = []
    tool_radius  = fs_result.tool.radius_mm
    feed_rate    = fs_result.feed_rate_mmpm
    plunge_feed  = fs_result.plunge_feed_mmpm
    helix_feed   = fs_result.helix_feed_mmpm
    ae           = stepover_mm if stepover_mm is not None \
                   else fs_result.cut.radial_depth_mm
    z_height     = top_face.z_height
    z_clear      = z_height + clearance_mm
    verts_2d     = np.array(top_face.vertices_2d)
    edges        = top_face.edges
    face_id      = top_face.face_id

    # ── 1. Build wall-only safe region ────────────────────────────────────
    safe_region, safe_area = compute_safe_region_walls_only(
        verts_2d, edges, tool_radius
    )

    if safe_region is None or safe_area < 1e-6:
        warnings.append(
            f"Face {face_id}: safe region is empty with tool r={tool_radius:.1f} mm — "
            "face too small or all edges are WALLs with no room.")
        return FaceToolpath(
            face_id=face_id, z_height=z_height,
            entry_type="skipped", entry_edge_label="n/a",
            n_passes=0, stepover_mm=ae,
            path_length_mm=0.0, estimated_time_s=0.0,
            safe_region=None, passes=[], warnings=warnings,
        )

    # ── 2. Generate raster passes (in 2D) ────────────────────────────────
    pass_coords_list = generate_raster(safe_region, ae, raster_angle)

    if not pass_coords_list:
        warnings.append(
            f"Face {face_id}: no raster passes generated "
            f"(safe_area={safe_area:.1f} mm², step={ae:.2f} mm).")
        return FaceToolpath(
            face_id=face_id, z_height=z_height,
            entry_type="skipped", entry_edge_label="n/a",
            n_passes=0, stepover_mm=ae,
            path_length_mm=0.0, estimated_time_s=0.0,
            safe_region=safe_region, passes=[], warnings=warnings,
        )

    # ── 3. Entry strategy ─────────────────────────────────────────────────
    cliff_edges = [e for e in edges if e.label == "cliff"]
    entry_type  = "helix"
    entry_label = "none"
    entry_2d    = None

    if cliff_edges:
        etype, entry_2d = _find_cliff_entry(
            edges, safe_region, pass_coords_list, tool_radius, lead_in_mm
        )
        if etype is not None:
            entry_type  = etype
            entry_label = "cliff"

    # ── 4. Build 3-D waypoints into ToolpathPass objects ─────────────────
    all_passes: list[ToolpathPass] = []
    total_length = 0.0

    # 4a. Approach: rapid to above entry / face centre
    if entry_type == "cliff_linear" and entry_2d is not None:
        x_approach, y_approach = entry_2d[0]
    else:
        if _SHAPELY_OK:
            cx = safe_region.centroid.x
            cy = safe_region.centroid.y
        else:
            cx, cy = float(np.mean(verts_2d[:, 0])), float(np.mean(verts_2d[:, 1]))
        x_approach, y_approach = cx, cy

    approach_pass = ToolpathPass(
        waypoints=[Waypoint(x_approach, y_approach, z_clear,
                            _RAPID_FEED, "rapid")],
        pass_type="approach",
    )
    all_passes.append(approach_pass)

    # 4b. Entry move
    if entry_type == "cliff_linear" and entry_2d is not None:
        x_out, y_out = entry_2d[0]
        x_in,  y_in  = entry_2d[1]
        entry_pass = ToolpathPass(
            waypoints=[
                Waypoint(x_out, y_out, z_height, plunge_feed, "linear"),
                Waypoint(x_in,  y_in,  z_height, feed_rate,   "linear"),
            ],
            pass_type="entry",
        )
        seg_len = math.hypot(x_in - x_out, y_in - y_out)
        total_length += seg_len
    else:
        # Helix entry — descend from z_clear to z_height
        helix_dia = fs_result.helix_diameter_mm
        pitch     = fs_result.helix_pitch_mm
        helix_wps = _make_helix_entry(
            verts_2d, z_height, z_clear, helix_dia, pitch, helix_feed
        )
        if not helix_wps:
            helix_wps = [Waypoint(x_approach, y_approach, z_height,
                                  plunge_feed, "linear")]
        # Helix path length ≈ arc length
        for i in range(1, len(helix_wps)):
            a, b = helix_wps[i - 1], helix_wps[i]
            total_length += math.sqrt(
                (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
            )
        entry_pass  = ToolpathPass(waypoints=helix_wps, pass_type="entry")
        entry_label = "helix"

    all_passes.append(entry_pass)

    # 4c. Raster passes + lateral links
    prev_end: Optional[tuple] = None

    for k, pass_coords in enumerate(pass_coords_list):
        # Link from previous pass end to this pass start (stay at z_height)
        if prev_end is not None:
            px, py = pass_coords[0]
            lx, ly = prev_end
            link_pass = ToolpathPass(
                waypoints=[
                    Waypoint(lx, ly, z_height, feed_rate, "linear"),
                    Waypoint(px, py, z_height, feed_rate, "linear"),
                ],
                pass_type="link",
            )
            seg_len = math.hypot(px - lx, py - ly)
            total_length += seg_len
            all_passes.append(link_pass)

        # Raster pass waypoints
        wps = [
            Waypoint(float(c[0]), float(c[1]), z_height, feed_rate, "linear")
            for c in pass_coords
        ]
        raster_pass = ToolpathPass(waypoints=wps, pass_type="raster")

        # Accumulate path length
        for i in range(1, len(wps)):
            a, b = wps[i - 1], wps[i]
            total_length += math.hypot(a.x - b.x, a.y - b.y)

        all_passes.append(raster_pass)
        prev_end = (pass_coords[-1][0], pass_coords[-1][1])

    # 4d. Retract
    if prev_end is not None:
        rx, ry = prev_end
        retract_pass = ToolpathPass(
            waypoints=[Waypoint(rx, ry, z_clear, _RAPID_FEED, "rapid")],
            pass_type="retract",
        )
        all_passes.append(retract_pass)

    # ── 5. Time estimate ──────────────────────────────────────────────────
    # Time = cutting_length / feed_rate  (ignore rapid time — negligible)
    estimated_time_s = (total_length / feed_rate) * 60.0 if feed_rate > 0 else 0.0

    return FaceToolpath(
        face_id          = face_id,
        z_height         = z_height,
        entry_type       = entry_type,
        entry_edge_label = entry_label,
        n_passes         = len(pass_coords_list),
        stepover_mm      = ae,
        path_length_mm   = total_length,
        estimated_time_s = estimated_time_s,
        safe_region      = safe_region,
        passes           = all_passes,
        warnings         = warnings,
    )


# ── Top-level planner ─────────────────────────────────────────────────────────

def plan(
    feature_result: object,
    fs_result:      FeedsSpeedsResult,
    clearance_mm:   float = _DEFAULT_CLEARANCE_MM,
    lead_in_mm:     float = _DEFAULT_LEAD_IN_MM,
    raster_angle:   float = _DEFAULT_RASTER_ANGLE,
    stepover_mm:    Optional[float] = None,
    min_face_area:  float = 1.0,
) -> ToolpathResult:
    """
    Plan toolpaths for all top faces in a FeatureResult (Phase 1 output).
    Faces are sorted highest-Z → lowest-Z (from unified_plan.tex).

    Parameters
    ----------
    feature_result  : FeatureResult from feature_extractor.extract_features()
    fs_result       : FeedsSpeedsResult from feeds_speeds_engine.compute()
    clearance_mm    : Z clearance for rapid moves above face
    lead_in_mm      : lead-in distance outside face boundary for cliff entry
    raster_angle    : raster direction in degrees
    stepover_mm     : override stepover; defaults to fs_result.cut.radial_depth_mm
    min_face_area   : skip faces smaller than this (mm²)

    Returns
    -------
    ToolpathResult
    """
    warnings_global: list[str] = []

    # Sort top faces: highest Z first  (machine highest features first)
    sorted_faces = sorted(
        feature_result.top_faces,
        key=lambda f: f.z_height,
        reverse=True,
    )

    face_toolpaths: list[FaceToolpath] = []
    total_length   = 0.0
    total_time_s   = 0.0

    for tf in sorted_faces:
        if tf.face_area < min_face_area:
            warnings_global.append(
                f"Face {tf.face_id}: area={tf.face_area:.2f} mm² < "
                f"min_face_area={min_face_area} mm² — skipped.")
            continue

        ft = plan_face(
            top_face     = tf,
            fs_result    = fs_result,
            clearance_mm = clearance_mm,
            lead_in_mm   = lead_in_mm,
            raster_angle = raster_angle,
            stepover_mm  = stepover_mm,
        )
        face_toolpaths.append(ft)
        total_length  += ft.path_length_mm
        total_time_s  += ft.estimated_time_s

    return ToolpathResult(
        seed                   = feature_result.seed,
        faces                  = face_toolpaths,
        total_path_length_mm   = total_length,
        total_estimated_time_s = total_time_s,
        warnings               = warnings_global,
    )

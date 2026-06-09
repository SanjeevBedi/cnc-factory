"""
feature_extractor.py — Phase 1 of the CNC Factory pipeline.

Responsibilities
----------------
1. Compute face normals from vertex geometry (Newell method).
2. Identify TOP faces (normal.z > threshold).
3. For every edge of every top face, find the adjacent face and classify:
       WALL  — adjacent vertical face rises ABOVE the current top-face z.
               Tool cannot enter from this side.
       CLIFF — no adjacent face, OR adjacent vertical face max-z ≤ top-face z.
               Open air or step-down — safe entry direction.
       LEVEL — adjacent face is horizontal (another flat face).
               Lateral move is safe.
4. Compute the safe cutting region for each top face using Shapely:
       safe_region = face_polygon.buffer(-tool_radius)
   The tool centre must remain inside this region to avoid overcut.

NO OpenCASCADE imports.  Pure numpy + shapely only.
Input  : list[dict]  from cnc_solid_bridge.load_face_polygons()
Output : FeatureResult  (dataclass, JSON-serialisable except Shapely objects)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid
    _SHAPELY_OK = True
except ImportError:
    _SHAPELY_OK = False
    print("[feature_extractor] WARNING: shapely not installed — "
          "safe_region will be None for all faces.")

import config


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EdgeFeature:
    edge_id:             int
    v_start:             list          # [x, y, z]
    v_end:               list          # [x, y, z]
    label:               str           # 'wall' | 'cliff' | 'level'
    adjacent_face_id:    Optional[int] # index into original face list, or None
    adjacent_face_normal: Optional[list] # [nx, ny, nz] or None


@dataclass
class TopFaceFeature:
    face_id:          int             # index in the original face list
    z_height:         float           # z-coordinate of the face (all vertices same z)
    normal:           list            # [0.0, 0.0, 1.0] or close
    vertices_3d:      list            # [[x,y,z], ...]
    vertices_2d:      list            # [[x,y], ...]  (z dropped)
    edges:            list            # list[EdgeFeature]
    wall_count:       int = 0
    cliff_count:      int = 0
    level_count:      int = 0
    face_area:        float = 0.0
    safe_region_area: float = 0.0
    safe_region:      object = field(default=None, repr=False)  # Shapely Polygon | None


@dataclass
class FeatureResult:
    seed:       object           # int or str seed identifier
    volume:     Optional[float]
    total_faces: int
    top_faces:  list             # list[TopFaceFeature]

    @property
    def n_top_faces(self) -> int:
        return len(self.top_faces)

    def summary(self) -> str:
        lines = [
            f"Seed: {self.seed}",
            f"Total faces: {self.total_faces}",
            f"Top faces:   {self.n_top_faces}",
            f"Volume:      {self.volume:.3f}" if self.volume else "Volume: unknown",
        ]
        for tf in self.top_faces:
            lines.append(
                f"  Face {tf.face_id:3d}  z={tf.z_height:7.2f}  "
                f"verts={len(tf.vertices_3d):2d}  "
                f"area={tf.face_area:8.1f}  "
                f"safe={tf.safe_region_area:8.1f}  "
                f"W={tf.wall_count} C={tf.cliff_count} L={tf.level_count}"
            )
        return "\n".join(lines)


# ── Core geometry helpers ─────────────────────────────────────────────────────

def compute_face_normal(vertices: np.ndarray) -> np.ndarray:
    """
    Compute the unit face normal using Newell's method.
    Robust for non-planar or concave polygons.

    Parameters
    ----------
    vertices : (N, 3) array of 3-D vertex coordinates.

    Returns
    -------
    Unit normal vector (3,).  Returns [0,0,0] if degenerate.
    """
    n = np.zeros(3)
    N = len(vertices)
    for i in range(N):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % N]
        n[0] += (v0[1] - v1[1]) * (v0[2] + v1[2])
        n[1] += (v0[2] - v1[2]) * (v0[0] + v1[0])
        n[2] += (v0[0] - v1[0]) * (v0[1] + v1[1])
    mag = np.linalg.norm(n)
    if mag < 1e-12:
        return np.zeros(3)
    return n / mag


def _edges_match(a1: np.ndarray, a2: np.ndarray,
                 b1: np.ndarray, b2: np.ndarray,
                 tol: float) -> bool:
    """True if edge a1→a2 matches edge b1→b2 (either direction)."""
    fwd = (np.allclose(a1, b1, atol=tol) and np.allclose(a2, b2, atol=tol))
    rev = (np.allclose(a1, b2, atol=tol) and np.allclose(a2, b1, atol=tol))
    return fwd or rev


def find_adjacent_face(
    v_start: np.ndarray,
    v_end:   np.ndarray,
    all_faces: list[dict],
    exclude_idx: int,
    tol: float = config.EDGE_MATCH_TOLERANCE,
) -> Optional[int]:
    """
    Return the index (into all_faces) of the face that shares the edge
    v_start → v_end, excluding the face at exclude_idx.
    Returns None if no face shares this edge (CLIFF).
    """
    for i, face in enumerate(all_faces):
        if i == exclude_idx:
            continue
        ob = face["outer_boundary"]
        N  = len(ob)
        for j in range(N):
            fv1 = ob[j]
            fv2 = ob[(j + 1) % N]
            if _edges_match(v_start, v_end, fv1, fv2, tol):
                return i
    return None


def classify_edge(
    v_start:      np.ndarray,
    v_end:        np.ndarray,
    top_face_z:   float,
    adj_face_idx: Optional[int],
    all_faces:    list[dict],
) -> tuple[str, Optional[list]]:
    """
    Classify one edge of a top face and return (label, adj_normal).

    Rules
    -----
    CLIFF  : no adjacent face exists, OR adjacent vertical face max-z ≤ top_face_z.
    WALL   : adjacent vertical face max-z >  top_face_z  (wall rises above).
    LEVEL  : adjacent face is horizontal (|normal.z| > NORMAL_UP_THRESHOLD).
    """
    if adj_face_idx is None:
        return "cliff", None

    adj_ob     = all_faces[adj_face_idx]["outer_boundary"]
    adj_normal = compute_face_normal(adj_ob)
    adj_nz     = abs(adj_normal[2])

    # Horizontal adjacent face → LEVEL
    if adj_nz > config.NORMAL_UP_THRESHOLD:
        return "level", adj_normal.tolist()

    # Vertical adjacent face: compare its maximum z to the current face z
    adj_z_max = adj_ob[:, 2].max()
    if adj_z_max > top_face_z + config.WALL_Z_ELEVATION_TOL:
        return "wall", adj_normal.tolist()
    else:
        return "cliff", adj_normal.tolist()


def compute_safe_region(
    vertices_2d: np.ndarray,
    tool_radius: float,
) -> tuple[object, float]:
    """
    Compute the Shapely safe-region polygon for the tool centre.

    The safe region is the face polygon eroded inward by tool_radius so the
    cutter never overhangs the face boundary.

    Returns
    -------
    (shapely_polygon_or_None, area_float)
    """
    if not _SHAPELY_OK:
        return None, 0.0

    if len(vertices_2d) < 3:
        return None, 0.0

    try:
        poly = ShapelyPolygon(vertices_2d)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area < 1e-6:
            return None, 0.0

        safe = poly.buffer(-tool_radius)
        if safe.is_empty or safe.area < 1e-6:
            return None, 0.0

        return safe, safe.area

    except Exception as exc:
        print(f"[feature_extractor] safe_region failed: {exc}")
        return None, 0.0


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_features(
    face_polygons: list[dict],
    seed:          object = None,
    volume:        Optional[float] = None,
    tool_radius:   float = config.DEFAULT_TOOL_RADIUS_MM,
    normal_up_thr: float = config.NORMAL_UP_THRESHOLD,
    flat_z_tol:    float = config.FLAT_FACE_Z_TOLERANCE,
    verbose:       bool  = False,
) -> FeatureResult:
    """
    Run Phase 1 feature extraction on the face-polygon list produced by
    Build_Solid.py / cnc_solid_bridge.load_face_polygons().

    Parameters
    ----------
    face_polygons : list of dicts with 'outer_boundary' (N,3) np.ndarray.
    seed          : seed identifier (for bookkeeping).
    volume        : solid volume (pass from cnc_solid_bridge.load_metadata).
    tool_radius   : cutter radius in mm for safe-region computation.
    normal_up_thr : normal.z > this → TOP face.
    flat_z_tol    : all z-values within this → truly flat face.
    verbose       : print per-edge classification details.

    Returns
    -------
    FeatureResult
    """
    top_face_features: list[TopFaceFeature] = []

    for face_idx, face in enumerate(face_polygons):
        ob = face["outer_boundary"]
        if len(ob) < 3:
            continue

        # ── 1. Compute normal ──────────────────────────────────────────────
        normal = compute_face_normal(ob)

        # ── 2. Is it a top face? ───────────────────────────────────────────
        if normal[2] < normal_up_thr:
            continue

        # ── 3. Verify flatness ─────────────────────────────────────────────
        z_vals = ob[:, 2]
        if (z_vals.max() - z_vals.min()) > flat_z_tol:
            continue   # not truly flat — skip

        z_height  = float(z_vals.mean())
        verts_2d  = ob[:, :2]              # drop z
        face_area = float(ShapelyPolygon(verts_2d).area) if _SHAPELY_OK else 0.0

        # ── 4. Classify each edge ──────────────────────────────────────────
        N      = len(ob)
        edges: list[EdgeFeature] = []
        wall_count = cliff_count = level_count = 0

        for j in range(N):
            v_start = ob[j]
            v_end   = ob[(j + 1) % N]

            adj_idx = find_adjacent_face(
                v_start, v_end, face_polygons, face_idx
            )
            label, adj_normal = classify_edge(
                v_start, v_end, z_height, adj_idx, face_polygons
            )

            if label == "wall":
                wall_count  += 1
            elif label == "cliff":
                cliff_count += 1
            else:
                level_count += 1

            if verbose:
                print(
                    f"  Face {face_idx:3d}  Edge {j:2d}  "
                    f"({v_start[0]:.1f},{v_start[1]:.1f},{v_start[2]:.1f})->"
                    f"({v_end[0]:.1f},{v_end[1]:.1f},{v_end[2]:.1f})  "
                    f"adj={adj_idx}  → {label.upper()}"
                )

            edges.append(EdgeFeature(
                edge_id            = j,
                v_start            = v_start.tolist(),
                v_end              = v_end.tolist(),
                label              = label,
                adjacent_face_id   = adj_idx,
                adjacent_face_normal = adj_normal,
            ))

        # ── 5. Compute safe region ─────────────────────────────────────────
        safe_poly, safe_area = compute_safe_region(verts_2d, tool_radius)

        top_face_features.append(TopFaceFeature(
            face_id          = face_idx,
            z_height         = z_height,
            normal           = normal.tolist(),
            vertices_3d      = ob.tolist(),
            vertices_2d      = verts_2d.tolist(),
            edges            = edges,
            wall_count       = wall_count,
            cliff_count      = cliff_count,
            level_count      = level_count,
            face_area        = face_area,
            safe_region_area = safe_area,
            safe_region      = safe_poly,
        ))

    return FeatureResult(
        seed        = seed,
        volume      = volume,
        total_faces = len(face_polygons),
        top_faces   = top_face_features,
    )


# ── Convenience: load from disk and extract in one call ───────────────────────

def extract_from_seed(
    seed:        int | str,
    tool_radius: float = config.DEFAULT_TOOL_RADIUS_MM,
    verbose:     bool  = False,
) -> FeatureResult:
    """
    Load face polygons from disk (via cnc_solid_bridge) and run extraction.
    Convenience wrapper so callers need only the seed value.
    """
    import cnc_solid_bridge as bridge
    meta          = bridge.load_metadata(seed)
    face_polygons = bridge.load_face_polygons(seed)
    return extract_features(
        face_polygons = face_polygons,
        seed          = seed,
        volume        = meta.get("volume"),
        tool_radius   = tool_radius,
        verbose       = verbose,
    )

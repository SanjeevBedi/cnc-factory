"""
cnc_solid_bridge.py — The ONLY CNC Factory file that interacts with
Build_Solid.py.  All communication is through disk files (.npy / .npz)
and subprocess calls.  Build_Solid.py is NEVER imported directly.

Public API
----------
load_face_polygons(seed)        -> list[dict]   raw face data from .npy
list_available_seeds()          -> list[int]    seeds with .npy on disk
generate_solid(seed, ...)       -> Path         run Build_Solid.py, return .npy path
solid_exists(seed)              -> bool
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

import config


# ── helpers ───────────────────────────────────────────────────────────────────

def _npy_path(seed: int | str) -> Path:
    return Path(config.SOLID_OUTPUT_DIR) / f"solid_faces_seed_{seed}.npy"


def _npz_path(seed: int | str) -> Path:
    return Path(config.SOLID_OUTPUT_DIR) / f"connectivity_matrices_seed_{seed}.npz"


# ── public API ────────────────────────────────────────────────────────────────

def solid_exists(seed: int | str) -> bool:
    """Return True if the .npy file for this seed already exists on disk."""
    return _npy_path(seed).exists()


def list_available_seeds() -> list[int]:
    """Return sorted list of integer seeds that have a .npy file on disk."""
    output_dir = Path(config.SOLID_OUTPUT_DIR)
    if not output_dir.exists():
        return []
    seeds = []
    for p in output_dir.glob("solid_faces_seed_*.npy"):
        m = re.search(r"solid_faces_seed_(\d+)\.npy$", p.name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def load_face_polygons(seed: int | str) -> list[dict]:
    """
    Load the face-polygon data saved by Build_Solid.py for the given seed.

    Returns
    -------
    list of dicts, each with keys:
        'outer_boundary' : np.ndarray  shape (N, 3)   — 3-D vertices
        'holes'          : list of np.ndarray          — hole polygons (rare)

    Raises
    ------
    FileNotFoundError if the .npy file does not exist for this seed.
    """
    path = _npy_path(seed)
    if not path.exists():
        raise FileNotFoundError(
            f"No face-polygon file found for seed {seed}.\n"
            f"Expected: {path}\n"
            f"Run generate_solid({seed}) first, or pick a seed from "
            f"list_available_seeds()."
        )
    data = np.load(str(path), allow_pickle=True).item()
    faces: list[dict] = data.get("faces", [])

    # ── 1. Normalise to numpy arrays ─────────────────────────────────────────
    for face in faces:
        if not isinstance(face["outer_boundary"], np.ndarray):
            face["outer_boundary"] = np.array(face["outer_boundary"], dtype=float)
        holes = face.get("holes", [])
        face["holes"] = [
            np.array(h, dtype=float) if not isinstance(h, np.ndarray) else h
            for h in holes
        ]

    # ── 2. Unit conversion: cm → mm ──────────────────────────────────────────
    # Build_Solid.py generates coordinates in cm; the toolpath planner,
    # feeds-and-speeds, and G-code generator all operate in mm.
    cm2mm = config.GEOM_CM_TO_MM        # default 10.0
    for face in faces:
        face["outer_boundary"] = face["outer_boundary"] * cm2mm
        face["holes"] = [h * cm2mm for h in face["holes"]]

    # ── 3. Minimum-footprint enforcement ─────────────────────────────────────
    # If the part’s XY footprint < MIN_FOOTPRINT_FRACTION of the workspace,
    # scale the whole solid up uniformly (XYZ) so the smallest workable
    # parts still produce realistic toolpaths and machining times.
    if faces and config.MIN_FOOTPRINT_FRACTION > 0.0:
        all_pts = np.vstack([
            face["outer_boundary"] for face in faces
            if len(face["outer_boundary"])
        ])
        part_x = float(all_pts[:, 0].max() - all_pts[:, 0].min())
        part_y = float(all_pts[:, 1].max() - all_pts[:, 1].min())
        part_area = part_x * part_y
        ws_area   = config.MACHINE_WORKSPACE_X_MM * config.MACHINE_WORKSPACE_Y_MM
        min_area  = ws_area * config.MIN_FOOTPRINT_FRACTION
        if part_area < min_area and part_area > 0.0:
            extra = (min_area / part_area) ** 0.5  # uniform XYZ scale
            for face in faces:
                face["outer_boundary"] = face["outer_boundary"] * extra
                face["holes"] = [h * extra for h in face["holes"]]

    return faces


def load_metadata(seed: int | str) -> dict:
    """
    Return top-level metadata (volume, num_faces) for the given seed.
    Does not load the full face-polygon data.
    """
    path = _npy_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"No .npy for seed {seed}: {path}")
    data = np.load(str(path), allow_pickle=True).item()
    return {
        "seed":      seed,
        "volume":    data.get("volume"),
        "num_faces": data.get("num_faces", len(data.get("faces", []))),
        "npy_path":  str(path),
        "npz_path":  str(_npz_path(seed)),
    }


def generate_solid(
    seed: int,
    output_dir: Optional[str] = None,
    no_graphics: bool = True,
    quiet: bool = True,
    force: bool = False,
) -> Path:
    """
    Run Build_Solid.py (in the pyocc conda environment) to generate the
    solid for *seed* and save its .npy file.

    Parameters
    ----------
    seed        : random seed passed to Build_Solid.py
    output_dir  : override output directory (default: config.SOLID_OUTPUT_DIR)
    no_graphics : pass --no-graphics flag (saves to PDF instead of display)
    quiet       : pass --quiet flag
    force       : re-generate even if .npy already exists

    Returns
    -------
    Path to the produced .npy file.

    Raises
    ------
    RuntimeError  if Build_Solid.py exits with a non-zero return code.
    FileNotFoundError if Build_Solid.py script is not found.
    """
    script = Path(config.BUILD_SOLID_SCRIPT)
    if not script.exists():
        raise FileNotFoundError(f"Build_Solid.py not found at: {script}")

    out_dir = output_dir or config.SOLID_OUTPUT_DIR
    npy_file = _npy_path(seed)

    if npy_file.exists() and not force:
        return npy_file

    cmd = [
        "conda", "run", "-n", config.PYOCC_CONDA_ENV,
        "python", str(script),
        "--seed",       str(seed),
        "--output-dir", str(out_dir),
    ]
    if no_graphics:
        cmd.append("--no-graphics")
    if quiet:
        cmd.append("--quiet")

    print(f"[cnc_solid_bridge] Generating seed {seed} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Build_Solid.py failed for seed {seed}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if not npy_file.exists():
        raise RuntimeError(
            f"Build_Solid.py succeeded but .npy not found at: {npy_file}"
        )

    print(f"[cnc_solid_bridge] Seed {seed} ready → {npy_file}")
    return npy_file

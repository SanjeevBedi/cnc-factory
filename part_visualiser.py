"""
part_visualiser.py — CNC Factory Part Geometry Visualiser
==========================================================
Visualises the face outlines and machining pockets for one or more seeds.

For each seed it produces a 3-panel matplotlib figure:

  Panel 1 — ALL FACES (top view, XY)
      Every face polygon coloured by Z-height (tall = warm, low = cool).
      Face index annotated at centroid.

  Panel 2 — TOP FACES with edge labels
      Only the upward-facing flat faces plotted.
      Each edge coloured by its classification:
          WALL  → red   (tool must not enter from this side)
          CLIFF → green (open air — safe entry)
          LEVEL → blue  (adjacent flat — lateral move safe)
      Safe/pocket region shown as a filled light-green polygon.

  Panel 3 — TOOLPATH PREVIEW
      Raster passes overlaid on the safe region.
      Entry move shown as a dashed orange arrow.
      Passes coloured from dark-blue (first) to light-blue (last).

Usage
-----
    python part_visualiser.py                    # picks first 4 available seeds
    python part_visualiser.py 42 1000 500        # specific seeds
    python part_visualiser.py --seed 42          # single seed, larger figure
    python part_visualiser.py --list             # print available seeds and exit

Dependencies: matplotlib, shapely, numpy (all available in the CNC Factory env)
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Polygon as ShapelyPolygon, LineString

# ── Project imports ───────────────────────────────────────────────────────────
import cnc_solid_bridge as bridge
import config
import feature_extractor as fe
import feeds_speeds_engine as fse
import toolpath_planner as tp

# ── Colour scheme (dark background) ──────────────────────────────────────────
BG_COLOR      = "#1a1a2e"   # deep navy
PANEL_COLOR   = "#16213e"   # panel background
GRID_COLOR    = "#2a2a4a"
TEXT_COLOR    = "#e0e0f0"
FACE_CMAP     = "plasma"    # face fill by Z-height
WALL_COLOR    = "#ff4f4f"   # red
CLIFF_COLOR   = "#4fff7f"   # bright green
LEVEL_COLOR   = "#4fa8ff"   # sky blue
SAFE_FILL     = "#1a6b2a"   # dark green fill for safe region
SAFE_EDGE     = "#3ddc52"   # bright green outline for safe region
RASTER_CMAP   = "cool"      # raster pass colour map
ENTRY_COLOR   = "#ffaa00"   # orange entry arrow


# ═════════════════════════════════════════════════════════════════════════════
#  Core drawing helpers
# ═════════════════════════════════════════════════════════════════════════════

def _poly_xy(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return closed (x, y) arrays for a polygon (appends first vertex)."""
    xs = np.append(verts[:, 0], verts[0, 0])
    ys = np.append(verts[:, 1], verts[0, 1])
    return xs, ys


def _shapely_to_xy(geom) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Convert a Shapely geometry (Polygon / MultiPolygon / GeometryCollection)
    into a list of (x_array, y_array) pairs for plotting.
    """
    results = []
    if geom is None or geom.is_empty:
        return results
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        results.append((np.array(x), np.array(y)))
        for interior in geom.interiors:
            xi, yi = interior.xy
            results.append((np.array(xi), np.array(yi)))
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for sub in geom.geoms:
            results.extend(_shapely_to_xy(sub))
    return results


def _centroid_2d(verts_2d: np.ndarray) -> tuple[float, float]:
    """Approximate centroid (mean of vertices)."""
    return float(verts_2d[:, 0].mean()), float(verts_2d[:, 1].mean())


# ═════════════════════════════════════════════════════════════════════════════
#  Panel 1: all faces coloured by Z-height
# ═════════════════════════════════════════════════════════════════════════════

def draw_all_faces(ax, face_polygons: list[dict], seed) -> None:
    """Panel 1 — bird's-eye view of ALL faces, shaded by Z-height."""
    ax.set_facecolor(PANEL_COLOR)
    ax.set_title(f"Seed {seed} — All Faces (by Z-height)", color=TEXT_COLOR,
                 fontsize=11, pad=6)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="--", alpha=0.5)

    # Gather Z range for colour normalisation
    all_z = []
    for f in face_polygons:
        ob = f["outer_boundary"]
        all_z.extend(ob[:, 2].tolist())
    z_min, z_max = min(all_z), max(all_z)
    z_range = z_max - z_min if z_max > z_min else 1.0
    cmap = plt.get_cmap(FACE_CMAP)

    for i, face in enumerate(face_polygons):
        ob = face["outer_boundary"]
        if len(ob) < 3:
            continue
        verts_2d = ob[:, :2]
        z_mean   = float(ob[:, 2].mean())
        norm_z   = (z_mean - z_min) / z_range
        colour   = cmap(norm_z)

        xs, ys = _poly_xy(verts_2d)
        ax.fill(xs, ys, color=colour, alpha=0.55, zorder=1)
        ax.plot(xs, ys, color=(*colour[:3], 0.85), linewidth=0.7, zorder=2)

        cx, cy = _centroid_2d(verts_2d)
        ax.text(cx, cy, str(i), color=TEXT_COLOR, fontsize=5,
                ha="center", va="center", zorder=3)

    # Colourbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Z height (mm)", color=TEXT_COLOR, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

    ax.set_xlabel("X (mm)", color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel("Y (mm)", color=TEXT_COLOR, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")


# ═════════════════════════════════════════════════════════════════════════════
#  Panel 2: top faces with edge labels + safe regions
# ═════════════════════════════════════════════════════════════════════════════

def draw_top_faces(ax, feat_result: fe.FeatureResult, seed) -> None:
    """Panel 2 — top faces only, edges colour-coded, safe pocket shown."""
    ax.set_facecolor(PANEL_COLOR)
    n = len(feat_result.top_faces)
    ax.set_title(f"Seed {seed} — Top Faces ({n}) + Edge Labels + Safe Regions",
                 color=TEXT_COLOR, fontsize=11, pad=6)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="--", alpha=0.5)

    if not feat_result.top_faces:
        ax.text(0.5, 0.5, "No top faces found", transform=ax.transAxes,
                color=TEXT_COLOR, ha="center", va="center", fontsize=12)
        return

    # Z range for face fill shading
    z_vals = [tf.z_height for tf in feat_result.top_faces]
    z_min, z_max = min(z_vals), max(z_vals)
    z_range = z_max - z_min if z_max > z_min else 1.0
    cmap = plt.get_cmap(FACE_CMAP)

    for tf in feat_result.top_faces:
        verts_2d = np.array(tf.vertices_2d)
        if len(verts_2d) < 3:
            continue
        norm_z  = (tf.z_height - z_min) / z_range
        fc      = (*cmap(norm_z)[:3], 0.25)   # translucent face fill

        xs, ys = _poly_xy(verts_2d)
        ax.fill(xs, ys, color=fc, zorder=1)

        # ── Safe / pocket region ─────────────────────────────────────────
        if tf.safe_region is not None and not tf.safe_region.is_empty:
            for sx, sy in _shapely_to_xy(tf.safe_region):
                ax.fill(sx, sy, color=SAFE_FILL, alpha=0.45, zorder=2)
                ax.plot(sx, sy, color=SAFE_EDGE, linewidth=1.2,
                        linestyle="-", zorder=3)

        # ── Face outline ─────────────────────────────────────────────────
        ax.plot(xs, ys, color=TEXT_COLOR, linewidth=0.8, zorder=4)

        # ── Edge labels ──────────────────────────────────────────────────
        for edge in tf.edges:
            x1, y1 = edge.v_start[0], edge.v_start[1]
            x2, y2 = edge.v_end[0],   edge.v_end[1]
            colour = (WALL_COLOR  if edge.label == "wall"  else
                      CLIFF_COLOR if edge.label == "cliff" else
                      LEVEL_COLOR)
            ax.plot([x1, x2], [y1, y2], color=colour,
                    linewidth=2.5, zorder=5, solid_capstyle="round")

        # Face id annotation
        cx, cy = _centroid_2d(verts_2d)
        ax.text(cx, cy, f"F{tf.face_id}\nz={tf.z_height:.0f}",
                color=TEXT_COLOR, fontsize=5.5,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="#00000066", ec="none"))

    # Legend
    legend_patches = [
        mpatches.Patch(color=WALL_COLOR,  label="WALL (no entry)"),
        mpatches.Patch(color=CLIFF_COLOR, label="CLIFF (open / entry OK)"),
        mpatches.Patch(color=LEVEL_COLOR, label="LEVEL (adjacent flat)"),
        mpatches.Patch(color=SAFE_EDGE,   label="Safe / pocket region"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=7, framealpha=0.3,
              labelcolor=TEXT_COLOR,
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)

    ax.set_xlabel("X (mm)", color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel("Y (mm)", color=TEXT_COLOR, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")


# ═════════════════════════════════════════════════════════════════════════════
#  Panel 3: toolpath raster passes
# ═════════════════════════════════════════════════════════════════════════════

def draw_toolpaths(ax, tp_result: tp.ToolpathResult,
                   feat_result: fe.FeatureResult, seed) -> None:
    """Panel 3 — raster toolpaths overlaid on the safe regions."""
    ax.set_facecolor(PANEL_COLOR)
    n_passes_total = sum(ft.n_passes for ft in tp_result.faces)
    t_min = tp_result.total_estimated_time_s / 60.0
    ax.set_title(
        f"Seed {seed} — Toolpaths  "
        f"({len(tp_result.faces)} faces, {n_passes_total} passes, "
        f"{t_min:.1f} min)",
        color=TEXT_COLOR, fontsize=11, pad=6)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="--", alpha=0.5)

    ramp_cmap = plt.get_cmap(RASTER_CMAP)

    for ft in tp_result.faces:
        if ft.entry_type == "skipped":
            continue

        # ── Safe region background ────────────────────────────────────────
        if ft.safe_region is not None and not ft.safe_region.is_empty:
            for sx, sy in _shapely_to_xy(ft.safe_region):
                ax.fill(sx, sy, color=SAFE_FILL, alpha=0.35, zorder=1)
                ax.plot(sx, sy, color=SAFE_EDGE, linewidth=0.8, zorder=2)

        # Collect raster passes for this face (colour by pass index)
        raster_passes = [p for p in ft.passes if p.pass_type == "raster"]
        n_rp = len(raster_passes) or 1

        for k, rp in enumerate(raster_passes):
            wps = rp.waypoints
            if len(wps) < 2:
                continue
            xs = [w.x for w in wps]
            ys = [w.y for w in wps]
            colour = ramp_cmap(k / n_rp)
            ax.plot(xs, ys, color=colour, linewidth=0.9, zorder=4)

        # ── Entry move ────────────────────────────────────────────────────
        entry_passes = [p for p in ft.passes if p.pass_type == "entry"]
        for ep in entry_passes:
            wps = ep.waypoints
            if len(wps) >= 2:
                # Just draw the 2-D projection of the entry move
                xs = [w.x for w in wps]
                ys = [w.y for w in wps]
                ax.annotate(
                    "", xy=(xs[-1], ys[-1]), xytext=(xs[0], ys[0]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=ENTRY_COLOR,
                        lw=1.5,
                    ),
                    zorder=5,
                )
                ax.plot(xs, ys, color=ENTRY_COLOR,
                        linewidth=1.2, linestyle="--", zorder=5)

        # Face label
        if ft.safe_region is not None and not ft.safe_region.is_empty:
            cx = ft.safe_region.centroid.x
            cy = ft.safe_region.centroid.y
        else:
            cx = cy = 0.0
        ax.text(cx, cy, f"F{ft.face_id}\n{ft.n_passes}p",
                color=TEXT_COLOR, fontsize=5.5,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="#00000066", ec="none"))

    legend_patches = [
        mpatches.Patch(color=SAFE_EDGE,    label="Safe pocket region"),
        mpatches.Patch(color=ramp_cmap(0), label="First raster pass"),
        mpatches.Patch(color=ramp_cmap(1), label="Last raster pass"),
        mpatches.Patch(color=ENTRY_COLOR,  label="Entry move"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=7, framealpha=0.3,
              labelcolor=TEXT_COLOR,
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)

    ax.set_xlabel("X (mm)", color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel("Y (mm)", color=TEXT_COLOR, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")


# ═════════════════════════════════════════════════════════════════════════════
#  Stats panel (text summary)
# ═════════════════════════════════════════════════════════════════════════════

def draw_stats(ax, feat_result: fe.FeatureResult,
               fs_result: fse.FeedsSpeedsResult,
               tp_result: tp.ToolpathResult) -> None:
    """Small text summary panel."""
    ax.set_facecolor(PANEL_COLOR)
    ax.axis("off")
    ax.set_title("Pipeline Summary", color=TEXT_COLOR, fontsize=10, pad=4)

    t_s  = tp_result.total_estimated_time_s
    t_m  = t_s / 60.0
    lines_text = [
        f"Seed:              {feat_result.seed}",
        f"Total faces:       {feat_result.total_faces}",
        f"Top faces:         {feat_result.n_top_faces}",
        f"Volume:            {feat_result.volume:.1f} mm³"
                             if feat_result.volume else "Volume: unknown",
        "",
        f"Material:          {fs_result.material}",
        f"Tool:              ⌀{fs_result.tool.diameter_mm:.0f} mm  "
                             f"{fs_result.tool.flutes}fl",
        f"Spindle:           {fs_result.rpm:.0f} RPM",
        f"Feed rate:         {fs_result.feed_rate_mmpm:.0f} mm/min",
        f"Stepover:          {fs_result.cut.radial_depth_mm:.1f} mm",
        f"Power:             {fs_result.power_kw:.2f} kW "
                             f"{'✓' if fs_result.power_ok else '✗ OVER'}",
        "",
        f"Toolpath faces:    {len(tp_result.faces)}",
        f"Total passes:      {sum(f.n_passes for f in tp_result.faces)}",
        f"Path length:       {tp_result.total_path_length_mm:.0f} mm",
        f"Est. mach. time:   {t_m:.1f} min  ({t_s:.0f} s)",
        "",
        f"Config T_avg:      {config.SIM_T_AVG_S / 60:.0f} min target",
    ]

    y = 0.97
    for line in lines_text:
        ax.text(0.05, y, line, transform=ax.transAxes,
                color=TEXT_COLOR, fontsize=8,
                va="top", fontfamily="monospace")
        y -= 0.058

    # Warnings
    all_warnings = (
        fs_result.warnings
        + tp_result.warnings
        + [w for ft in tp_result.faces for w in ft.warnings]
    )
    if all_warnings:
        ax.text(0.05, y - 0.02, "Warnings:", transform=ax.transAxes,
                color=WALL_COLOR, fontsize=8, va="top", fontweight="bold")
        y -= 0.07
        for w in all_warnings[:6]:   # cap at 6 warnings
            ax.text(0.05, y, f"⚠ {w[:60]}", transform=ax.transAxes,
                    color=WALL_COLOR, fontsize=6.5, va="top", wrap=True)
            y -= 0.045


# ═════════════════════════════════════════════════════════════════════════════
#  Main per-seed figure
# ═════════════════════════════════════════════════════════════════════════════

def visualise_seed(seed: int, show: bool = True,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Run the pipeline for one seed and produce the 4-panel visualisation figure.

    Parameters
    ----------
    seed      : integer seed number (must have a .npy file on disk)
    show      : call plt.show() after building the figure
    save_path : if given, save the figure to this path (PNG/PDF/SVG)

    Returns
    -------
    matplotlib Figure
    """
    print(f"[part_visualiser] Processing seed {seed} …")

    # ── Phase 0: load raw faces ───────────────────────────────────────────
    try:
        meta          = bridge.load_metadata(seed)
        face_polygons = bridge.load_face_polygons(seed)
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    print(f"  Loaded {len(face_polygons)} faces")

    # ── Phase 1: feature extraction ───────────────────────────────────────
    feat_result = fe.extract_features(
        face_polygons = face_polygons,
        seed          = seed,
        volume        = meta.get("volume"),
        tool_radius   = config.DEFAULT_TOOL_RADIUS_MM,
    )
    print(f"  {feat_result.n_top_faces} top faces found")

    # ── Phase 2: feeds & speeds ───────────────────────────────────────────
    fs_result = fse.compute(
        material         = config.DEFAULT_MATERIAL,
        tool_diameter_mm = config.DEFAULT_TOOL_DIAMETER_MM,
        tool_flutes      = config.DEFAULT_TOOL_FLUTES,
        tool_type        = config.DEFAULT_TOOL_TYPE,
    )
    print(f"  Feed {fs_result.feed_rate_mmpm:.0f} mm/min, "
          f"{fs_result.rpm:.0f} RPM")

    # ── Phase 3: toolpath planning ────────────────────────────────────────
    tp_result = tp.plan(
        feature_result = feat_result,
        fs_result      = fs_result,
    )
    t_min = tp_result.total_estimated_time_s / 60.0
    print(f"  Toolpath: {len(tp_result.faces)} faces, "
          f"{sum(f.n_passes for f in tp_result.faces)} passes, "
          f"{t_min:.1f} min")

    # ── Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor=BG_COLOR)
    fig.suptitle(
        f"CNC Factory — Seed {seed}  |  "
        f"{len(face_polygons)} faces  |  "
        f"{feat_result.n_top_faces} top  |  "
        f"Est. {t_min:.1f} min machining",
        color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.98,
    )

    # Layout: 2 rows × 2 cols
    # [Panel 1: all faces]   [Panel 2: top faces + edges]
    # [Panel 3: toolpaths ]  [Panel 4: stats text       ]
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.97, top=0.94, bottom=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    draw_all_faces(ax1, face_polygons, seed)
    draw_top_faces(ax2, feat_result, seed)
    draw_toolpaths(ax3, tp_result, feat_result, seed)
    draw_stats(ax4, feat_result, fs_result, tp_result)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=BG_COLOR)
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  Multi-seed overview figure
# ═════════════════════════════════════════════════════════════════════════════

def visualise_multi(seeds: list[int], show: bool = True,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Build a compact 2-row overview for multiple seeds side-by-side.
    Row 1: all faces (Panel 1 style)
    Row 2: toolpaths (Panel 3 style)
    """
    n = len(seeds)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 10),
                              facecolor=BG_COLOR)
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        "CNC Factory — Part Geometry & Toolpath Overview",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.99,
    )

    for col, seed in enumerate(seeds):
        print(f"[part_visualiser] Processing seed {seed} …")
        try:
            meta          = bridge.load_metadata(seed)
            face_polygons = bridge.load_face_polygons(seed)
        except FileNotFoundError as exc:
            print(f"  SKIP seed {seed}: {exc}")
            for row in range(2):
                axes[row, col].set_facecolor(PANEL_COLOR)
                axes[row, col].text(0.5, 0.5, f"Seed {seed}\nnot found",
                                    transform=axes[row, col].transAxes,
                                    color=TEXT_COLOR, ha="center", va="center")
                axes[row, col].axis("off")
            continue

        feat_result = fe.extract_features(
            face_polygons = face_polygons,
            seed          = seed,
            volume        = meta.get("volume"),
            tool_radius   = config.DEFAULT_TOOL_RADIUS_MM,
        )
        fs_result = fse.compute(
            material         = config.DEFAULT_MATERIAL,
            tool_diameter_mm = config.DEFAULT_TOOL_DIAMETER_MM,
            tool_flutes      = config.DEFAULT_TOOL_FLUTES,
            tool_type        = config.DEFAULT_TOOL_TYPE,
        )
        tp_result = tp.plan(feat_result, fs_result)

        t_min = tp_result.total_estimated_time_s / 60.0
        print(f"  Seed {seed}: {feat_result.n_top_faces} top faces, "
              f"{sum(f.n_passes for f in tp_result.faces)} passes, "
              f"{t_min:.1f} min")

        draw_all_faces(axes[0, col], face_polygons, seed)
        draw_toolpaths(axes[1, col], tp_result, feat_result, seed)

        # Compact subtitle per column
        axes[1, col].set_title(
            axes[1, col].get_title()
            + f"\nFeed {fs_result.feed_rate_mmpm:.0f} mm/min  "
              f"⌀{fs_result.tool.diameter_mm:.0f} mm",
            color=TEXT_COLOR, fontsize=9,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=BG_COLOR)
        print(f"[part_visualiser] Saved overview → {save_path}")

    if show:
        plt.show()

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise CNC Factory part geometry (faces + toolpaths).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "seeds", nargs="*", type=int,
        help="Seed numbers to visualise (default: first 4 available).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Single seed — opens a full 4-panel detail figure.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available seeds and exit.",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save figure(s) to this base path (e.g. /tmp/part_vis).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show() (useful for headless rendering).",
    )
    return parser.parse_args()


def main():
    matplotlib.rcParams["axes.facecolor"]  = PANEL_COLOR
    matplotlib.rcParams["figure.facecolor"] = BG_COLOR
    matplotlib.rcParams["text.color"]       = TEXT_COLOR
    matplotlib.rcParams["axes.labelcolor"]  = TEXT_COLOR
    matplotlib.rcParams["xtick.color"]      = TEXT_COLOR
    matplotlib.rcParams["ytick.color"]      = TEXT_COLOR

    args = _parse_args()
    show = not args.no_show

    available = bridge.list_available_seeds()
    if not available:
        print("ERROR: No seeds found. "
              f"Check SOLID_OUTPUT_DIR = {config.SOLID_OUTPUT_DIR}")
        sys.exit(1)

    if args.list:
        print(f"Available seeds ({len(available)}):")
        for chunk_start in range(0, min(len(available), 100), 10):
            print("  ", available[chunk_start:chunk_start + 10])
        if len(available) > 100:
            print(f"  … and {len(available) - 100} more")
        sys.exit(0)

    # Single-seed detail mode
    if args.seed is not None:
        save = f"{args.save}_{args.seed}.png" if args.save else None
        visualise_seed(args.seed, show=show, save_path=save)
        return

    # Multi-seed overview
    if args.seeds:
        seeds = args.seeds
    else:
        # Default: pick a spread of the first 4 available seeds
        # Favour seeds 42, 100, 500, 1000 if present, else first 4
        preferred = [42, 100, 500, 1000]
        seeds = [s for s in preferred if s in available]
        if len(seeds) < 4:
            extras = [s for s in available if s not in seeds]
            seeds += extras[:4 - len(seeds)]
        seeds = seeds[:4]

    if len(seeds) == 1:
        save = f"{args.save}_{seeds[0]}.png" if args.save else None
        visualise_seed(seeds[0], show=show, save_path=save)
    else:
        save = f"{args.save}_overview.png" if args.save else None
        visualise_multi(seeds, show=show, save_path=save)


if __name__ == "__main__":
    main()

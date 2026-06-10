"""
timing_model.py — CNC Factory timing model
===========================================

Single source of truth for all factory timing arithmetic.

Public API
----------
compute_t_avg(seeds, n_samples)  → float
    Run the CAM pipeline on a sample of available seeds (with random material
    assignment) and return the mean pure-cutting time T_AVG_MACH_S.

seed_probability(t_avg_s)        → float
    Given T_AVG_MACH_S, return the per-tick seed-arrival probability that
    achieves config.SIM_IDLE_PCT % idle time.

shift_capacity(t_avg_s)          → dict
    Return a full breakdown: parts/machine, total capacity, target parts,
    ticks_per_shift, P_seed, etc.

assign_material()                → str
    Sample a material name from config.MATERIAL_WEIGHTS.

stamp_waypoints(passes, t0)      → float
    Walk through all ToolpathPass objects (from one FaceToolpath) and
    attach t_start / t_end fields to every Waypoint in decimal seconds.
    Returns the total face machining time in seconds.

stamp_toolpath(tp_result, fs_result, t0) → float
    Stamp all faces in a ToolpathResult. Returns total job machining time.

Timing rules
------------
  * Rapid (G00) moves use feed = RAPID_SPEED_MM_MIN for time calculation
    (they are fast but not instantaneous).
  * Feed (G01) moves use the waypoint's feed_rate field (mm/min).
  * Helix moves use the waypoint's feed_rate field.
  * All times are in decimal seconds; sub-T_TICK_S resolution is preserved.

At each tick the GUI executor does:
    current_sim_s = tick_number × config.T_TICK_S
    for each waypoint not yet tagged:
        if waypoint.t_end <= current_sim_s:
            mark as done

This means that on tick k, all moves whose END time falls in
  (  (k-1)×T_TICK_S,  k×T_TICK_S  ]
are completed in batch.  Sub-tick resolution is maintained in the
t_start/t_end stamps and is used for the completed-parts table.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import config

# Rapid traverse speed used for time-budget calculations (mm/min).
# Real machines: 15 000–30 000 mm/min.  Conservative default for planning.
RAPID_SPEED_MM_MIN: float = 15_000.0


# ─────────────────────────────────────────────────────────────────────────────
#  Material assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_material(rng: Optional[random.Random] = None) -> str:
    """
    Sample a material name from config.MATERIAL_WEIGHTS.

    Parameters
    ----------
    rng : optional random.Random instance (for reproducibility in tests).
          If None, uses the module-level random.

    Returns
    -------
    One of the keys from config.MATERIAL_TABLE.
    """
    names   = list(config.MATERIAL_WEIGHTS.keys())
    weights = [config.MATERIAL_WEIGHTS[n] for n in names]
    if rng is not None:
        return rng.choices(names, weights=weights, k=1)[0]
    return random.choices(names, weights=weights, k=1)[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Waypoint stamping
# ─────────────────────────────────────────────────────────────────────────────

def stamp_waypoints(passes, t0: float = 0.0) -> float:
    """
    Attach t_start and t_end (decimal seconds) to every Waypoint in a list
    of ToolpathPass objects.

    Parameters
    ----------
    passes : list[ToolpathPass]  — from a FaceToolpath
    t0     : start time of the first waypoint (seconds)

    Returns
    -------
    t_end of the last waypoint (= cumulative time through all passes).
    """
    t = t0
    for tp_pass in passes:
        wps = tp_pass.waypoints
        for i, wp in enumerate(wps):
            wp.t_start = t
            if i == 0:
                # First waypoint of pass — no move yet
                wp.t_end = t
            else:
                prev  = wps[i - 1]
                dist  = math.sqrt(
                    (wp.x - prev.x) ** 2
                    + (wp.y - prev.y) ** 2
                    + (wp.z - prev.z) ** 2
                )
                feed  = (wp.feed_rate
                         if wp.feed_rate > 0.0
                         else RAPID_SPEED_MM_MIN)
                dt    = (dist / feed) * 60.0   # mm / (mm/min) → min → ×60 → s
                t    += dt
                wp.t_end = t
    return t


def stamp_toolpath(tp_result, fs_result, t0: float = 0.0) -> float:
    """
    Stamp all faces in a ToolpathResult.

    Parameters
    ----------
    tp_result  : ToolpathResult (from toolpath_planner.plan)
    fs_result  : FeedsSpeedsResult (unused currently — feed comes from waypoints)
    t0         : absolute start time (seconds) of the whole job

    Returns
    -------
    Total job machining time in seconds (pure cutting, no setup/removal).
    """
    t = t0
    for face_tp in tp_result.faces:
        t = stamp_waypoints(face_tp.passes, t_start := t)
    return t - t0


# ─────────────────────────────────────────────────────────────────────────────
#  Average machining time computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_t_avg(
    seeds: Optional[list[int]] = None,
    n_samples: int = 20,
    verbose: bool = True,
) -> float:
    """
    Estimate T_AVG_MACH_S by running the Phase 1-3 CAM pipeline on a
    random sample of available seeds with random material assignment.

    Parameters
    ----------
    seeds     : list of seed integers to sample from.
                If None, reads all available seeds from disk.
    n_samples : number of seeds to sample (with replacement).
    verbose   : print per-seed timing to stdout.

    Returns
    -------
    Mean pure-cutting time in seconds.

    Side effect
    -----------
    Updates config.T_AVG_MACH_S in-place.
    """
    import cnc_solid_bridge as bridge
    import feature_extractor as fe
    import feeds_speeds_engine as fse
    import toolpath_planner as tp_mod

    if seeds is None:
        seeds = bridge.list_available_seeds()
    if not seeds:
        print("[timing_model] No seeds available — keeping T_AVG_MACH_S =",
              config.T_AVG_MACH_S)
        return config.T_AVG_MACH_S

    # Sample (with replacement when n_samples > len(seeds))
    sample = random.choices(seeds, k=min(n_samples, len(seeds)))
    if n_samples > len(seeds):
        sample = random.choices(seeds, k=n_samples)

    times: list[float] = []
    rng = random.Random(42)   # reproducible

    for seed in sample:
        try:
            material = assign_material(rng)
            meta     = bridge.load_metadata(seed)
            faces    = bridge.load_face_polygons(seed)
            feat     = fe.extract_features(
                face_polygons = faces,
                seed          = seed,
                volume        = meta.get("volume"),
                tool_radius   = config.DEFAULT_TOOL_RADIUS_MM,
            )
            fs = fse.compute(
                material         = material,
                tool_diameter_mm = config.DEFAULT_TOOL_DIAMETER_MM,
                tool_flutes      = config.DEFAULT_TOOL_FLUTES,
                tool_type        = config.DEFAULT_TOOL_TYPE,
            )
            tpr = tp_mod.plan(feat, fs)

            # Stamp waypoints so we get precise decimal-second times
            t_mach = stamp_toolpath(tpr, fs, t0=0.0)
            # Fall back to ToolpathResult's own estimate if stamping gives 0
            if t_mach <= 0.0:
                t_mach = tpr.total_estimated_time_s

            times.append(t_mach)
            if verbose:
                print(f"  Seed {seed:6d}  mat={material:<22s}  "
                      f"t={t_mach/60:.2f} min ({t_mach:.0f} s)")
        except Exception as exc:
            if verbose:
                print(f"  Seed {seed:6d}  SKIP: {exc}")

    if not times:
        print("[timing_model] All seeds failed — keeping T_AVG_MACH_S =",
              config.T_AVG_MACH_S)
        return config.T_AVG_MACH_S

    t_avg = sum(times) / len(times)
    config.T_AVG_MACH_S = t_avg   # update in-place

    if verbose:
        print(f"\n[timing_model] n={len(times)}  "
              f"min={min(times)/60:.1f} min  "
              f"max={max(times)/60:.1f} min  "
              f"MEAN={t_avg/60:.2f} min  ({t_avg:.0f} s)")
        cap = shift_capacity(t_avg)
        print(f"[timing_model] {_cap_summary(cap)}")

    return t_avg


# ─────────────────────────────────────────────────────────────────────────────
#  Shift capacity & seed probability
# ─────────────────────────────────────────────────────────────────────────────

def shift_capacity(t_avg_mach_s: Optional[float] = None) -> dict:
    """
    Compute full shift-capacity breakdown for the current configuration.

    Parameters
    ----------
    t_avg_mach_s : mean pure-cutting time (s).  Defaults to config.T_AVG_MACH_S.

    Returns
    -------
    dict with keys:
        t_avg_mach_s        float  — pure cutting time (s)
        avg_cycle_s         float  — full cycle incl. setup / removal / buffer
        max_parts_per_mach  int    — physical max parts per machine per shift
        total_capacity      int    — max × N_MACHINES
        idle_pct            float  — config.SIM_IDLE_PCT
        target_parts        int    — parts to produce (capacity × utilisation)
        ticks_per_shift     int    — shift_s / T_TICK_S
        p_seed_per_tick     float  — probability of offering one seed each tick
        shift_s             float  — shift duration (s)
    """
    t_avg  = t_avg_mach_s if t_avg_mach_s is not None else config.T_AVG_MACH_S
    shift_s = config.SIM_SHIFT_HOURS * 3600.0

    avg_cycle_s = (t_avg
                   + config.PART_SETUP_TIME_S
                   + config.PART_REMOVAL_TIME_S
                   + config.PART_BUFFER_TIME_S)

    max_per_mach = int(shift_s / avg_cycle_s)          # floor
    total_cap    = max_per_mach * config.SIM_N_MACHINES

    utilisation  = 1.0 - config.SIM_IDLE_PCT / 100.0
    target       = max(1, int(total_cap * utilisation))

    ticks        = int(shift_s / config.T_TICK_S)
    p_seed       = target / ticks if ticks > 0 else 0.0

    return {
        "t_avg_mach_s":       round(t_avg, 1),
        "avg_cycle_s":        round(avg_cycle_s, 1),
        "max_parts_per_mach": max_per_mach,
        "total_capacity":     total_cap,
        "idle_pct":           config.SIM_IDLE_PCT,
        "target_parts":       target,
        "ticks_per_shift":    ticks,
        "p_seed_per_tick":    round(p_seed, 8),
        "shift_s":            shift_s,
    }


def seed_probability(t_avg_mach_s: Optional[float] = None) -> float:
    """
    Return the per-tick seed-arrival probability for the current config.

    Convenience wrapper around shift_capacity().
    """
    return shift_capacity(t_avg_mach_s)["p_seed_per_tick"]


def _cap_summary(cap: dict) -> str:
    return (
        f"T_avg={cap['t_avg_mach_s']/60:.1f} min  "
        f"cycle={cap['avg_cycle_s']/60:.1f} min  "
        f"cap/mach={cap['max_parts_per_mach']}  "
        f"total_cap={cap['total_capacity']}  "
        f"idle={cap['idle_pct']:.0f}%  "
        f"target={cap['target_parts']} parts  "
        f"ticks={cap['ticks_per_shift']}  "
        f"P={cap['p_seed_per_tick']:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tick executor helper — used by factory_gui._do_sim_step
# ─────────────────────────────────────────────────────────────────────────────

def waypoints_due(passes, current_sim_s: float) -> list:
    """
    Return all Waypoints across the given passes whose t_end ≤ current_sim_s
    and that have NOT yet been tagged as done.

    Each Waypoint is expected to have:
        t_start  : float   (set by stamp_waypoints)
        t_end    : float   (set by stamp_waypoints)
        _done    : bool    (set here; absent = not done)

    Parameters
    ----------
    passes         : list[ToolpathPass]
    current_sim_s  : current simulated time (tick × T_TICK_S)

    Returns
    -------
    List of Waypoint objects that became due this tick.
    """
    due = []
    for tp_pass in passes:
        for wp in tp_pass.waypoints:
            if getattr(wp, "_done", False):
                continue
            t_end = getattr(wp, "t_end", None)
            if t_end is None:
                continue
            if t_end <= current_sim_s:
                wp._done = True
                due.append(wp)
    return due


def count_done(passes) -> tuple[int, int]:
    """
    Return (n_done, n_total) waypoints across all passes.
    """
    n_done  = sum(
        1 for p in passes for wp in p.waypoints
        if getattr(wp, "_done", False)
    )
    n_total = sum(len(p.waypoints) for p in passes)
    return n_done, n_total


# ─────────────────────────────────────────────────────────────────────────────
#  CLI self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Compute T_avg and shift capacity for the CNC factory."
    )
    ap.add_argument("--samples", type=int, default=20,
                    help="Number of seeds to sample (default 20).")
    ap.add_argument("--idle", type=float, default=None,
                    help="Override SIM_IDLE_PCT (e.g. 25, 10, 5).")
    args = ap.parse_args()

    if args.idle is not None:
        config.SIM_IDLE_PCT = args.idle

    print(f"T_TICK_S      = {config.T_TICK_S} s")
    print(f"T_G_S         = {config.T_G_S} s  ({config.T_G_MULT} ticks)")
    print(f"Shift         = {config.SIM_SHIFT_HOURS} h  "
          f"= {config._SHIFT_S:.0f} s  "
          f"= {config._TICKS_PER_SHIFT} ticks")
    print(f"SIM_IDLE_PCT  = {config.SIM_IDLE_PCT} %")
    print()

    # Compute from real seeds
    t_avg = compute_t_avg(n_samples=args.samples, verbose=True)

    print()
    print("─" * 72)
    print("IDLE SCENARIO COMPARISON")
    print("─" * 72)
    for idle in (25.0, 10.0, 5.0):
        config.SIM_IDLE_PCT = idle
        cap = shift_capacity(t_avg)
        label = {25.0: "base-case (25% idle)",
                 10.0: "normal    (10% idle)",
                 5.0:  "high-load ( 5% idle)"}[idle]
        print(f"  {label}  →  {_cap_summary(cap)}")

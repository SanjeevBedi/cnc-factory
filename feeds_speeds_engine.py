"""
feeds_speeds_engine.py — Phase 2 of the CNC Factory pipeline.

Responsibilities
----------------
1. Look up material cutting speed (SFM) and specific cutting force (Kc, MPa)
   from the shop-floor table in config.py.
2. Convert SFM → cutting speed Vc (m/min).
3. Compute spindle speed   N  = 1000·Vc / (π·D)          [RPM]
4. Select chip load        fz  from diameter-category table.
5. Compute feed rate       F   = N·Z·fz                   [mm/min]
6. Compute material-removal rate  MRR = ae·ap·F           [mm³/min]
7. Compute peak cutting force     Fc  = Kc·fz·ap          [N]      (per-tooth, worst-case)
8. Compute spindle power          P   = Fc·Vc / 60 000    [kW]
9. Compute spindle torque         T   = Fc·D  / 2 000     [Nm]
10. Check P and T against machine limits (config.py).
11. Compute entry-strategy feed overrides:
        Plunge  : 40 % of F
        Ramp    : F (horizontal component at ramp_angle)
        Helix   : 80 % of F,  dia = 1.3·D,  pitch = 0.75 mm/rev

Formulas source
---------------
feeds_speeds.tex   — SFM table, N formula, F formula, entry methods
Physics_based_model.tex — Fc = Kc·Ac, P = Fc·Vc/60000

NO OpenCASCADE.  Pure Python + math only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import config

# ── Constants ─────────────────────────────────────────────────────────────────

_FT_PER_MIN_TO_M_PER_MIN = 0.3048   # 1 SFM = 0.3048 m/min

# Plunge / ramp / helix multipliers (from feeds_speeds.tex)
_PLUNGE_FRACTION  = 0.40   # 40% of normal feed (midpoint of 30–50%)
_HELIX_FRACTION   = 0.80   # 80% of normal feed
_HELIX_DIA_FACTOR = 1.30   # helix diameter = 1.3 × tool diameter
_HELIX_PITCH_MM   = 0.75   # mm per revolution (midpoint of 0.5–1.0)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    """Tool geometry descriptor."""
    diameter_mm: float
    flutes:      int
    type:        str = "flat"   # flat | bull | ball

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0


@dataclass
class CutParameters:
    """Depth-of-cut parameters for a single pass."""
    axial_depth_mm:  float   # ap — depth in Z per pass (mm)
    radial_depth_mm: float   # ae — stepover / engagement width (mm)


@dataclass
class FeedsSpeedsResult:
    """
    Full result from feeds_speeds_engine.compute().
    All numeric values are in SI units (mm, N, kW, Nm) unless noted.
    """
    # ── Inputs (stored for traceability) ──────────────────────────────────
    material:    str
    tool:        ToolSpec
    cut:         CutParameters

    # ── Cutting speed ─────────────────────────────────────────────────────
    sfm_used:          float   # SFM value actually used
    cutting_speed_mpm: float   # Vc  (m/min)

    # ── Spindle ───────────────────────────────────────────────────────────
    rpm: float                 # N   (rev/min)

    # ── Chip load ─────────────────────────────────────────────────────────
    chip_load_mm: float        # fz  (mm/tooth)

    # ── Feed ─────────────────────────────────────────────────────────────
    feed_rate_mmpm: float      # F   (mm/min)

    # ── MRR ──────────────────────────────────────────────────────────────
    mrr_mm3pm: float           # ae·ap·F  (mm³/min)

    # ── Physics ───────────────────────────────────────────────────────────
    chip_area_mm2:   float     # Ac = fz·ap           (mm²)
    cutting_force_n: float     # Fc = Kc·Ac           (N)
    power_kw:        float     # P  = Fc·Vc / 60 000  (kW)
    torque_nm:       float     # T  = Fc·D  / 2 000   (Nm)

    # ── Constraint results ────────────────────────────────────────────────
    power_ok:  bool
    torque_ok: bool

    # ── Entry strategy feeds (mm/min) ─────────────────────────────────────
    plunge_feed_mmpm:  float   # 40 % of F
    ramp_feed_mmpm:    float   # F (at ramp angle — horizontal component)
    ramp_angle_deg:    float
    helix_feed_mmpm:   float   # 80 % of F
    helix_diameter_mm: float   # 1.3 × D
    helix_pitch_mm:    float   # 0.75 mm/rev

    # ── Warnings (default empty list — must stay last) ─────────────────────
    warnings:  list = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        return self.power_ok and self.torque_ok

    def summary(self) -> str:
        ok = "✅ FEASIBLE" if self.is_feasible else "❌ INFEASIBLE"
        lines = [
            f"=== Feeds & Speeds  [{ok}] ===",
            f"  Material      : {self.material}",
            f"  Tool          : ⌀{self.tool.diameter_mm:.1f}mm  {self.tool.flutes}fl  {self.tool.type}",
            f"  ap / ae       : {self.cut.axial_depth_mm:.2f} / {self.cut.radial_depth_mm:.2f} mm",
            f"  SFM used      : {self.sfm_used:.0f}  →  Vc = {self.cutting_speed_mpm:.1f} m/min",
            f"  Spindle       : {self.rpm:.0f} RPM",
            f"  Chip load fz  : {self.chip_load_mm:.4f} mm/tooth",
            f"  Feed rate     : {self.feed_rate_mmpm:.0f} mm/min",
            f"  MRR           : {self.mrr_mm3pm:.0f} mm³/min",
            f"  Cut force Fc  : {self.cutting_force_n:.1f} N",
            f"  Power         : {self.power_kw:.3f} kW  "
            f"({'OK' if self.power_ok else 'OVER LIMIT — ' + str(config.MACHINE_POWER_LIMIT_KW) + ' kW'})",
            f"  Torque        : {self.torque_nm:.2f} Nm  "
            f"({'OK' if self.torque_ok else 'OVER LIMIT — ' + str(config.MACHINE_TORQUE_LIMIT_NM) + ' Nm'})",
            f"  --- Entry feeds ---",
            f"  Plunge        : {self.plunge_feed_mmpm:.0f} mm/min",
            f"  Ramp ({self.ramp_angle_deg:.0f}°)    : {self.ramp_feed_mmpm:.0f} mm/min",
            f"  Helix         : {self.helix_feed_mmpm:.0f} mm/min  "
            f"(⌀{self.helix_diameter_mm:.1f}mm, pitch {self.helix_pitch_mm:.2f}mm/rev)",
        ]
        if self.warnings:
            lines.append("  --- Warnings ---")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sfm_to_mpm(sfm: float) -> float:
    """Convert surface feet per minute to metres per minute."""
    return sfm * _FT_PER_MIN_TO_M_PER_MIN


def _rpm(cutting_speed_mpm: float, diameter_mm: float) -> float:
    """
    N = 1000·Vc / (π·D)
    Vc in m/min, D in mm  →  N in RPM.
    Imperial equivalent: RPM = SFM × 3.82 / D_inches.
    """
    if diameter_mm <= 0:
        raise ValueError(f"Tool diameter must be > 0, got {diameter_mm}")
    return (1000.0 * cutting_speed_mpm) / (math.pi * diameter_mm)


def _chip_load(diameter_mm: float) -> tuple[float, float, float]:
    """
    Return (fz_min, fz_max, fz_mid) for the tool diameter category.
    Categories and values from feeds_speeds.tex.
    """
    if diameter_mm < 12.0:
        lo, hi = config.CHIP_LOAD_TABLE["small"]
    elif diameter_mm < 25.0:
        lo, hi = config.CHIP_LOAD_TABLE["medium"]
    else:
        lo, hi = config.CHIP_LOAD_TABLE["large"]
    return lo, hi, (lo + hi) / 2.0


def _default_depths(diameter_mm: float) -> tuple[float, float]:
    """
    Conservative default depths when the caller does not specify them.
        ap = 0.50 × D  (axial)
        ae = 0.40 × D  (radial / stepover)
    """
    return 0.50 * diameter_mm, 0.40 * diameter_mm


# ── Public API ────────────────────────────────────────────────────────────────

def list_materials() -> list[str]:
    """Return all material names in the lookup table."""
    return list(config.MATERIAL_TABLE.keys())


def material_info(material: str) -> dict:
    """
    Return the SFM range and Kc for a material.

    Returns
    -------
    {"sfm_min": int, "sfm_max": int, "kc_mpa": int}
    """
    if material not in config.MATERIAL_TABLE:
        raise KeyError(
            f"Unknown material '{material}'. "
            f"Available: {list(config.MATERIAL_TABLE.keys())}"
        )
    (sfm_min, sfm_max), kc = config.MATERIAL_TABLE[material]
    return {"sfm_min": sfm_min, "sfm_max": sfm_max, "kc_mpa": kc}


def compute(
    material:        str,
    tool_diameter_mm: float,
    tool_flutes:     int,
    tool_type:       str   = "flat",
    axial_depth_mm:  Optional[float] = None,
    radial_depth_mm: Optional[float] = None,
    sfm_override:    Optional[float] = None,
    fz_override:     Optional[float] = None,
    sfm_fraction:    float = 0.85,
    ramp_angle_deg:  float = 2.0,
) -> FeedsSpeedsResult:
    """
    Compute feeds, speeds, forces, and entry parameters for one operation.

    Parameters
    ----------
    material         : key from config.MATERIAL_TABLE
                       e.g. "aluminium_6061", "mild_steel_1018"
    tool_diameter_mm : cutter diameter (mm)
    tool_flutes      : number of cutting edges
    tool_type        : "flat" | "bull" | "ball"
    axial_depth_mm   : ap — depth per pass (mm).  Default = 0.50 × D.
    radial_depth_mm  : ae — stepover (mm).        Default = 0.40 × D.
    sfm_override     : use this SFM instead of the table value.
    fz_override      : use this chip load (mm/tooth) instead of the table value.
    sfm_fraction     : fraction of sfm_max to use when no override given.
                       0.85 = conservative/safe starting point.
    ramp_angle_deg   : ramp-entry angle in degrees (1–3 typical).

    Returns
    -------
    FeedsSpeedsResult
    """
    # ── 0. Validate inputs ────────────────────────────────────────────────
    info = material_info(material)
    if tool_diameter_mm <= 0:
        raise ValueError(f"tool_diameter_mm must be > 0, got {tool_diameter_mm}")
    if tool_flutes < 1:
        raise ValueError(f"tool_flutes must be >= 1, got {tool_flutes}")

    warnings: list[str] = []

    tool = ToolSpec(diameter_mm=tool_diameter_mm,
                    flutes=tool_flutes, type=tool_type)

    # ── 1. Depths of cut ─────────────────────────────────────────────────
    ap_default, ae_default = _default_depths(tool_diameter_mm)
    ap = axial_depth_mm  if axial_depth_mm  is not None else ap_default
    ae = radial_depth_mm if radial_depth_mm is not None else ae_default

    if ap <= 0 or ae <= 0:
        raise ValueError(f"Depths must be > 0: ap={ap}, ae={ae}")
    if ap > tool_diameter_mm:
        warnings.append(
            f"Axial depth ap={ap:.2f} mm exceeds tool diameter "
            f"{tool_diameter_mm:.1f} mm — risk of tool breakage.")
    if ae > tool_diameter_mm:
        warnings.append(
            f"Radial depth ae={ae:.2f} mm exceeds tool diameter "
            f"{tool_diameter_mm:.1f} mm — full-slot cut, reduce feed 20-30%.")

    cut = CutParameters(axial_depth_mm=ap, radial_depth_mm=ae)

    # ── 2. Cutting speed (SFM → m/min) ───────────────────────────────────
    if sfm_override is not None:
        sfm = sfm_override
    else:
        sfm = info["sfm_max"] * sfm_fraction

    vc = _sfm_to_mpm(sfm)

    # ── 3. Spindle speed (RPM) ────────────────────────────────────────────
    n = _rpm(vc, tool_diameter_mm)

    # ── 4. Chip load ──────────────────────────────────────────────────────
    fz_min, fz_max, fz_mid = _chip_load(tool_diameter_mm)
    if fz_override is not None:
        fz = fz_override
        if fz < fz_min:
            warnings.append(
                f"fz={fz:.4f} below recommended minimum {fz_min:.4f} — "
                f"chips may be dust, accelerating tool wear.")
        elif fz > fz_max:
            warnings.append(
                f"fz={fz:.4f} above recommended maximum {fz_max:.4f} — "
                f"risk of chatter or tool breakage.")
    else:
        fz = fz_mid

    # ── 5. Feed rate ──────────────────────────────────────────────────────
    feed = n * tool_flutes * fz   # F = N·Z·fz  (mm/min)

    # Full-slot warning
    if ae >= tool_diameter_mm * 0.99:
        warnings.append(
            "Full-slot cut (ae ≥ D): consider reducing feed by 25%.")

    # ── 6. MRR ───────────────────────────────────────────────────────────
    mrr = ae * ap * feed          # mm³/min

    # ── 7. Cutting force (per-tooth, worst-case) ──────────────────────────
    kc    = info["kc_mpa"]        # N/mm²
    ac    = fz * ap               # mm²   (chip cross-section)
    fc    = kc * ac               # N     (Fc = Kc·Ac)

    # ── 8. Power ─────────────────────────────────────────────────────────
    # P = Fc·Vc / 60 000   (kW, with Fc in N and Vc in m/min)
    power = (fc * vc) / 60_000.0

    # ── 9. Torque ─────────────────────────────────────────────────────────
    # T = Fc·D / 2 000     (Nm, with Fc in N and D in mm)
    torque = (fc * tool_diameter_mm) / 2_000.0

    # ── 10. Constraint checks ─────────────────────────────────────────────
    power_ok  = power  <= config.MACHINE_POWER_LIMIT_KW
    torque_ok = torque <= config.MACHINE_TORQUE_LIMIT_NM

    if not power_ok:
        warnings.append(
            f"Power {power:.3f} kW exceeds machine limit "
            f"{config.MACHINE_POWER_LIMIT_KW} kW. "
            f"Reduce ae, ap, or feed.")
    if not torque_ok:
        warnings.append(
            f"Torque {torque:.2f} Nm exceeds machine limit "
            f"{config.MACHINE_TORQUE_LIMIT_NM} Nm. "
            f"Reduce ap or fz.")

    # ── 11. Entry strategy feeds ──────────────────────────────────────────
    plunge_feed  = feed * _PLUNGE_FRACTION
    ramp_feed    = feed * math.cos(math.radians(ramp_angle_deg))
    helix_feed   = feed * _HELIX_FRACTION
    helix_dia    = tool_diameter_mm * _HELIX_DIA_FACTOR
    helix_pitch  = _HELIX_PITCH_MM

    return FeedsSpeedsResult(
        material           = material,
        tool               = tool,
        cut                = cut,
        sfm_used           = sfm,
        cutting_speed_mpm  = vc,
        rpm                = n,
        chip_load_mm       = fz,
        feed_rate_mmpm     = feed,
        mrr_mm3pm          = mrr,
        chip_area_mm2      = ac,
        cutting_force_n    = fc,
        power_kw           = power,
        torque_nm          = torque,
        power_ok           = power_ok,
        torque_ok          = torque_ok,
        warnings           = warnings,
        plunge_feed_mmpm   = plunge_feed,
        ramp_feed_mmpm     = ramp_feed,
        ramp_angle_deg     = ramp_angle_deg,
        helix_feed_mmpm    = helix_feed,
        helix_diameter_mm  = helix_dia,
        helix_pitch_mm     = helix_pitch,
    )


def compute_for_face(
    face,
    material:        str   = config.DEFAULT_MATERIAL,
    tool_diameter_mm: float = config.DEFAULT_TOOL_DIAMETER_MM,
    tool_flutes:     int   = config.DEFAULT_TOOL_FLUTES,
    tool_type:       str   = config.DEFAULT_TOOL_TYPE,
    ap_fraction:     float = 0.5,
    ae_fraction:     float = 0.4,
    **kwargs,
) -> FeedsSpeedsResult:
    """
    Convenience wrapper: compute feeds/speeds for a TopFaceFeature from Phase 1.

    ap defaults to ap_fraction × tool_diameter (configurable).
    ae defaults to ae_fraction × tool_diameter (configurable).
    Any extra kwargs are forwarded to compute().
    """
    ap = ap_fraction * tool_diameter_mm
    ae = ae_fraction * tool_diameter_mm
    return compute(
        material         = material,
        tool_diameter_mm = tool_diameter_mm,
        tool_flutes      = tool_flutes,
        tool_type        = tool_type,
        axial_depth_mm   = ap,
        radial_depth_mm  = ae,
        **kwargs,
    )

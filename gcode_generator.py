"""
gcode_generator.py — Phase 4 of the CNC Factory pipeline.

Responsibilities
----------------
1. Convert a Phase 3 ToolpathResult into a complete, executable G-code program.
2. Produce output that is directly consumable by the CNC simulator in
       /Users/sbedi/Nextcloud/Python/Solid/random_solids/Simulator/
   via  CncSimClient.program(lines)  from cnc_llm_client.py.
3. Validate the generated program for structural correctness.

Simulator compatibility notes (from cnc_sim/gcode.py / core.py)
-----------------------------------------------------------------
- Use G00 / G01 (two-character codes, upper-case, no space between G and digit).
- Comments must be in (parentheses). Semicolons and % are silently ignored
  but NOT stripped — use (parentheses) only.
- Modal F: set on first G01; re-emit only when the value changes.
- Spindle: S{rpm} M3 to start, M5 to stop.
- Simulator default max_rpm = 4000.  The computed engineering RPM may exceed this.
  Pass spindle_rpm_cap=4000 (default) to clamp the S word to the simulator limit.
  When running on a real machine or a re-configured simulator, use spindle_rpm_cap=None.
- TOOL SHAPE and TOOL SIZE commands configure the tool in the simulator.
- M30 terminates the program.

G-code structure
----------------
  (header comments)
  TOOL SHAPE {flat|bull|ball}
  TOOL SIZE O{R_o} I{R_i} L{Lt}
  G21            (mm)
  G90            (absolute)
  S{rpm} M3      (spindle on)
  (--- Face N ---) repeated for each face
    G00 X.. Y.. Z..   approach rapid
    G01 X.. Y.. Z.. F{plunge}  entry plunge / lead-in
    G01 ...            raster passes + links
    G00 X.. Y.. Z..   retract rapid
  M5             (spindle off)
  M30            (end of program)

Sources
-------
CNC_SIM_COMMANDS.md, cnc_llm_client.py, cnc_sim/gcode.py, cnc_sim/core.py
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config
from feeds_speeds_engine import FeedsSpeedsResult
from toolpath_planner import ToolpathResult, FaceToolpath, ToolpathPass, Waypoint


# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_SPINDLE_RPM_CAP = 4000.0    # simulator default max_rpm
_DEFAULT_TOOL_LT_MM      = 50.0      # tool stickout length (mm)
_COORD_TOL               = 1e-4      # duplicate-position tolerance (mm)
_COORD_DECIMALS          = 3         # decimal places for X/Y/Z
_FEED_DECIMALS           = 0         # decimal places for F (integer mm/min)
_RPM_DECIMALS            = 0         # decimal places for S


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GCodeResult:
    """
    Complete G-code program produced by generate().
    """
    seed:                  object
    program_number:        int
    lines:                 list      # list[str] — the full program
    line_count:            int
    motion_line_count:     int       # G00 + G01 lines only
    n_faces:               int
    n_active_faces:        int       # faces with n_passes > 0
    total_path_length_mm:  float
    estimated_time_s:      float
    spindle_rpm_emitted:   float     # actual S word value used (may be capped)
    validation_errors:     list = field(default_factory=list)
    warnings:              list = field(default_factory=list)

    def to_string(self) -> str:
        """Return program as a single newline-joined string."""
        return "\n".join(self.lines)

    def save(self, filepath: str) -> None:
        """Write program to a .nc / .gcode file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as fh:
            fh.write(self.to_string())
            fh.write("\n")


# ── Internal writer ───────────────────────────────────────────────────────────

class _Writer:
    """
    Accumulates G-code lines with modal-state tracking.

    Modal state tracked
    -------------------
    current_pos   : [x, y, z] last emitted coordinate (or None)
    current_f     : last emitted F value (None = not yet set)
    motion_count  : number of G00/G01 lines emitted
    """

    def __init__(self):
        self._lines:       list  = []
        self.current_pos:  Optional[list] = None   # [x, y, z]
        self.current_f:    Optional[float] = None
        self.motion_count: int = 0

    @property
    def lines(self) -> list:
        return self._lines

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fmt_coord(self, v: float) -> str:
        return f"{v:.{_COORD_DECIMALS}f}"

    def _same_pos(self, x: float, y: float, z: float) -> bool:
        if self.current_pos is None:
            return False
        cx, cy, cz = self.current_pos
        return (abs(x - cx) < _COORD_TOL and
                abs(y - cy) < _COORD_TOL and
                abs(z - cz) < _COORD_TOL)

    # ── emit primitives ───────────────────────────────────────────────────────

    def blank(self):
        self._lines.append("")

    def comment(self, text: str):
        self._lines.append(f"({text})")

    def raw(self, text: str):
        self._lines.append(text)

    def rapid(self, x: float, y: float, z: float) -> bool:
        """Emit G00.  Returns False and skips if position unchanged."""
        if self._same_pos(x, y, z):
            return False
        xf, yf, zf = self._fmt_coord(x), self._fmt_coord(y), self._fmt_coord(z)
        self._lines.append(f"G00 X{xf} Y{yf} Z{zf}")
        self.current_pos = [x, y, z]
        self.motion_count += 1
        return True

    def feed(self, x: float, y: float, z: float, f: float) -> bool:
        """
        Emit G01 with modal F handling.
        Returns False and skips if position unchanged.
        """
        if self._same_pos(x, y, z):
            return False
        xf, yf, zf = self._fmt_coord(x), self._fmt_coord(y), self._fmt_coord(z)

        if self.current_f is None or abs(f - self.current_f) > 0.5:
            line = f"G01 X{xf} Y{yf} Z{zf} F{f:.{_FEED_DECIMALS}f}"
            self.current_f = f
        else:
            line = f"G01 X{xf} Y{yf} Z{zf}"

        self._lines.append(line)
        self.current_pos = [x, y, z]
        self.motion_count += 1
        return True

    def spindle_on(self, rpm: float):
        self._lines.append(f"S{rpm:.{_RPM_DECIMALS}f} M3")

    def spindle_off(self):
        self._lines.append("M5")

    def end_program(self):
        self._lines.append("M30")


# ── per-pass emitter ──────────────────────────────────────────────────────────

def _emit_pass(writer: _Writer, tp: ToolpathPass) -> None:
    """Emit all waypoints in a ToolpathPass as G00 / G01 lines."""
    for wp in tp.waypoints:
        if not math.isfinite(wp.x) or not math.isfinite(wp.y) or not math.isfinite(wp.z):
            continue  # skip NaN/inf waypoints
        if wp.feed_rate == 0.0 or wp.move_type == "rapid":
            writer.rapid(wp.x, wp.y, wp.z)
        else:
            writer.feed(wp.x, wp.y, wp.z, wp.feed_rate)


# ── public API ────────────────────────────────────────────────────────────────

def generate(
    toolpath_result:    ToolpathResult,
    fs_result:          FeedsSpeedsResult,
    program_number:     int            = 1,
    spindle_rpm_cap:    Optional[float] = _DEFAULT_SPINDLE_RPM_CAP,
    tool_lt_mm:         float          = _DEFAULT_TOOL_LT_MM,
) -> GCodeResult:
    """
    Convert a Phase 3 ToolpathResult into a complete G-code program.

    Parameters
    ----------
    toolpath_result   : output from toolpath_planner.plan()
    fs_result         : output from feeds_speeds_engine.compute()
    program_number    : reference number embedded in comments
    spindle_rpm_cap   : clamp S word to this value for simulator compatibility.
                        None = use computed RPM unchanged.
    tool_lt_mm        : tool length (L parameter for TOOL SIZE command)

    Returns
    -------
    GCodeResult
    """
    warnings: list[str] = []

    # ── Spindle RPM ───────────────────────────────────────────────────────────
    rpm_raw    = fs_result.rpm
    rpm_emit   = rpm_raw
    if spindle_rpm_cap is not None and rpm_raw > spindle_rpm_cap:
        warnings.append(
            f"Spindle RPM {rpm_raw:.0f} exceeds cap {spindle_rpm_cap:.0f} — "
            f"emitting S{spindle_rpm_cap:.0f}.  "
            f"Launch simulator with a higher max_rpm for real speeds."
        )
        rpm_emit = spindle_rpm_cap

    # ── Tool geometry ─────────────────────────────────────────────────────────
    tool_shape = fs_result.tool.type          # flat | bull | ball
    tool_Ro    = fs_result.tool.radius_mm     # outer radius
    tool_Ri    = 0.0 if tool_shape == "flat" else tool_Ro * 0.46   # corner radius
    tool_Lt    = tool_lt_mm

    # ── Face accounting ───────────────────────────────────────────────────────
    active_faces = [ft for ft in toolpath_result.faces if ft.n_passes > 0]

    w = _Writer()

    # ── 1. Header ─────────────────────────────────────────────────────────────
    w.comment(f"CNC Factory — Seed {toolpath_result.seed}  Program O{program_number:04d}")
    w.comment(f"Material : {fs_result.material}")
    w.comment(f"Tool     : D{fs_result.tool.diameter_mm:.1f}mm  "
              f"{fs_result.tool.flutes}fl  {fs_result.tool.type}")
    w.comment(f"RPM      : {rpm_raw:.0f}  (emitting S{rpm_emit:.0f})")
    w.comment(f"Feed     : {fs_result.feed_rate_mmpm:.0f} mm/min  "
              f"Plunge: {fs_result.plunge_feed_mmpm:.0f} mm/min")
    w.comment(f"Faces    : {len(toolpath_result.faces)} total  "
              f"{len(active_faces)} active")
    w.comment(f"Path     : {toolpath_result.total_path_length_mm:.1f} mm  "
              f"Est {toolpath_result.total_estimated_time_s:.1f} s")
    w.comment(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w.blank()

    # ── 2. Tool setup (simulator-specific) ───────────────────────────────────
    w.comment("Tool setup (CNC simulator commands)")
    w.raw(f"TOOL SHAPE {tool_shape}")
    w.raw(f"TOOL SIZE O{tool_Ro:.3f} I{tool_Ri:.3f} L{tool_Lt:.3f}")
    w.blank()

    # ── 3. Setup G-codes ──────────────────────────────────────────────────────
    w.raw("G21")   # metric
    w.raw("G90")   # absolute
    w.blank()

    # ── 4. Spindle on ─────────────────────────────────────────────────────────
    w.spindle_on(rpm_emit)
    w.blank()

    # ── 5. Faces ──────────────────────────────────────────────────────────────
    # Each face block is bracketed by structured sentinel comments so that
    # the resume-after-tool-change logic can locate unmachined faces:
    #
    #   (FACE_START face_id=N z=Z.ZZZ passes=P entry=TYPE)
    #   ... G-code for this face ...
    #   (FACE_END face_id=N)
    #
    # These tokens survive any post-processing and allow a line-number scan
    # to find the last FACE_END before the interrupted position.
    for ft in toolpath_result.faces:
        # Structured start sentinel — machine-parseable
        w.comment(
            f"FACE_START face_id={ft.face_id} z={ft.z_height:.3f} "
            f"passes={ft.n_passes} entry={ft.entry_type}"
        )
        # Human-readable detail on next line
        w.comment(
            f"Face {ft.face_id}  z={ft.z_height:.3f}mm  "
            f"passes={ft.n_passes}  entry={ft.entry_type}"
        )

        if ft.n_passes == 0:
            w.comment(f"  (skipped — {', '.join(ft.warnings) if ft.warnings else 'no passes'})")
            w.comment(f"FACE_END face_id={ft.face_id}")
            w.blank()
            continue

        for tp in ft.passes:
            if tp.pass_type in ("approach", "retract"):
                w.comment(tp.pass_type)
            elif tp.pass_type == "entry":
                w.comment(f"entry ({ft.entry_type})")
            elif tp.pass_type == "raster":
                pass   # no per-raster-line comment — keeps file compact
            elif tp.pass_type == "link":
                pass   # silent stay-down link
            _emit_pass(w, tp)

        w.comment(f"FACE_END face_id={ft.face_id}")
        w.blank()

    # ── 6. End of program ─────────────────────────────────────────────────────
    w.spindle_off()
    w.end_program()

    # ── Build result ──────────────────────────────────────────────────────────
    result = GCodeResult(
        seed                 = toolpath_result.seed,
        program_number       = program_number,
        lines                = w.lines,
        line_count           = len(w.lines),
        motion_line_count    = w.motion_count,
        n_faces              = len(toolpath_result.faces),
        n_active_faces       = len(active_faces),
        total_path_length_mm = toolpath_result.total_path_length_mm,
        estimated_time_s     = toolpath_result.total_estimated_time_s,
        spindle_rpm_emitted  = rpm_emit,
        warnings             = warnings + toolpath_result.warnings,
    )
    result.validation_errors = validate(result)
    return result


# ── Validator ─────────────────────────────────────────────────────────────────

_WORD_RE  = re.compile(r"([A-Za-z])([-+]?\d+(?:\.\d+)?)")
_G01_RE   = re.compile(r"\bG01\b")
_G00_RE   = re.compile(r"\bG00\b")
_F_RE     = re.compile(r"\bF([-+]?\d+(?:\.\d+)?)")


def validate(result: GCodeResult) -> list[str]:
    """
    Check structural correctness of the generated program.

    Rules checked
    -------------
    1. Every G01 line has a feed value (modal or explicit).
    2. All X, Y, Z, F values are finite numbers.
    3. M30 is present.
    4. M5 appears before M30.
    5. No NaN or inf in the coordinate stream.

    Returns list of error strings. Empty = valid.
    """
    errors: list[str] = []
    current_f: Optional[float] = None
    has_m30    = False
    has_m5     = False
    m5_line_no = -1
    m30_line_no= -1

    for i, raw_line in enumerate(result.lines, 1):
        # Strip comments  (parentheses)
        line = re.sub(r"\(.*?\)", "", raw_line).strip()
        if not line:
            continue

        # Check for M30 / M5
        if "M30" in line.upper():
            has_m30 = True
            m30_line_no = i
        if "M5" in line.upper() and "M30" not in line.upper():
            has_m5 = True
            m5_line_no = i

        # Extract F if present
        f_match = _F_RE.search(line)
        if f_match:
            f_val = float(f_match.group(1))
            if not math.isfinite(f_val):
                errors.append(f"Line {i}: non-finite F value: {f_val!r}")
            else:
                current_f = f_val

        # Check G01 has feed
        if _G01_RE.search(line):
            if current_f is None:
                errors.append(f"Line {i}: G01 with no prior F word: {raw_line!r}")

        # Check coordinate finiteness
        for letter, val_str in _WORD_RE.findall(line):
            if letter.upper() in ("X", "Y", "Z"):
                try:
                    val = float(val_str)
                    if not math.isfinite(val):
                        errors.append(
                            f"Line {i}: non-finite {letter.upper()} value "
                            f"{val!r} in: {raw_line!r}"
                        )
                except ValueError:
                    errors.append(
                        f"Line {i}: cannot parse {letter}{val_str!r} "
                        f"in: {raw_line!r}"
                    )

    if not has_m30:
        errors.append("Program has no M30 end-of-program command.")
    if not has_m5:
        errors.append("Program has no M5 (spindle off) before M30.")
    if has_m5 and has_m30 and m5_line_no > m30_line_no:
        errors.append(
            f"M5 (line {m5_line_no}) appears AFTER M30 (line {m30_line_no})."
        )

    return errors

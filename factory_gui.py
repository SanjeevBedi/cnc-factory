"""
factory_gui.py — Visual dashboard for the Autonomous CNC Factory
================================================================

Layout  (main window)
─────────────────────
  ┌──────────┬──────────┬──────────────────────┐
  │  M01     │  M02     │  Factory Agent       │
  │  panel   │  panel   │  • Policy selector   │
  │          │          │  • Production KPIs   │
  ├──────────┼──────────┤  • Event log         │
  │  M03     │  M04     │  • [Start][Stop]      │
  │  panel   │  panel   │  • Factory Chat       │
  └──────────┴──────────┴──────────────────────┘

Each machine panel
──────────────────
  ● Status light  Machine ID  Part seed
  [ G-code scroll — live execution ]
  [ Tool crib table ]
  [ 2-D bird's-eye toolpath canvas ]
  [Launch 3D Sim]  [Inject Error]  [Agent Chat ▶]

Tool-breakage scenario
──────────────────────
  1. Click [Inject Error] on any machine.
  2. Choose "Tool Breakage (no replacement in stock)".
  3. Factory detects break → inventory is 0.
  4. Factory sends rich prompt to OpenAI (3 temps).
  5. All 3 candidates + policy scores shown in Chat window.
  6. Best response dispatched and machine state updated.

Simulator connection
────────────────────
  Each machine tries HTTP on ports 8001-8004.
  [Launch 3D Sim] starts cnc_sim_api_server.py in the
  appropriate conda environment as a subprocess.
  Falls back to the built-in 2-D canvas when offline.

Usage
─────
  python factory_gui.py
  python factory_gui.py --openai-key sk-...
  python factory_gui.py --auto           # auto-start simulation
  python factory_gui.py --demo-break M02 5   # break M02/T3 at tick 5
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from cnc_agent import CncAgent, ToolRecord, build_default_crib
from factory_agent import FactoryAgent, FactoryCommand, build_factory_agent
from scheduler import make_job, job_from_gcode
from agent_dialog import AgentConversationWindow, Phase
from pipeline_monitor import PipelineMonitorWindow
from disturbance_engine import (
    REGISTRY, BY_CATEGORY, Category, DisturbanceSpec, GUIField,
    DisturbanceContext, LLMCandidate, ENGINE,
)

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None  # type: ignore

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ── Theme ─────────────────────────────────────────────────────────────────────

BG      = "#0d1117"
PNL     = "#161b22"
ENTRY   = "#0f3460"
FG      = "#c9d1d9"
DIM     = "#6e7681"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
RED     = "#f85149"
ORANGE  = "#e3b341"
PURPLE  = "#bc8cff"

STATUS_COLOR = {
    "idle":             DIM,
    "running":          GREEN,
    "awaiting_factory": ORANGE,
    "stopped":          RED,
    "error":            RED,
    "paused":           ACCENT,
}

# Button spec: (text, command_attr, bg, fg)
# Bright backgrounds, black text — maximum readability
_BTN_SPEC = [
    ("▶  Start",  "start_sim",  "#4caf50", "black"),
    ("⏸  Pause",  "pause_sim",  "#ffb300", "black"),
    ("⏹  Stop",   "stop_sim",   "#ef5350", "black"),
    ("⏭  Step",   "step_sim",   "#42a5f5", "black"),
]

MACHINE_ACCENT = {
    "M01": "#1f6feb",
    "M02": "#238636",
    "M03": "#9e6a03",
    "M04": "#8957e5",
}

SIM_PORTS = {"M01": 8001, "M02": 8002, "M03": 8003, "M04": 8004}
SIM_SERVER = "/Users/sbedi/Nextcloud/Python/Solid/random_solids/Simulator/cnc_sim_api_server.py"
SIM_PYTHON  = "/Users/sbedi/Nextcloud/Python/Solid/random_solids/.conda/bin/python"


# ── Update events (posted by sim thread, consumed by GUI timer) ────────────────

@dataclass
class GuiEvent:
    kind:  str   # machine_update | factory_update | chat_message | gcode_step
    mid:   str   # machine id or "factory"
    data:  dict  = field(default_factory=dict)


# ── 2-D G-code canvas ─────────────────────────────────────────────────────────

class GCodeCanvas(tk.Canvas):
    """Bird's-eye 2-D view of the toolpath with animated tool cursor."""

    MARGIN = 14

    def __init__(self, parent, **kw):
        kw.setdefault("bg", "#0a1628")
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", "#30363d")
        super().__init__(parent, **kw)
        self._segs:   list[tuple] = []   # (x0,y0,x1,y1, is_cut)
        self._done:   int         = 0
        self._tool:   tuple       = (0.0, 0.0)
        self._bounds: tuple       = (0, 200, 0, 200)
        self.bind("<Configure>", lambda _: self._redraw())

    # ── public API ──────────────────────────────────────────────────────────
    def load_gcode(self, lines: list[str]) -> None:
        self._segs, cx, cy = [], 0.0, 0.0
        for ln in lines:
            ln = ln.strip().upper()
            if not (ln.startswith("G0") or ln.startswith("G1")):
                continue
            nx = self._val(ln, "X", cx)
            ny = self._val(ln, "Y", cy)
            is_cut = ln.startswith("G01") or ln.startswith("G1 ")
            if (nx, ny) != (cx, cy):
                self._segs.append((cx, cy, nx, ny, is_cut))
            cx, cy = nx, ny
        if self._segs:
            xs = [s[0] for s in self._segs] + [s[2] for s in self._segs]
            ys = [s[1] for s in self._segs] + [s[3] for s in self._segs]
            pad = 15
            self._bounds = (min(xs)-pad, max(xs)+pad, min(ys)-pad, max(ys)+pad)
        self._done = 0
        self._redraw()

    def advance(self, n: int, tool_xy: tuple) -> None:
        """Mark first n segments as completed; move tool cursor."""
        self._done = n
        self._tool = tool_xy
        self._redraw()

    def reset(self) -> None:
        self._segs, self._done = [], 0
        self._redraw()

    # ── internals ────────────────────────────────────────────────────────────
    @staticmethod
    def _val(line: str, axis: str, default: float) -> float:
        m = re.search(rf"{axis}([-\d.]+)", line)
        return float(m.group(1)) if m else default

    def _px(self, x: float, y: float) -> tuple[float, float]:
        W = max(1, self.winfo_width()  - 2 * self.MARGIN)
        H = max(1, self.winfo_height() - 2 * self.MARGIN)
        xlo, xhi, ylo, yhi = self._bounds
        dx, dy = xhi - xlo or 1, yhi - ylo or 1
        px = self.MARGIN + (x - xlo) / dx * W
        py = self.MARGIN + (1 - (y - ylo) / dy) * H
        return px, py

    def _redraw(self) -> None:
        self.delete("all")
        if not self._segs:
            w, h = self.winfo_width(), self.winfo_height()
            self.create_text(w//2, h//2, text="No job loaded",
                             fill=DIM, font=("Courier", 9))
            return
        for i, (x0, y0, x1, y1, is_cut) in enumerate(self._segs):
            p0, p1 = self._px(x0, y0), self._px(x1, y1)
            if i < self._done:
                color = GREEN if is_cut else "#4488cc"
                width = 2.0 if is_cut else 1.0
            else:
                color = "#555" if is_cut else "#334"
                width = 1.0
            self.create_line(*p0, *p1, fill=color, width=width,
                             dash=None if is_cut else (3, 5))
        # tool cursor
        tx, ty = self._px(*self._tool)
        r = 6
        self.create_oval(tx-r, ty-r, tx+r, ty+r,
                         fill=ACCENT, outline="white", width=1)
        self.create_line(tx-r-3, ty, tx+r+3, ty, fill="white", width=1)
        self.create_line(tx, ty-r-3, tx, ty+r+3, fill="white", width=1)


# ── Chat window ───────────────────────────────────────────────────────────────

class ChatWindow(tk.Toplevel):
    """Popup showing machine ↔ factory agent conversation."""

    def __init__(self, parent, machine_id: str, factory_gui: "FactoryGUI"):
        super().__init__(parent)
        self.title(f"Agent Chat — {machine_id} ↔ Factory")
        self.configure(bg=BG)
        self.geometry("680x520")
        self.resizable(True, True)
        self._mid   = machine_id
        self._app   = factory_gui

        # ── chat log ──────────────────────────────────────────────────────
        self._log = tk.Text(
            self, bg=PNL, fg=FG, font=("Courier", 10),
            wrap="word", state="disabled",
            selectbackground=ENTRY,
        )
        sb = ttk.Scrollbar(self, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        self._log.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        sb.grid(row=0, column=1, sticky="ns", pady=6)

        # configure tags for speaker colours
        self._log.tag_config("machine",  foreground=ACCENT)
        self._log.tag_config("factory",  foreground=GREEN)
        self._log.tag_config("openai",   foreground=PURPLE)
        self._log.tag_config("system",   foreground=DIM)
        self._log.tag_config("best",     foreground=YELLOW)
        self._log.tag_config("error",    foreground=RED)
        self._log.tag_config("meta",     foreground=DIM,    font=("Courier", 9))

        # ── input row ─────────────────────────────────────────────────────
        frm = tk.Frame(self, bg=BG)
        frm.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self._entry = tk.Entry(frm, bg=ENTRY, fg=FG,
                               insertbackground=FG, font=("Courier", 10))
        self._entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._entry.bind("<Return>", self._send)
        tk.Button(frm, text="Inject prompt", bg=MACHINE_ACCENT.get(machine_id, ACCENT),
                  fg="white", relief="flat", padx=8,
                  command=self._send).pack(side="right")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def append(self, speaker: str, text: str, tag: str = "system") -> None:
        self._log.configure(state="normal")
        ts = time.strftime("%H:%M:%S")
        self._log.insert("end", f"[{ts}] ", "meta")
        self._log.insert("end", f"{speaker:10s}  ", tag)
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _send(self, _=None) -> None:
        txt = self._entry.get().strip()
        if not txt:
            return
        self._entry.delete(0, "end")
        self.append(self._mid, f"[manual] {txt}", "machine")
        agent = next((a for a in self._app.fa.agents
                      if a.machine_id == self._mid), None)
        if agent:
            agent.inject_error(txt)
        self._app._eq.put(GuiEvent("manual_error", self._mid, {"desc": txt}))


# ── Category colours ──────────────────────────────────────────────────────────

CAT_COLOR = {
    Category.TOOL:     "#c0392b",
    Category.MATERIAL: "#d35400",
    Category.PROCESS:  "#8e44ad",
    Category.MACHINE:  "#2980b9",
    Category.SCHEDULE: "#27ae60",
}
CAT_ICON = {
    Category.TOOL:     "🔧",
    Category.MATERIAL: "🧱",
    Category.PROCESS:  "⚙",
    Category.MACHINE:  "🔩",
    Category.SCHEDULE: "📅",
}


# ── Disturbance dialog ────────────────────────────────────────────────────────

class DisturbanceDialog(tk.Toplevel):
    """
    Tabbed inject-disturbance dialog.
    One tab per category; selecting a type shows engineering-appropriate
    extra fields (e.g. material dropdowns for material_harder).
    """

    def __init__(self, parent, machine_id: str, callback):
        super().__init__(parent)
        self.title(f"Inject Disturbance — {machine_id}")
        self.configure(bg=BG)
        self.geometry("700x560")
        self.resizable(True, True)
        self._callback   = callback
        self._machine    = machine_id
        self._sel_key: Optional[str] = None
        self._extra_vars: dict[str, tk.Variable] = {}

        # notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._radios: dict[str, tk.StringVar] = {}

        for cat in Category:
            specs = BY_CATEGORY[cat]
            if not specs:
                continue
            tab  = tk.Frame(nb, bg=BG)
            nb.add(tab, text=f"{CAT_ICON[cat]} {cat.value.capitalize()}")
            svar = tk.StringVar(value="")
            self._radios[cat.value] = svar

            for spec in specs:
                color = CAT_COLOR[cat]
                rb = tk.Radiobutton(
                    tab, text=spec.label,
                    variable=svar, value=spec.key,
                    bg=BG, fg=(RED if spec.safety_critical else FG),
                    selectcolor=ENTRY,
                    activebackground=BG, activeforeground=color,
                    command=lambda k=spec.key, c=cat.value: self._on_select(k, c),
                )
                rb.pack(anchor="w", padx=10, pady=1)
                tk.Label(tab, text=f"   {spec.short_desc}",
                         bg=BG, fg=DIM,
                         font=("Helvetica", 8, "italic"),
                         ).pack(anchor="w", padx=14)

        # extra fields
        self._extra_outer = tk.LabelFrame(
            self, text="Disturbance parameters",
            bg=BG, fg=ACCENT, font=("Helvetica", 9),
        )
        self._extra_outer.pack(fill="x", padx=6, pady=2)

        # buttons
        btn = tk.Frame(self, bg=BG)
        btn.pack(fill="x", padx=6, pady=6)
        tk.Button(btn, text="⚠ Inject", bg=RED, fg="white",
                  relief="flat", padx=14,
                  font=("Helvetica", 10, "bold"),
                  command=self._inject).pack(side="left", padx=4)
        tk.Button(btn, text="Cancel", bg=PNL, fg=FG,
                  relief="flat", padx=10,
                  command=self.destroy).pack(side="left")
        self._note = tk.Label(btn, text="", bg=BG, fg=ORANGE,
                               font=("Helvetica", 9))
        self._note.pack(side="left", padx=8)

    def _on_select(self, key: str, cat_val: str) -> None:
        for cv, sv in self._radios.items():
            if cv != cat_val:
                sv.set("")
        self._sel_key = key
        spec = REGISTRY[key]
        self._note.configure(
            text="⚠ SAFETY CRITICAL — emergency stop will be dispatched"
            if spec.safety_critical else "")
        self._rebuild_extra(spec)

    def _rebuild_extra(self, spec: DisturbanceSpec) -> None:
        for w in self._extra_outer.winfo_children():
            w.destroy()
        self._extra_vars.clear()
        if not spec.gui_fields:
            tk.Label(self._extra_outer, text="No additional parameters.",
                     bg=BG, fg=DIM, font=("Helvetica", 9)).pack(padx=8, pady=4)
            return
        for f in spec.gui_fields:
            row = tk.Frame(self._extra_outer, bg=BG)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=f.label+":", bg=BG, fg=FG,
                     font=("Courier", 9), width=30, anchor="e"
                     ).pack(side="left")
            if f.kind == "bool":
                var = tk.BooleanVar(value=bool(f.default))
                tk.Checkbutton(row, variable=var, bg=BG, fg=FG,
                               selectcolor=ENTRY,
                               activebackground=BG).pack(side="left")
            elif f.kind == "choice":
                var = tk.StringVar(value=str(f.default))
                ttk.OptionMenu(row, var, str(f.default),
                               *f.choices).pack(side="left", fill="x", expand=True)
            else:
                var = tk.StringVar(value=str(f.default))
                tk.Entry(row, textvariable=var, bg=ENTRY, fg=FG,
                         insertbackground=FG,
                         font=("Courier", 9), width=32
                         ).pack(side="left", fill="x", expand=True)
            self._extra_vars[f.name] = var

    def _collect_extra(self) -> dict:
        result = {}
        for name, var in self._extra_vars.items():
            val = var.get()
            try:
                result[name] = float(val) if "." in str(val) else int(val)
            except (ValueError, TypeError):
                result[name] = val
        return result

    def _inject(self) -> None:
        if not self._sel_key:
            self._note.configure(text="Select a disturbance type first.")
            return
        spec  = REGISTRY[self._sel_key]
        extra = self._collect_extra()
        desc  = f"[{spec.key}] {spec.label}"
        if extra:
            desc += " | " + "; ".join(f"{k}={v}" for k, v in extra.items())
        self._callback(self._machine, desc,
                       spec_key=self._sel_key, extra=extra)
        self.destroy()


# legacy alias
InjectErrorDialog = DisturbanceDialog


# ── Machine panel ─────────────────────────────────────────────────────────────

class MachinePanel(ttk.Frame):
    """One machine's complete panel."""

    def __init__(self, parent, machine_id: str, app: "FactoryGUI"):
        super().__init__(parent, style="Machine.TFrame")
        self._mid  = machine_id
        self._app  = app
        self._chat: Optional[AgentConversationWindow] = None
        self._sim_proc: Optional[subprocess.Popen] = None

        accent = MACHINE_ACCENT[machine_id]

        # ── header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=accent, height=28)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        self._status_dot = tk.Label(hdr, text="●", fg=DIM,
                                    bg=accent, font=("Helvetica", 12))
        self._status_dot.pack(side="left", padx=6)

        tk.Label(hdr, text=machine_id, fg="white",
                 bg=accent, font=("Helvetica", 11, "bold")).pack(side="left")

        self._part_lbl = tk.Label(hdr, text="Idle",
                                  fg="#aaa", bg=accent, font=("Courier", 9))
        self._part_lbl.pack(side="left", padx=8)

        self._sim_btn = tk.Button(
            hdr, text="3D↗", bg=accent, fg="white",
            relief="flat", font=("Helvetica", 8),
            command=self._launch_sim,
        )
        self._sim_btn.pack(side="right", padx=4)

        # ── tool line ─────────────────────────────────────────────────────
        tl = tk.Frame(self, bg=PNL)
        tl.pack(fill="x", padx=2, pady=1)
        tk.Label(tl, text="Tool:", bg=PNL, fg=DIM,
                 font=("Courier", 8)).pack(side="left", padx=4)
        self._tool_lbl = tk.Label(tl, text="—",
                                   bg=PNL, fg=ACCENT, font=("Courier", 8))
        self._tool_lbl.pack(side="left")
        self._life_bar = ttk.Progressbar(tl, length=80, maximum=100,
                                          mode="determinate")
        self._life_bar.pack(side="left", padx=4)
        self._life_lbl = tk.Label(tl, text="", bg=PNL,
                                   fg=FG, font=("Courier", 8))
        self._life_lbl.pack(side="left")

        # ── g-code scroll ─────────────────────────────────────────────────
        gc_frm = tk.Frame(self, bg=BG)
        gc_frm.pack(fill="x", padx=2)
        tk.Label(gc_frm, text="G-code", bg=BG,
                 fg=DIM, font=("Courier", 8)).pack(anchor="w", padx=4)
        self._gcode_txt = tk.Text(
            gc_frm, height=5, bg="#070d15", fg="#a0c0a0",
            font=("Courier", 9), state="disabled",
            selectbackground=ENTRY,
            insertbackground=FG,
        )
        self._gcode_txt.pack(fill="x", padx=2)
        self._gcode_txt.tag_config("cur", background="#1e3a5f", foreground="white")
        self._gcode_txt.tag_config("done", foreground=DIM)

        # ── 2-D canvas ────────────────────────────────────────────────────
        self._canvas = GCodeCanvas(self, width=220, height=130)
        self._canvas.pack(fill="both", padx=2, pady=2)
        self._canvas_lines_id: int = 0   # id() of last list passed to load_gcode

        # ── tool crib mini-table ──────────────────────────────────────────
        cols = ("Tool", "⌀mm", "Life%", "Warn")
        self._crib_tree = ttk.Treeview(self, columns=cols,
                                        show="headings", height=4)
        for c, w in zip(cols, (40, 40, 45, 40)):
            self._crib_tree.heading(c, text=c)
            self._crib_tree.column(c, width=w, anchor="center")
        self._crib_tree.pack(fill="x", padx=2, pady=2)
        self._crib_tree.tag_configure("warn", foreground=ORANGE)
        self._crib_tree.tag_configure("crit", foreground=RED)

        # ── bottom buttons ────────────────────────────────────────────────
        btn = tk.Frame(self, bg=BG)
        btn.pack(fill="x", padx=2, pady=4)
        tk.Button(
            btn, text="⚠ Inject Disturbance", bg="#2d1b1b", fg=ORANGE,
            relief="flat", font=("Helvetica", 9),
            command=lambda: DisturbanceDialog(
                self._app, self._mid, self._app.inject_disturbance
            ),
        ).pack(side="left", padx=3)
        tk.Button(
            btn, text="💬 Agent Chat", bg="#1b2d1b", fg=GREEN,
            relief="flat", font=("Helvetica", 9),
            command=self._open_chat,
        ).pack(side="left", padx=3)

    # ── public update ─────────────────────────────────────────────────────

    def set_machining_label(self, seed, status_str: str = "▶") -> None:
        """Update the header part-label during machining."""
        self._part_lbl.configure(
            text=f"{status_str} Seed {seed}", fg="white")

    def set_idle_label(self) -> None:
        self._part_lbl.configure(text="— Idle", fg="#aaa")

    def set_unclamp_label(self, seed, steps_left: int) -> None:
        self._part_lbl.configure(
            text=f"⏳ Unloading {seed}  ({steps_left} steps)", fg=ORANGE)

    def update_state(self, state: dict) -> None:
        status = state.get("status", "idle")
        color  = STATUS_COLOR.get(status, DIM)
        self._status_dot.configure(fg=color)
        seed = state.get("current_seed") or state.get("current_job")
        # Label driven by _do_sim_step via set_*_label calls;
        # only set from state when we don’t already have richer info
        if seed and self._part_lbl.cget("text") in ("— Idle", "Idle", ""):
            self._part_lbl.configure(text=f"▶ Seed {seed}", fg="white")
        elif not seed:
            self.set_idle_label()

        # tool crib
        for row in self._crib_tree.get_children():
            self._crib_tree.delete(row)
        min_life = 100.0
        for t in state.get("tool_crib", []):
            life = t.get("remaining_life_pct", 100.0)
            min_life = min(min_life, life)
            warn  = t.get("needs_replacement", False)
            tags  = ("crit",) if warn else (("warn",) if life < 30 else ())
            self._crib_tree.insert("", "end",
                values=(t["tool_id"], f'{t["diameter_mm"]:.0f}',
                        f"{life:.0f}", "⚠" if warn else ""),
                tags=tags)
        self._life_bar["value"] = min_life
        self._life_lbl.configure(text=f"{min_life:.0f}%",
                                  fg=(RED if min_life < 20 else
                                      ORANGE if min_life < 40 else GREEN))

    def update_gcode(self, lines: list[str], cursor: int) -> None:
        if not lines:
            # Job finished or no job — clear everything
            if self._canvas_lines_id != 0:
                self._canvas.reset()
                self._canvas_lines_id = 0
            self._gcode_txt.configure(state="normal")
            self._gcode_txt.delete("1.0", "end")
            self._gcode_txt.configure(state="disabled")
            return
        # Only call load_gcode when a new job starts (new list object)
        lid = id(lines)
        if lid != self._canvas_lines_id:
            self._canvas.load_gcode(lines)
            self._canvas_lines_id = lid
        self._canvas.advance(cursor, self._extract_pos(lines, cursor))
        # Scroll the G-code text view
        self._gcode_txt.configure(state="normal")
        self._gcode_txt.delete("1.0", "end")
        start = max(0, cursor - 3)
        end   = min(len(lines), cursor + 6)
        for i in range(start, end):
            tag = "cur" if i == cursor else ("done" if i < cursor else "")
            self._gcode_txt.insert("end", lines[i] + "\n", tag)
        self._gcode_txt.configure(state="disabled")

    def update_tool_label(self, tool_id: str, dia: float) -> None:
        self._tool_lbl.configure(text=f"{tool_id} ⌀{dia:.0f}mm")

    def append_chat(self, speaker_key: str, text: str,
                    tag: str = "system") -> None:
        if self._chat and self._chat.winfo_exists():
            self._chat.append(speaker_key, text)

    # ── internals ─────────────────────────────────────────────────────────
    @staticmethod
    def _extract_pos(lines: list[str], idx: int) -> tuple[float, float]:
        x, y = 0.0, 0.0
        for i in range(min(idx + 1, len(lines))):
            ln = lines[i].upper()
            if "X" in ln:
                m = re.search(r"X([-\d.]+)", ln)
                if m: x = float(m.group(1))
            if "Y" in ln:
                m = re.search(r"Y([-\d.]+)", ln)
                if m: y = float(m.group(1))
        return x, y

    def _open_chat(self) -> None:
        if self._chat is None or not self._chat.winfo_exists():
            self._chat = AgentConversationWindow(self._app, self._mid, self._app)
        else:
            self._chat.open()

    def _launch_sim(self) -> None:
        port = SIM_PORTS[self._mid]
        if not os.path.exists(SIM_SERVER):
            messagebox.showwarning("Simulator",
                f"Server not found:\n{SIM_SERVER}")
            return
        if self._sim_proc and self._sim_proc.poll() is None:
            messagebox.showinfo("Simulator",
                f"Simulator for {self._mid} already running (port {port})")
            return
        python_exec = SIM_PYTHON if os.path.exists(SIM_PYTHON) else sys.executable
        try:
            self._sim_proc = subprocess.Popen(
                [python_exec, SIM_SERVER,
                 "--port", str(port), "--with-ui",
                 "--host", "127.0.0.1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._sim_btn.configure(text="3D●", fg=GREEN)
            messagebox.showinfo("Simulator",
                f"{self._mid} simulator starting on port {port}\n"
                f"Connect: http://127.0.0.1:{port}")
        except Exception as e:
            messagebox.showerror("Simulator", str(e))

    def destroy(self) -> None:
        if self._sim_proc and self._sim_proc.poll() is None:
            self._sim_proc.terminate()
        super().destroy()


# ── Factory panel ─────────────────────────────────────────────────────────────

class FactoryPanel(ttk.Frame):
    def __init__(self, parent, app: "FactoryGUI"):
        super().__init__(parent, style="Factory.TFrame")
        self._app = app

        hdr = tk.Frame(self, bg="#1c2128", height=28)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🏭  FACTORY AGENT",
                 bg="#1c2128", fg=ACCENT,
                 font=("Helvetica", 11, "bold")).pack(side="left", padx=8)

        # policy selector
        pf = tk.Frame(self, bg=PNL)
        pf.pack(fill="x", padx=4, pady=4)
        tk.Label(pf, text="Policy:", bg=PNL, fg=DIM,
                 font=("Courier", 9)).pack(side="left", padx=4)
        self._policy_var = tk.StringVar(value="min_time")
        policies = ["min_time", "min_cost", "best_finish", "max_tool_life"]
        om = ttk.OptionMenu(pf, self._policy_var, "min_time", *policies,
                            command=self._on_policy)
        om.pack(side="left")

        # KPI grid
        kf = tk.Frame(self, bg=PNL)
        kf.pack(fill="x", padx=4, pady=2)
        self._kpi: dict[str, tk.Label] = {}
        kpis = [
            ("Jobs done",     "total_jobs_completed"),
            ("Rework",        "rework_queue_depth"),
            ("Errors",        "total_errors_processed"),
            ("Tools replaced","total_tools_replaced"),
            ("Avg tool life", "avg_tool_life_pct"),
            ("Tick",          "tick"),
        ]
        for i, (label, key) in enumerate(kpis):
            r, c = divmod(i, 2)
            tk.Label(kf, text=label+":",
                     bg=PNL, fg=DIM, font=("Courier", 9)
                     ).grid(row=r, column=c*2,   sticky="e", padx=3)
            lbl = tk.Label(kf, text="—",
                           bg=PNL, fg=ACCENT, font=("Courier", 9, "bold"))
            lbl.grid(row=r, column=c*2+1, sticky="w", padx=2)
            self._kpi[key] = lbl

        # event log (wrapped in frame so we can add completed-parts below)
        log_frm = tk.Frame(self, bg=BG)
        log_frm.pack(fill="both", expand=True, padx=0, pady=0)

        tk.Label(log_frm, text="Event log", bg=BG, fg=DIM,
                 font=("Courier", 8)).pack(anchor="w", padx=6, pady=(4,0))
        log_inner = tk.Frame(log_frm, bg=BG)
        log_inner.pack(fill="both", expand=True, padx=6, pady=2)
        self._log = tk.Text(
            log_inner, bg="#0a1628", fg=FG, font=("Courier", 9),
            height=8, state="disabled", wrap="word",
        )
        sb = ttk.Scrollbar(log_inner, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        self._log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._log.tag_config("error",   foreground=RED)
        self._log.tag_config("warn",    foreground=ORANGE)
        self._log.tag_config("ok",      foreground=GREEN)
        self._log.tag_config("llm",     foreground=PURPLE)
        self._log.tag_config("tool",    foreground=YELLOW)
        self._log.tag_config("dim",     foreground=DIM)

        # ── Completed parts log ─────────────────────────────────────────
        sep = tk.Frame(self, bg=DIM, height=1)
        sep.pack(fill="x", padx=4, pady=(2, 0))

        hdr_frm = tk.Frame(self, bg=BG)
        hdr_frm.pack(fill="x", padx=6, pady=(3, 0))
        tk.Label(hdr_frm, text="Completed Parts", bg=BG, fg=DIM,
                 font=("Courier", 8, "bold")).pack(side="left")
        self._parts_count_lbl = tk.Label(hdr_frm, text="0 parts",
                                         bg=BG, fg=ACCENT,
                                         font=("Courier", 8))
        self._parts_count_lbl.pack(side="right")

        parts_frm = tk.Frame(self, bg=BG)
        parts_frm.pack(fill="x", padx=6, pady=(2, 4))

        style = ttk.Style()
        style.configure("Parts.Treeview",
                         background="#0a1628", foreground=FG,
                         fieldbackground="#0a1628",
                         font=("Courier", 8), rowheight=16)
        style.configure("Parts.Treeview.Heading",
                         background=PNL, foreground=DIM,
                         font=("Courier", 8, "bold"))

        _cols = ("Seed", "Machine", "Lines", "Mach.s", "Total.s", "Load.s", "Unload.s")
        self._parts_tree = ttk.Treeview(
            parts_frm, columns=_cols, show="headings",
            height=5, style="Parts.Treeview",
        )
        col_widths = {"Seed": 55, "Machine": 48, "Lines": 46,
                      "Mach.s": 52, "Total.s": 52, "Load.s": 48, "Unload.s": 55}
        for c in _cols:
            self._parts_tree.heading(c, text=c)
            self._parts_tree.column(c, width=col_widths.get(c, 60),
                                    anchor="center", stretch=False)
        parts_sb = ttk.Scrollbar(parts_frm, orient="vertical",
                                  command=self._parts_tree.yview)
        self._parts_tree.configure(yscrollcommand=parts_sb.set)
        self._parts_tree.pack(side="left", fill="x", expand=True)
        parts_sb.pack(side="right", fill="y")

        # control buttons (packed by parent — see FactoryGUI)

    def update_kpis(self, kpis: dict, tick: int) -> None:
        for key, lbl in self._kpi.items():
            val = kpis.get(key, tick if key == "tick" else "—")
            if key == "avg_tool_life_pct":
                lbl.configure(text=f"{val:.1f}%",
                               fg=(RED if val < 20 else
                                   ORANGE if val < 40 else ACCENT))
            else:
                lbl.configure(text=str(val))

    def add_completed_part(self, stats: dict) -> None:
        """Insert one row at the top of the completed-parts treeview."""
        row = (
            str(stats.get("seed", "?")),
            stats.get("machine", "?"),
            str(stats.get("gcode_lines", 0)),
            f"{stats.get('machining_s', 0):.0f}",
            f"{stats.get('total_s', 0):.0f}",
            f"{stats.get('load_s', 0):.0f}",
            f"{stats.get('unload_s', 0):.0f}",
        )
        self._parts_tree.insert("", 0, values=row)
        # Keep only the last 200 rows
        children = self._parts_tree.get_children()
        if len(children) > 200:
            self._parts_tree.delete(children[-1])
        count = len(self._parts_tree.get_children())
        self._parts_count_lbl.configure(text=f"{count} part{'s' if count != 1 else ''}")

    def log(self, text: str, tag: str = "") -> None:
        self._log.configure(state="normal")
        ts = time.strftime("%H:%M:%S")
        self._log.insert("end", f"{ts}  ", "dim")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _on_policy(self, val: str) -> None:
        self._app.fa.policy    = val
        self._app.fa.scheduler.policy = val
        self.log(f"Policy → {val}", "ok")


# ── Main application ──────────────────────────────────────────────────────────

class FactoryGUI(tk.Tk):

    def __init__(self, openai_key: Optional[str] = None,
                 auto_start: bool = False,
                 demo_break: Optional[tuple[str, int]] = None):
        # Use project-local key from factory_secrets.ini if no key supplied
        if not openai_key:
            openai_key = config.OPENAI_API_KEY or None
        super().__init__()
        self.title("Autonomous CNC Factory — Visual Dashboard")
        self.configure(bg=BG)
        self.geometry("1420x820")
        self.resizable(True, True)

        self._openai_key  = openai_key
        self._demo_break  = demo_break   # (machine_id, tick)
        self._running     = False
        # ── Base time loop ─────────────────────────────────────────────
        # No fixed DT -- each G-code line sleeps dt_sim / compression / speed.
        # Three counters driven by the same base tick (SIM_TICK_S = 1 sim-s).
        self._speed:          float = 1.0
        self._sim_step_count: int   = 0
        self._gui_acc:        float = 0.0   # simulated-s since last GUI refresh
        self._seed_acc:       float = 0.0   # simulated-s since last seed offer
        self._eq: queue.Queue[GuiEvent]  = queue.Queue()
        self._gcode_cursors: dict[str, int]        = {}
        self._gcode_lines:   dict[str, list[str]]  = {}
        self._gcode_job_id:  dict[str, str]        = {}   # mid → job_id driving canvas
        self._chat_wins:     dict[str, AgentConversationWindow] = {}
        self._pipeline_wins: dict[int, PipelineMonitorWindow]   = {}  # seed → window

        # Animation queues — one deque per machine; GUI pops to animate
        self._anim_queues: dict[str, deque] = {}
        self._anim_seen:   set[str]         = set()  # job_ids already queued

        # Unclamp (part-removal) phase — tracks countdown per machine
        # dict: mid → {"job": job, "steps": int}
        self._unclamp: dict[str, Optional[dict]] = {}
        # Steps for visual unclamp phase (proportional to PART_REMOVAL_TIME_S)
        # Unclamp and setup durations in sim-steps.
        # Each step represents T_avg/L_avg seconds of machining time,
        # so these express the physical durations relative to the job length.
        # UNCLAMP = PART_REMOVAL_TIME_S / (T_avg/L_avg)
        #         = PART_REMOVAL_TIME_S * L_avg / T_avg
        # Use a fixed ratio: removal is ~1.5% of machining time  (5/340)
        # giving  round(500 * 5/340) = 7 steps.
        _T_avg_s  = 324.5
        _L_avg    = 500
        self._UNCLAMP_STEPS: int = max(3, round(
            config.PART_REMOVAL_TIME_S * _L_avg / _T_avg_s))

        # Indexed solid library — built once on first Load Seeds press.
        # Index i → seed value.  Random index selection allows repeats.
        self._seed_index: list[int] = []   # [seed0, seed1, …, seedN]
        self._seed_max_idx: int     = -1   # len(_seed_index) - 1


        # Completed parts log
        self._completed_parts: list[dict] = []

        # Parts queue display window (single window, opened on demand)
        self._queue_win: Optional["PartsQueueWindow"] = None

        # Track all seeds ever submitted this session (for queue window)
        self._all_jobs: list[dict] = []   # {seed, mid, status, lines, job_id}

        # build ttk styles
        self._build_styles()

        # build factory
        self.fa = build_factory_agent(
            n_machines=4, policy="min_time",
            openai_api_key=openai_key, dry_run=True,
        )
        for ag in self.fa.agents:
            self._gcode_cursors[ag.machine_id] = 0
            self._gcode_lines[ag.machine_id]   = []
            self._gcode_job_id[ag.machine_id]  = ""
            self._anim_queues[ag.machine_id]   = deque()
            self._unclamp[ag.machine_id]       = None

        # build layout
        self._build_layout()
        self._initial_populate()

        # start poll
        self.after(150, self._poll)

        if auto_start:
            self.after(400, self.start_sim)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_styles(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Machine.TFrame", background=BG)
        s.configure("Factory.TFrame", background=PNL)
        s.configure("Treeview",
                    background=PNL, fieldbackground=PNL,
                    foreground=FG, rowheight=16)
        s.configure("Treeview.Heading",
                    background=ENTRY, foreground=ACCENT)
        s.map("Treeview", background=[("selected", ACCENT)])
        s.configure("TProgressbar",
                    troughcolor="#222", background=GREEN)

    def _build_layout(self) -> None:
        # ── top control bar ────────────────────────────────────────────────
        top = tk.Frame(self, bg="#0d1117", height=36)
        top.pack(fill="x", padx=0, pady=0)
        top.pack_propagate(False)

        tk.Label(top, text="🏭 Autonomous CNC Factory",
                 bg="#0d1117", fg=ACCENT,
                 font=("Helvetica", 13, "bold")).pack(side="left", padx=10)

        self._tick_lbl = tk.Label(top, text="tick —",
                                   bg="#0d1117", fg=DIM,
                                   font=("Courier", 10))
        self._tick_lbl.pack(side="left", padx=12)

        # ── Time-mapping speed control ──────────────────────────────────
        # Speed slider: linear multiplier 0.1× … 10× real-time.
        # 1× = one G-code line animated per BASE_MS_PER_LINE ms of wall time.
        # BASE_MS_PER_LINE is derived from the typical job
        # (estimated_time_s / n_lines × 1000).  For our seeds: 30s / 20 = 1500 ms.
        # At 1×: step_ms = 1500 ms  →  20-line job takes 30 s real.
        # At 10×: step_ms = 150 ms  →  same job takes  3 s real.
        # At 0.1×: step_ms = 15000 ms → same job takes 300 s real.
        # Speed slider: 0.1× … 10×  (linear, direct multiplier of 1/dt)
        # Label shows current speed and the resulting dt.
        self._speed_var = tk.DoubleVar(value=1.0)
        self._speed_lbl = tk.Label(top, text=f"1.0×",  # updated by _on_speed
                                   bg="#0d1117", fg=DIM, font=("Courier", 9))
        self._speed_lbl.pack(side="right", padx=(0, 2))
        tk.Label(top, text="speed:", bg="#0d1117", fg=DIM,
                 font=("Courier", 8)).pack(side="right", padx=(4, 0))
        sc = ttk.Scale(top, variable=self._speed_var,
                       from_=0.1, to=10.0, orient="horizontal", length=120,
                       command=self._on_speed)
        sc.pack(side="right", padx=2)

        # control buttons — dark backgrounds, white text for contrast
        for txt, attr, bg, fg in _BTN_SPEC:
            tk.Button(top, text=txt, bg=bg, fg=fg,
                      relief="flat", font=("Helvetica", 9, "bold"), padx=10,
                      activebackground=bg, activeforeground=fg,
                      command=getattr(self, attr)
                      ).pack(side="right", padx=3, pady=4)

        # LLM status indicator (replaces 🔑 button — key is auto-loaded)
        llm_txt  = ("● OpenAI" if self._openai_key else "○ LLM off")
        llm_fg   = GREEN if self._openai_key else DIM
        self._llm_lbl = tk.Label(top, text=llm_txt,
                                  bg="#0d1117", fg=llm_fg,
                                  font=("Courier", 9))
        self._llm_lbl.pack(side="right", padx=8)

        tk.Button(top, text="🌱 Load Seeds",
                  bg="#1a4a1a", fg="#7fff7f",
                  relief="flat", font=("Helvetica", 9, "bold"), padx=8,
                  activebackground="#1a4a1a", activeforeground="#7fff7f",
                  command=self._prompt_load_seeds).pack(side="right", padx=6)

        tk.Button(top, text="📋 Queue",
                  bg="#e0e0e0", fg="black",
                  relief="flat", font=("Helvetica", 9, "bold"), padx=8,
                  activebackground="#bdbdbd", activeforeground="black",
                  command=self._open_queue_window).pack(side="right", padx=6)

        # ── main area ─────────────────────────────────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=4, pady=4)
        main.columnconfigure(0, weight=1, minsize=320)
        main.columnconfigure(1, weight=1, minsize=320)
        main.columnconfigure(2, weight=0, minsize=280)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        self._panels: dict[str, MachinePanel] = {}
        positions = {"M01":(0,0), "M02":(0,1), "M03":(1,0), "M04":(1,1)}
        for mid, (r, c) in positions.items():
            p = MachinePanel(main, mid, self)
            p.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            self._panels[mid] = p

        self._fpanel = FactoryPanel(main, self)
        self._fpanel.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=2, pady=2)

        # ── status bar ────────────────────────────────────────────────────
        sb = tk.Frame(self, bg="#090d13", height=22)
        sb.pack(fill="x")
        sb.pack_propagate(False)
        self._status_bar = tk.Label(
            sb, text="Ready — press ▶ Start",
            bg="#090d13", fg=DIM, font=("Courier", 9), anchor="w",
        )
        self._status_bar.pack(fill="x", padx=8)

    def _initial_populate(self) -> None:
        for mid, panel in self._panels.items():
            ag    = next(a for a in self.fa.agents if a.machine_id == mid)
            state = ag.get_state()
            panel.update_state(state)
        kpis = self.fa.get_production_kpis()
        self._fpanel.update_kpis(kpis, self.fa.scheduler.tick)

    # ── Simulation control ────────────────────────────────────────────────────
    # ── Seed loading ────────────────────────────────────────────────────────────

    def _prompt_load_seeds(self) -> None:
        """
        Build the indexed solid library on first call, then ask which seeds
        to load initially.  Repeats are always allowed — the same solid can
        be machined any number of times.
        """
        import tkinter.simpledialog as sd
        import cnc_solid_bridge as bridge

        # ── Build (or refresh) the indexed library ─────────────────────────
        available = bridge.list_available_seeds()   # sorted list of int seeds
        if not available:
            from tkinter import messagebox
            messagebox.showwarning(
                "No Seeds",
                f"No .npy files found in:\n{config.SOLID_OUTPUT_DIR}")
            return

        self._seed_index   = available           # index 0 … N-1
        self._seed_max_idx = len(available) - 1  # inclusive upper bound

        self._fpanel.log(
            f"🗂  Solid library: {len(available)} parts indexed  "
            f"(idx 0 … {self._seed_max_idx})", "ok")

        # ── Ask user which seeds to start with ──────────────────────────
        ans = sd.askstring(
            "Load Seeds",
            f"{len(available)} solids in library (idx 0 – {self._seed_max_idx}).\n"
            "Enter seed numbers, OR leave blank to pick 4 at random:",
            parent=self,
        )
        if ans is None:          # cancelled
            return

        if ans.strip():
            try:
                seeds = [int(x) for x in ans.split()]
            except ValueError:
                seeds = self._pick_random_seeds(4)
        else:
            seeds = self._pick_random_seeds(4)

        self._fpanel.log(f"Loading {len(seeds)} seed(s): {seeds}", "ok")
        threading.Thread(
            target=self._load_seeds_thread,
            args=(seeds,),
            daemon=True,
        ).start()

    def _pick_random_seeds(self, n: int) -> list[int]:
        """
        Pick *n* seeds by drawing random indices in [0, _seed_max_idx].
        The same solid may appear more than once — that is intentional.
        Returns an empty list if the library has not been built yet.
        """
        if not self._seed_index:
            return []
        return [
            self._seed_index[random.randint(0, self._seed_max_idx)]
            for _ in range(n)
        ]

    def _load_seeds_thread(self, seeds: list[int]) -> None:
        """
        Background thread: run the full CAM pipeline for each seed.
        Pipeline windows are NOT auto-opened — user opens via 📋 Queue button.
        Progress callbacks are delivered to whichever PipelineMonitorWindow
        already exists for that seed (if the user manually opened one).
        """
        for seed in seeds:
            # Deliver progress only if the user already opened the window
            win = self._pipeline_wins.get(seed)
            progress_cb = (win.progress
                           if win and win.winfo_exists() else None)

            # Run pipeline — this submits the job to the scheduler
            job = self.fa.submit_new_seed(seed, progress_cb=progress_cb)

            if job:
                entry = {
                    "seed":    seed,
                    "job_id":  job.job_id,
                    "lines":   len(job.gcode_lines),
                    "status":  "queued",
                    "machine": "—",
                }
                self._all_jobs.append(entry)
                self._eq.put(GuiEvent("job_registered", "factory",
                                       {"entry": entry}))

        # auto-start the sim after seeds are loaded
        self._eq.put(GuiEvent("autostart", "factory", {}))

    def start_sim(self) -> None:
        if self._running:
            return
        self._running = True
        self._status_bar.configure(text="Simulation running…")
        t = threading.Thread(target=self._sim_loop, daemon=True)
        t.start()

    def pause_sim(self) -> None:
        self._running = False
        self._status_bar.configure(text="Paused")

    def stop_sim(self) -> None:
        self._running = False
        self.fa = build_factory_agent(
            n_machines=4, policy=self._fpanel._policy_var.get(),
            openai_api_key=self._openai_key, dry_run=True,
        )
        for ag in self.fa.agents:
            self._gcode_cursors[ag.machine_id] = 0
            self._gcode_lines[ag.machine_id]   = []
            self._gcode_job_id[ag.machine_id]  = ""
            self._anim_queues[ag.machine_id]   = deque()
            self._unclamp[ag.machine_id]       = None
        self._initial_populate()
        self._status_bar.configure(text="Stopped — factory reset")
        self._fpanel.log("Factory reset", "warn")

    def step_sim(self) -> None:
        self._do_sim_step()

    def _on_speed(self, _=None) -> None:
        """Speed slider: scales wall-clock sleep per tick."""
        self._speed = max(0.1, min(10.0, self._speed_var.get()))
        self._speed_lbl.configure(text=f"{self._speed:.1f}x")

    def _sim_loop(self) -> None:
        """
        Main simulation loop.

        Each iteration:
          - Advances SIM_TICK_S (= 1) simulated seconds.
          - Sleeps SIM_TICK_S / (SIM_COMPRESSION x speed) wall-seconds.
          - GUI counter   : accumulates sim-s; refresh every SIM_GUI_INTERVAL_S.
          - Seed counter  : accumulates sim-s; offer seed every SIM_SEED_INTERVAL_S.
          - Scheduler tick: fa.tick() called once (T_UNIT_SECONDS = 1).
        """
        import time as _time
        wall_sleep_base = config.SIM_TICK_S / config.SIM_COMPRESSION

        while self._running:
            self._do_sim_step()
            self._sim_step_count += 1

            # GUI counter
            self._gui_acc += config.SIM_TICK_S
            if self._gui_acc >= config.SIM_GUI_INTERVAL_S:
                self._post_gui_events()
                self._gui_acc = 0.0

            # Seed counter
            self._seed_acc += config.SIM_TICK_S
            if self._seed_acc >= config.SIM_SEED_INTERVAL_S:
                self._auto_seed_tick()
                self._seed_acc = 0.0

            _time.sleep(wall_sleep_base / self._speed)

    def _do_sim_step(self) -> None:
        """
        One sim-step -- called from _sim_loop.

        Each step animates one G-code line; sleep = dt_sim / compression / speed.
          1. Scheduler tick  -- fa.tick() advances job assignments/completions.
          2. Capture jobs    -- newly assigned jobs enter per-machine anim queues.
          3. Start anim      -- if canvas idle and queue non-empty, load next job.
          4. Advance cursor  -- move one G-code line forward.
          5a. Anim complete  -- enter unclamp countdown (_UNCLAMP_STEPS steps).
          5b. Unclamp tick   -- count down; on zero, free scheduler + clear canvas.
          7. Overflow guard  -- pause if any anim queue > 10 parts.
          8. Auto-seed       -- pre-queue jobs to keep machines fed.
          9. GUI events      -- post KPI + machine-state updates.
        """
        # demo break injection
        if self._demo_break:
            mid, break_tick = self._demo_break
            if self.fa.scheduler.tick == break_tick:
                self._demo_break = None
                self.inject_error(
                    mid,
                    "T3 insert broken mid-cut — catastrophic failure; "
                    "no replacement in factory inventory",
                )

        # ── 1. Factory tick ───────────────────────────────────────────────
        result = self.fa.tick()
        cur_tick = self.fa.scheduler.tick

        # ── 2. Capture jobs from this tick, then catch-up if scheduler is ahead ──
        # _capture_jobs is called after EACH tick so anim_queue depth is
        # accurate before the next catch-up check.  This prevents the old
        # bug where 8 ticks ran without updating depth, flooding each machine
        # with up to 8 queued jobs and triggering the overflow pause.
        def _capture_jobs(tick_id: int) -> None:
            for sm in self.fa.scheduler.machines:
                mid = sm.machine_id
                if sm.current_job and sm.current_job.job_id not in self._anim_seen:
                    self._anim_seen.add(sm.current_job.job_id)
                    self._anim_queues[mid].append(sm.current_job)
            # Jobs that completed THIS tick (est_time_s < T_UNIT means a job
            # starts in tick N and finishes in tick N+1; checking started_at_tick
            # always misses them because started!=finished.  Use finished_at_tick.
            for job in self.fa.scheduler.completed_jobs:
                if (job.finished_at_tick == tick_id
                        and job.assigned_machine_id
                        and job.job_id not in self._anim_seen):
                    self._anim_seen.add(job.job_id)
                    self._anim_queues[job.assigned_machine_id].append(job)

        _capture_jobs(cur_tick)

        # Catch-up: if the scheduler freed a machine (via Fix-A remaining_s=-1)
        # and has jobs queued, run one extra tick so the assignment happens this
        # step rather than the next.  Cap at 1 extra tick per step -- enough to
        # handle the fix-A unclamp case without ever assigning more than 1 new
        # job per machine per step.
        _MAX_CATCH_UP = 1
        for _ in range(_MAX_CATCH_UP):
            needs = [
                sm for sm in self.fa.scheduler.machines
                if sm.is_idle() and self.fa.scheduler.job_queue
            ]
            if not needs:
                break
            self.fa.tick()
            cur_tick = self.fa.scheduler.tick
            _capture_jobs(cur_tick)

        # ── 3-6. Advance animation per machine ─────────────────────────────
        for ag in self.fa.agents:
            mid   = ag.machine_id
            lines = self._gcode_lines.get(mid, [])
            cur   = self._gcode_cursors.get(mid, 0)
            q     = self._anim_queues[mid]
            panel = self._panels.get(mid)

            # ── 5b. Unclamp countdown (runs INSTEAD of normal anim when active) ──
            if self._unclamp[mid] is not None:
                info = self._unclamp[mid]
                info["steps"] -= 1
                if panel:
                    panel.set_unclamp_label(info["job"].seed, info["steps"])
                if info["steps"] <= 0:
                    # Unclamp done — record completion and clear
                    self._record_completed_job(
                        mid, info["job"], info["n_lines"], cur_tick)
                    self._unclamp[mid] = None
                    self._gcode_lines[mid]   = []
                    self._gcode_cursors[mid] = 0
                    self._gcode_job_id[mid]  = ""
                    if panel:
                        panel.set_idle_label()
                    self._eq.put(GuiEvent("gcode_clear", mid, {}))
                    # Fix A: force the scheduler machine idle NOW so
                    # _assign_jobs can pick up the next queued job
                    # on the very next fa.tick() call.
                    for sm in self.fa.scheduler.machines:
                        if (sm.machine_id == mid
                                and sm.current_job is not None
                                and sm.current_job.job_id == info["job"].job_id):
                            sm.remaining_s = -1.0   # triggers _complete_job next tick
                            break
                continue   # don’t advance animation during unclamp

            # 3. Start animating next queued job if canvas is empty
            if not lines and q:
                next_job = q[0]   # peek — don’t pop yet
                self._gcode_job_id[mid]  = next_job.job_id
                self._gcode_lines[mid]   = next_job.gcode_lines
                self._gcode_cursors[mid] = 0
                lines = next_job.gcode_lines
                cur   = 0
                if panel:
                    panel.set_machining_label(next_job.seed, "▶")
                self._eq.put(GuiEvent("gcode_step", mid,
                                      {"lines": lines, "cursor": 0}))

            # 4. Advance cursor one line
            if lines and cur < len(lines):
                self._gcode_cursors[mid] = cur + 1
                self._eq.put(GuiEvent("gcode_step", mid,
                                      {"lines": lines, "cursor": cur + 1}))

            # 5a. Animation complete — enter unclamp phase
            elif lines and cur >= len(lines):
                done_job = q.popleft() if q else None
                if done_job:
                    # Start unclamp countdown instead of immediately completing
                    self._unclamp[mid] = {
                        "job":    done_job,
                        "steps":  self._UNCLAMP_STEPS,
                        "n_lines": len(lines),
                    }
                    if panel:
                        panel.set_unclamp_label(done_job.seed, self._UNCLAMP_STEPS)
                else:
                    # Nothing in queue, just clear
                    self._gcode_lines[mid]   = []
                    self._gcode_cursors[mid] = 0
                    self._gcode_job_id[mid]  = ""
                    if panel:
                        panel.set_idle_label()
                    self._eq.put(GuiEvent("gcode_clear", mid, {}))

        # ── 7. Queue overflow guard — pause if any machine queue > 10 ───────
        max_q = max((len(q) for q in self._anim_queues.values()), default=0)
        if max_q > 10 and self._running:
            self.pause_sim()
            self._fpanel.log(
                f"⚠ Queue overflow ({max_q} parts). Simulation paused.", "warn")
            return

        # ── 8. Auto-seed generation ────────────────────────────────────────
        self._auto_seed_tick()

        # ── 9. Stash tick/result for GUI flush ────────────────────────────
        # _post_gui_events() is called by _sim_loop on a wall-clock timer.
        self._last_tick   = cur_tick
        self._last_result = result

    def _post_gui_events(self) -> None:
        """Post KPI + machine-state events.  Called by _sim_loop every
        a wall-clock timer so the GUI repaint rate is capped at SIM_GUI_MAX_FPS
        regardless of how fast the sim-loop is running."""
        result = self._last_result
        cmds   = ([{"mid": c.target_machine_id, "action": c.action}
                   for c in result.commands_sent]
                  if result else [])
        self._eq.put(GuiEvent("factory_update", "factory", {
            "tick":  self._last_tick,
            "kpis":  self.fa.get_production_kpis(),
            "cmds":  cmds,
        }))
        for ag in self.fa.agents:
            self._eq.put(GuiEvent("machine_update", ag.machine_id,
                                   ag.get_state()))

    # ── Job completion recording ─────────────────────────────────────────────

    def _record_completed_job(
        self, mid: str, job, n_lines: int, tick_done: int
    ) -> None:
        """Record a finished job and post it to the completed-parts panel."""
        load_s   = config.PART_SETUP_TIME_S
        unload_s = config.PART_REMOVAL_TIME_S
        mach_s   = round(job.estimated_time_s, 1)
        total_s  = round(load_s + mach_s + unload_s, 1)
        stats = {
            "seed":        job.seed,
            "machine":     mid,
            "gcode_lines": n_lines,
            "load_s":      load_s,
            "machining_s": mach_s,
            "unload_s":    unload_s,
            "total_s":     total_s,
            "tick_done":   tick_done,
        }
        self._completed_parts.append(stats)
        # Update status in _all_jobs tracking list
        for entry in self._all_jobs:
            if entry.get("job_id") == job.job_id:
                entry["status"]  = "done"
                entry["machine"] = mid
                break
        self._eq.put(GuiEvent("job_complete", mid, stats))
        self._fpanel.log(
            f"✓ {mid} • Seed {job.seed}  "
            f"{n_lines} lines  "
            f"{mach_s:.0f}s mach  "
            f"{total_s:.0f}s total",
            "ok"
        )

    # ── Auto-seed generation ─────────────────────────────────────────────────

    def _auto_seed_tick(self) -> None:
        """
        Called every SIM_SEED_INTERVAL_S simulated seconds by _sim_loop.
        Offers one seed to the scheduler if the queue is not already full.
        Rate: SHIFT_S / SIM_SEED_INTERVAL_S = 355 seeds per shift.
        """
        if not self._seed_index:
            return
        if len(self.fa.scheduler.job_queue) >= config.SIM_MAX_QUEUE_AHEAD:
            return
        idx  = random.randint(0, self._seed_max_idx)
        seed = self._seed_index[idx]
        self._fpanel.log(f"\U0001f331 Auto-seed {seed}", "ok")
        threading.Thread(
            target=self._auto_seed_thread,
            args=(seed,),
            daemon=True,
        ).start()

    def _auto_seed_thread(self, seed: int) -> None:
        """Background thread: run CAM pipeline and submit job to scheduler."""
        try:
            win = self._pipeline_wins.get(seed)
            progress_cb = (win.progress
                           if win and win.winfo_exists() else None)
            job = self.fa.submit_new_seed(seed, progress_cb=progress_cb)
            if job:
                entry = {
                    "seed":    seed,
                    "job_id":  job.job_id,
                    "lines":   len(job.gcode_lines),
                    "status":  "queued",
                    "machine": "\u2014",
                }
                self._all_jobs.append(entry)
                self._eq.put(GuiEvent("job_registered", "factory",
                                       {"entry": entry}))
            else:
                self._fpanel.log(
                    f"\u26a0 Pipeline returned None for seed {seed}", "warn")
        except Exception as exc:
            self._fpanel.log(f"\u274c Auto-seed error: {exc}", "warn")

    def _open_queue_window(self) -> None:
        """Open (or raise) the single parts-queue display window."""
        if self._queue_win and self._queue_win.winfo_exists():
            self._queue_win.deiconify()
            self._queue_win.lift()
        else:
            self._queue_win = PartsQueueWindow(self)

    # ── Disturbance injection ─────────────────────────────────────────────────
    def inject_disturbance(
        self, machine_id: str, description: str,
        spec_key: Optional[str] = None,
        extra: Optional[dict]   = None,
    ) -> None:
        """Primary injection entry-point used by DisturbanceDialog."""
        ag = next((a for a in self.fa.agents if a.machine_id == machine_id), None)
        if ag is None:
            return
        extra = extra or {}
        spec  = REGISTRY.get(spec_key or "") if spec_key else None

        # safety-critical: emergency-stop immediately, no LLM
        if spec and spec.safety_critical:
            ag.inject_error(description)
            self._fpanel.log(
                f"⚠ SAFETY CRITICAL: {machine_id} → {spec.label}", "error")
            panel = self._panels[machine_id]
            if panel._chat is None or not panel._chat.winfo_exists():
                panel._open_chat()
            panel._chat.append(machine_id,
                f"SAFETY CRITICAL: {description}", "error")
            panel._chat.append("Factory",
                f"Emergency stop dispatched to {machine_id}", "factory")
            ag.handle_factory_response({"action": "emergency_stop"})
            self._eq.put(
                GuiEvent("machine_update", machine_id, ag.get_state()))
            return

        # inventory clear for tool-breakage with no stock
        if spec_key == "tool_breakage":
            tool_id = extra.get("tool_id", "T3")
            if extra.get("no_stock", True):
                self.fa.tool_inventory[tool_id] = []
                self._fpanel.log(
                    f"⚠ Factory inventory: {tool_id} cleared (out of stock)",
                    "warn")

        ag.inject_error(description)
        panel = self._panels[machine_id]
        if panel._chat is None or not panel._chat.winfo_exists():
            panel._open_chat()
        cw = panel._chat
        cw.append(machine_id, f"DISTURBANCE → {description}", "error")
        self._fpanel.log(
            f"{machine_id}: [{spec_key or 'generic'}] "
            f"{description[:60]}", "error")

        threading.Thread(
            target=self._diagnose_disturbance,
            args=(ag, spec_key, extra, cw),
            daemon=True,
        ).start()

    # legacy shim for --demo-break CLI path
    def inject_error(self, machine_id: str, description: str) -> None:
        key   = None
        extra = {}
        if ("broken" in description.lower()
                and "no replacement" in description.lower()):
            key   = "tool_breakage"
            extra = {"tool_id": "T3", "no_stock": True}
        self.inject_disturbance(machine_id, description,
                                 spec_key=key, extra=extra)

    # ── Multi-turn disturbance diagnosis ────────────────────────────────────
    def _diagnose_disturbance(
        self, agent: "CncAgent",
        spec_key: Optional[str],
        extra: dict,
        cw: "AgentConversationWindow",
    ) -> None:
        """
        Full multi-turn conversation flow:
          Phase 1 — machine reports (structured context dump)
          Phase 2 — factory builds engineering context + shows numbers
          Phase 3 — 3× OpenAI calls with streaming (or rule-based fallback)
          Phase 4 — candidate scoring and selection
          Phase 5 — inter-machine negotiation if action == transfer
          Phase 6 — dispatch best command + machine confirmation
          Phase 7 — follow-up call: concrete implementation steps (streaming)
        """
        mid  = agent.machine_id
        spec = REGISTRY.get(spec_key or "") if spec_key else None

        # ── helpers: post events safely from background thread ─────────────
        def ev(kind, **data):
            self._eq.put(GuiEvent(kind, mid, data))

        def phase(p):
            ev("phase", phase=p)

        def say(speaker_key, text):
            ev("chat_append", speaker=speaker_key, text=text, tag=speaker_key)

        def think(active, who=""):
            ev("thinking", active=active, who=who)

        def solution(action, risk, score, reasoning, params=None):
            ev("solution", action=action, risk=risk, score=score,
               reasoning=reasoning, params=params or {})

        def stream_llm(messages, temp, label):
            """
            Call OpenAI with streaming; post token events.
            Returns (stream_id, full_text).
            """
            sid = f"{mid}_{temp}_{time.time():.0f}"
            ev("stream_start", stream_id=sid,
               speaker_key="openai", label=label)
            full = ""
            if self._openai_key and _OPENAI_AVAILABLE:
                try:
                    client = _OpenAI(api_key=self._openai_key)
                    stream = client.chat.completions.create(
                        model=config.OPENAI_MODEL,
                        max_tokens=config.OPENAI_MAX_TOKENS,
                        temperature=temp,
                        stream=True,
                        messages=messages,
                    )
                    for chunk in stream:
                        token = chunk.choices[0].delta.content or ""
                        if token:
                            full += token
                            ev("stream_token", stream_id=sid, token=token)
                except Exception as exc:
                    full = json.dumps({
                        "action": "rework", "risk_level": "safe",
                        "reasoning": f"API error: {exc}", "parameters": {},
                    })
                    ev("stream_token", stream_id=sid, token=full)
            else:
                full = "⟨no OpenAI key — using rule-based fallback⟩"
                ev("stream_token", stream_id=sid, token=full)
            ev("stream_end", stream_id=sid)
            return sid, full

        # ── Phase 1: machine report ─────────────────────────────────────────
        phase(Phase.MACHINE_REPORT)
        state = agent.get_state()
        report_lines = [
            f"Disturbance type : {spec.label if spec else spec_key or 'unknown'}",
            f"Status           : {state.get('status','—')}",
            f"Current job      : {state.get('current_job') or '—'}",
            f"Section          : {state.get('current_section') or '—'}",
            f"Sections done    : {state.get('completed_sections', [])}",
            f"Queue depth      : {state.get('queue_depth', 0)}",
        ]
        if extra:
            report_lines.append("Extra data       : " +
                "  ".join(f"{k}={v}" for k,v in extra.items()))
        say("machine", "\n".join(report_lines))

        # ── Phase 2: factory builds context ────────────────────────────────
        phase(Phase.FACTORY_ANALYSIS)
        think(True, "Factory")
        time.sleep(0.05)  # tiny delay so GUI can render

        if spec:
            ctx = ENGINE.build_context(
                agent, spec, extra,
                factory_inventory=self.fa.tool_inventory,
                tick=self.fa.scheduler.tick,
                policy=self.fa.policy,
            )
        else:
            ctx = DisturbanceContext(
                disturbance_key="generic", machine_id=mid,
                tick=self.fa.scheduler.tick,
                factory_policy=self.fa.policy,
                machine_status=state.get("status", ""),
                current_job=state.get("current_job"),
                current_section=state.get("current_section"),
                extra=extra,
                inventory={k: len(v) for k, v in self.fa.tool_inventory.items()},
            )

        analysis_lines = [
            f"Policy           : {ctx.factory_policy}",
            f"Tool inventory   : { {k:v for k,v in ctx.inventory.items()} }",
        ]
        if ctx.kc_ratio not in (0.0, 1.0):
            analysis_lines += [
                f"Material change  : {ctx.original_material} → {ctx.actual_material}",
                f"Kc ratio         : {ctx.kc_ratio:.2f}×  "
                f"({'harder' if ctx.kc_ratio > 1 else 'softer'})",
                f"Feed             : {ctx.feed_original:.0f} → {ctx.feed_required:.0f} mm/min"
                f"  (Δ −{ctx.feed_reduction_pct:.0f}%)",
                f"RPM              : {ctx.rpm_original:.0f} → {ctx.rpm_required:.0f}",
                f"Power required   : {ctx.power_required:.2f} kW"
                f"  (limit {ctx.machine_power_kw:.1f} kW)"
                f"  {'⚠ EXCEEDS' if ctx.power_required > ctx.machine_power_kw else '✓ OK'}",
            ]
            if ctx.tool_grade_note:
                analysis_lines.append(f"Tool adequacy    : ⚠ {ctx.tool_grade_note}")
        think(False)
        say("factory", "\n".join(analysis_lines))

        # ── Phase 3: LLM queries with streaming ────────────────────────────
        phase(Phase.LLM_QUERY)
        base_prompt  = ENGINE.build_prompt(ctx)
        system_msg   = "You are a CNC factory supervisor. Respond with JSON only."
        temps        = list(config.FACTORY_LLM_TEMPS)[: config.FACTORY_N_LLM_CANDIDATES]
        raw_results: list[tuple[float, str]] = []

        if self._openai_key and _OPENAI_AVAILABLE:
            for i, temp in enumerate(temps):
                think(True, f"OpenAI t={temp:.1f}  (candidate {i+1}/{len(temps)})")
                label = f"OpenAI t={temp:.1f}"
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": base_prompt},
                ]
                _, full = stream_llm(messages, temp, label)
                raw_results.append((temp, full))
                think(False)
        else:
            # rule-based fallback — show as if typing
            say("factory", "OpenAI unavailable — using rule-based fallback.")
            fallbacks = ENGINE.fallback_responses(ctx)
            for i, cand in enumerate(fallbacks):
                label = f"Fallback t={cand.temperature:.1f}"
                sid   = f"{mid}_fb_{i}_{time.time():.0f}"
                ev("stream_start", stream_id=sid,
                   speaker_key="fallback", label=label)
                raw = json.dumps({
                    "action":     cand.action,
                    "risk_level": cand.risk_level,
                    "parameters": cand.parameters,
                    "reasoning":  cand.reasoning,
                })
                # stream character by character for visual effect
                for ch in raw:
                    ev("stream_token", stream_id=sid, token=ch)
                    time.sleep(0.004)
                ev("stream_end", stream_id=sid)
                raw_results.append((cand.temperature, raw))

        # ── Phase 4: score candidates ───────────────────────────────────────
        phase(Phase.SCORING)
        llm_candidates: list["LLMCandidate"] = []
        for temp, raw in raw_results:
            try:
                clean = (raw.strip()
                         .lstrip("```json").lstrip("```")
                         .rstrip("```").strip())
                obj = json.loads(clean)
            except Exception:
                obj = {"action": "rework", "risk_level": "safe",
                       "reasoning": raw[:120], "parameters": {}}
            cand = LLMCandidate(
                action    = obj.get("action",     "rework"),
                risk_level= obj.get("risk_level", "safe"),
                parameters= obj.get("parameters", {}),
                reasoning = obj.get("reasoning",  ""),
                temperature=temp, raw=raw,
            )
            cand.score = ENGINE.score(cand, ctx.factory_policy)
            llm_candidates.append(cand)

        for i, c in enumerate(llm_candidates):
            risk_sym = {"safe": "✓", "risky": "⚠", "catastrophic": "✖"
                        }.get(c.risk_level, "?")
            say("factory",
                f"Candidate {i+1}  action={c.action:<22s}"
                f"  {risk_sym} {c.risk_level:<12s}  score={c.score:.3f}\n"
                f"   {c.reasoning}"
                + (f"\n   params={c.parameters}" if c.parameters else ""))

        best     = max(llm_candidates, key=lambda c: c.score)
        best_idx = llm_candidates.index(best)
        say("best",
            f"Selected candidate {best_idx+1}  →  {best.action}  "
            f"(score={best.score:.3f}  risk={best.risk_level})\n"
            f"   {best.reasoning}")
        solution(best.action, best.risk_level, best.score,
                 best.reasoning, best.parameters)

        # ── Phase 5: inter-machine negotiation (if transfer) ───────────────
        if best.action == "transfer":
            phase(Phase.NEGOTIATION)
            target = (best.parameters.get("target_machine")
                      or next(
                          (a.machine_id for a in self.fa.agents
                           if a.machine_id != mid
                           and a.get_state().get("queue_depth", 99) < 3),
                          None))
            if target:
                say("factory",
                    f"Querying {target}: can you accept "
                    f"job {state.get('current_job')}?")
                time.sleep(0.12)   # simulate network hop
                t_agent = next((a for a in self.fa.agents
                                if a.machine_id == target), None)
                if t_agent:
                    t_state = t_agent.get_state()
                    t_crib  = {t["tool_id"]: t["remaining_life_pct"]
                               for t in t_state.get("tool_crib", [])}
                    ev("chat_append",
                       speaker=f"M{target[-2:]}", text=(
                           f"Queue={t_state.get('queue_depth',0)}  "
                           f"Tools={t_crib}  "
                           f"Status={t_state.get('status','idle')}.  "
                           "Ready to accept."),
                       tag="machine",  mid=target)
                    say("factory",
                        f"Transfer confirmed — dispatching to {target}.")
                    best.parameters["target_machine"] = target
            else:
                say("factory",
                    "No suitable transfer target found — falling back to rework.")
                best.action = "rework"
                solution("rework", "safe", 0.0,
                         "No transfer target available.", {})

        # ── Phase 6: dispatch command ───────────────────────────────────────
        phase(Phase.DECISION)
        safe_actions = {
            "abort", "rework", "continue", "reduce_feed",
            "emergency_stop", "change_tool", "add_finish_pass",
            "recalculate", "reduce_doc", "change_rpm",
        }
        action = best.action if best.action in safe_actions else "rework"
        cmd_params = {k: v for k, v in best.parameters.items()
                      if isinstance(v, (int, float, str))}
        agent.handle_factory_response({"action": action, **cmd_params})
        ev("machine_update", **agent.get_state())
        say("factory",
            f"Command dispatched to {mid}: {action}  params={cmd_params}")
        say("machine",
            f"Received command '{action}'.  "
            + ("Executing." if action not in ("abort","rework","emergency_stop")
               else "Halting job."))
        self._fpanel.log(
            f"{mid}: [{spec_key or 'generic'}] → {action}  "
            f"score={best.score:.2f}", "llm")

        # ── Phase 7: follow-up — concrete implementation steps ─────────────
        phase(Phase.FOLLOWUP)
        think(True, "OpenAI — implementation steps")
        followup_prompt = (
            f"The factory supervisor has chosen: {action}.\n"
            f"Parameters: {cmd_params}\n"
            f"Reasoning: {best.reasoning}\n\n"
            "Provide 3–5 numbered, concrete implementation steps that the "
            "machine operator or G-code post-processor should execute. "
            "Be specific (include example G-code if relevant). "
            "Plain text — no JSON."
        )
        followup_msgs = [
            {"role": "system",
             "content": "CNC machining expert. Practical, concise numbered steps."},
            {"role": "user",   "content": followup_prompt},
        ]
        if self._openai_key and _OPENAI_AVAILABLE:
            _, steps_text = stream_llm(followup_msgs, 0.2, "Steps")
        else:
            # rule-based implementation steps
            steps = {
                "recalculate": [
                    f"1. Update feed rate override to {cmd_params.get('feed_rate_mmpm','—')} mm/min.",
                    f"2. Update spindle speed to {cmd_params.get('rpm','—')} RPM.",
                    f"3. Verify first cut on test piece or at {cmd_params.get('depth_of_cut_mm','—')} mm DOC.",
                    "4. Confirm surface finish and dimension before continuing batch.",
                ],
                "reduce_feed": [
                    f"1. Apply feed-rate override: {cmd_params.get('override_pct', 70):.0f}%.",
                    "2. Monitor spindle load for one full pass.",
                    "3. If load < 80%, step feed back up in 5% increments.",
                ],
                "change_tool": [
                    "1. Home Z-axis and open tool-change dialog.",
                    "2. Remove worn tool from spindle.",
                    "3. Load replacement tool; update tool offset in controller.",
                    "4. Run tool-length measurement cycle (G43).",
                    "5. Resume program from last completed section.",
                ],
                "abort": [
                    "1. Execute M0 (program stop) or press cycle-stop.",
                    "2. Move spindle to safe home position (G28).",
                    "3. Log event in MES with timestamp and description.",
                    "4. Place workpiece in rework bin.",
                ],
                "rework": [
                    "1. Stop machine and record current program line.",
                    "2. Remove part and measure all completed features.",
                    "3. Assess rework feasibility; route to rework station if salvageable.",
                ],
                "emergency_stop": [
                    "1. PRESS E-STOP immediately.",
                    "2. Do not re-enable drives until root cause is identified.",
                    "3. Contact maintenance supervisor.",
                ],
            }.get(action, ["1. Follow standard operating procedure."])
            steps_text = "\n".join(steps)
            # simulate streaming for fallback
            sid = f"{mid}_steps_{time.time():.0f}"
            ev("stream_start", stream_id=sid,
               speaker_key="factory", label="Implementation steps")
            for ch in steps_text:
                ev("stream_token", stream_id=sid, token=ch)
                time.sleep(0.003)
            ev("stream_end", stream_id=sid)

        think(False)
        phase(Phase.IMPLEMENTATION)
        say("confirm",
            f"Solution complete.  Action={action}  Score={best.score:.3f}")

    # ── GUI poll loop ─────────────────────────────────────────────────────────
    def _poll(self) -> None:
        try:
            while True:
                ev = self._eq.get_nowait()
                self._apply(ev)
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def _apply(self, ev: GuiEvent) -> None:
        kind, mid = ev.kind, ev.mid

        if kind == "gcode_step":
            panel = self._panels.get(mid)
            if panel:
                panel.update_gcode(ev.data["lines"], ev.data["cursor"])

        elif kind == "gcode_clear":
            panel = self._panels.get(mid)
            if panel:
                panel.update_gcode([], 0)   # clears canvas + text

        elif kind == "job_complete":
            self._fpanel.add_completed_part(ev.data)

        elif kind == "machine_update":
            panel = self._panels.get(mid)
            if panel:
                panel.update_state(ev.data)

        elif kind == "job_registered":
            # Update queue window if open
            if self._queue_win and self._queue_win.winfo_exists():
                self._queue_win.refresh()

        elif kind == "open_pipeline":
            # Only fired by user request (queue window ‘Open Pipeline’ button)
            seed = ev.data["seed"]
            if seed not in self._pipeline_wins or not self._pipeline_wins[seed].winfo_exists():
                self._pipeline_wins[seed] = PipelineMonitorWindow(self, seed)
            else:
                self._pipeline_wins[seed].deiconify()
                self._pipeline_wins[seed].lift()

        elif kind == "autostart":
            if not self._running:
                self.start_sim()

        elif kind == "factory_update":
            kpis = ev.data["kpis"]
            tick = ev.data["tick"]
            self._fpanel.update_kpis(kpis, tick)
            self._tick_lbl.configure(text=f"tick {tick}")
            for cmd in ev.data.get("cmds", []):
                tag = "warn" if cmd["action"] in ("abort","rework") else "ok"
                self._fpanel.log(
                    f"CMD → {cmd['mid']} : {cmd['action']}", tag)

        elif kind == "chat_append":
            panel = self._panels.get(mid)
            if panel:
                panel.append_chat(
                    ev.data["speaker"], ev.data["text"], ev.data.get("tag","system"))

        elif kind == "manual_error":
            pass  # handled by AgentConversationWindow._send already

        elif kind == "stream_start":
            panel = self._panels.get(mid)
            if panel and panel._chat and panel._chat.winfo_exists():
                panel._chat.stream_start(
                    ev.data["stream_id"],
                    ev.data.get("speaker_key", "openai"),
                    ev.data.get("label", ""),
                )

        elif kind == "stream_token":
            panel = self._panels.get(mid)
            if panel and panel._chat and panel._chat.winfo_exists():
                panel._chat.stream_token(
                    ev.data["stream_id"], ev.data["token"])

        elif kind == "stream_end":
            panel = self._panels.get(mid)
            if panel and panel._chat and panel._chat.winfo_exists():
                panel._chat.stream_end(ev.data["stream_id"])

        elif kind == "phase":
            panel = self._panels.get(mid)
            if panel and panel._chat and panel._chat.winfo_exists():
                panel._chat.phase_header(ev.data["phase"])

        elif kind == "thinking":
            panel = self._panels.get(mid)
            if panel and panel._chat and panel._chat.winfo_exists():
                panel._chat.set_thinking(
                    ev.data["active"], ev.data.get("who", ""))

        elif kind == "solution":
            panel = self._panels.get(mid)
            if panel and panel._chat and panel._chat.winfo_exists():
                panel._chat.set_solution(
                    ev.data["action"], ev.data["risk"],
                    ev.data["score"],  ev.data["reasoning"],
                    ev.data.get("params", {}),
                )

    # ── Misc ──────────────────────────────────────────────────────────────────
    def _set_key(self) -> None:
        key = simpledialog.askstring(
            "OpenAI API Key",
            "Paste your OpenAI API key (sk-…):",
            show="*", parent=self,
        )
        if key and key.strip():
            self._openai_key = key.strip()
            self.fa._use_llm = _OPENAI_AVAILABLE
            if _OPENAI_AVAILABLE:
                self.fa._llm_client = _OpenAI(api_key=self._openai_key)
            self._fpanel.log("OpenAI key set — LLM enabled", "ok")
            self._status_bar.configure(text="OpenAI LLM connected")

    def _on_close(self) -> None:
        self._running = False
        for panel in self._panels.values():
            if panel._sim_proc and panel._sim_proc.poll() is None:
                panel._sim_proc.terminate()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────
#   Parts Queue Window
# ───────────────────────────────────────────────────────────────────────────

class PartsQueueWindow(tk.Toplevel):
    """
    Single window showing all parts registered this session.
    Columns: #  Seed  Lines  Status  Machine  Mach.s

    Bottom panel: enter a Seed number → “Open Pipeline” opens (or raises)
    the PipelineMonitorWindow for that seed.
    Refreshed every 2 seconds via after().
    """

    _COLS = ("#", "Seed", "Lines", "Status", "Machine", "Mach.s")
    _WIDTHS = {"#": 35, "Seed": 60, "Lines": 55, "Status": 80,
               "Machine": 60, "Mach.s": 60}
    _STATUS_FG = {
        "queued":    ACCENT,
        "machining": GREEN,
        "unloading": ORANGE,
        "done":      DIM,
    }

    def __init__(self, app: "FactoryGUI"):
        super().__init__(app)
        self._app = app
        self.title("📋 Parts Queue")
        self.configure(bg=BG)
        self.resizable(True, True)

        # ── Header
        tk.Label(self, text="Parts Queue — all registered parts this session",
                 bg=BG, fg=FG, font=("Helvetica", 10, "bold")).pack(
                 anchor="w", padx=10, pady=(8, 2))

        # ── Treeview
        frm = tk.Frame(self, bg=BG)
        frm.pack(fill="both", expand=True, padx=8, pady=4)

        style = ttk.Style()
        style.configure("Queue.Treeview",
                         background="#0a1628", foreground=FG,
                         fieldbackground="#0a1628",
                         font=("Courier", 9), rowheight=18)
        style.configure("Queue.Treeview.Heading",
                         background=PNL, foreground=DIM,
                         font=("Courier", 9, "bold"))

        self._tree = ttk.Treeview(
            frm, columns=self._COLS, show="headings",
            height=20, style="Queue.Treeview",
            selectmode="browse",
        )
        for c in self._COLS:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=self._WIDTHS.get(c, 70),
                              anchor="center", stretch=False)
        vsb = ttk.Scrollbar(frm, orient="vertical",
                             command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Colour tags
        self._tree.tag_configure("queued",    foreground=ACCENT)
        self._tree.tag_configure("machining", foreground=GREEN)
        self._tree.tag_configure("unloading", foreground=ORANGE)
        self._tree.tag_configure("done",      foreground=DIM)

        # ── Pipeline-open panel
        sep = tk.Frame(self, bg=DIM, height=1)
        sep.pack(fill="x", padx=8, pady=(2, 0))

        bot = tk.Frame(self, bg=BG)
        bot.pack(fill="x", padx=10, pady=8)

        tk.Label(bot, text="Open pipeline for Seed:",
                 bg=BG, fg=FG, font=("Courier", 9)).pack(side="left")
        self._seed_entry = tk.Entry(
            bot, bg=ENTRY, fg=FG,
            insertbackground=FG, font=("Courier", 10), width=8,
        )
        self._seed_entry.pack(side="left", padx=6)
        tk.Button(
            bot, text="Open Pipeline",
            bg="#42a5f5", fg="black",
            relief="flat", font=("Helvetica", 9, "bold"), padx=8,
            activebackground="#1976d2", activeforeground="white",
            command=self._open_pipeline,
        ).pack(side="left")

        self._count_lbl = tk.Label(bot, text="",
                                   bg=BG, fg=DIM, font=("Courier", 9))
        self._count_lbl.pack(side="right")

        # ── Initial populate + periodic refresh
        self.refresh()
        self._sched_refresh()

    # ── public API ─────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the treeview from the current _all_jobs + live state."""
        if not self.winfo_exists():
            return

        # Gather live machine assignments
        live_status: dict[str, str] = {}   # job_id → status
        live_machine: dict[str, str] = {}  # job_id → mid
        for ag in self._app.fa.agents:
            mid = ag.machine_id
            # Currently animating
            jid = self._app._gcode_job_id.get(mid, "")
            if jid:
                live_status[jid]  = "machining"
                live_machine[jid] = mid
            # In unclamp phase
            unc = self._app._unclamp.get(mid)
            if unc and unc.get("job"):
                live_status[unc["job"].job_id]  = "unloading"
                live_machine[unc["job"].job_id] = mid
            # Queued (not yet animating)
            for job in self._app._anim_queues[mid]:
                if job.job_id not in live_status:
                    live_status[job.job_id]  = "queued"
                    live_machine[job.job_id] = mid

        # Rebuild treeview
        self._tree.delete(*self._tree.get_children())
        jobs = self._app._all_jobs
        for idx, entry in enumerate(reversed(jobs), 1):
            jid    = entry.get("job_id", "")
            status = live_status.get(jid, entry.get("status", "queued"))
            mach   = live_machine.get(jid, entry.get("machine", "—"))

            # Machining time from completed parts (if done)
            mach_s = "—"
            for cp in self._app._completed_parts:
                if cp.get("seed") == entry.get("seed"):
                    mach_s = f"{cp['machining_s']:.0f}"
                    break

            row = (
                str(len(jobs) - idx + 1),
                str(entry.get("seed", "?")),
                str(entry.get("lines", 0)),
                status,
                mach,
                mach_s,
            )
            self._tree.insert("", "end", iid=jid or str(idx),
                               values=row, tags=(status,))

        total  = len(jobs)
        done   = sum(1 for e in jobs if e.get("status") == "done")
        active = sum(1 for s in live_status.values() if s in ("machining", "unloading"))
        queued = total - done - active
        self._count_lbl.configure(
            text=f"Total {total}  •  machining {active}  •  queued {queued}  •  done {done}")

    def _open_pipeline(self) -> None:
        """Open the PipelineMonitorWindow for the entered seed."""
        raw = self._seed_entry.get().strip()
        if not raw.isdigit():
            tk.messagebox.showerror(
                "Invalid seed", f"'{raw}' is not a valid seed number.",
                parent=self)
            return
        seed = int(raw)
        # Check seed is known
        known = {e["seed"] for e in self._app._all_jobs}
        if seed not in known:
            tk.messagebox.showwarning(
                "Unknown seed",
                f"Seed {seed} has not been submitted this session.",
                parent=self)
            return
        self._app._eq.put(GuiEvent("open_pipeline", "factory", {"seed": seed}))

    def _sched_refresh(self) -> None:
        if self.winfo_exists():
            self.refresh()
            self.after(2000, self._sched_refresh)


def main() -> None:
    ap = argparse.ArgumentParser(description="CNC Factory Visual Dashboard")
    ap.add_argument("--openai-key", default=None,
                    help="OpenAI key (overrides factory_secrets.ini)")
    ap.add_argument("--auto",       action="store_true",
                    help="Start simulation automatically")
    ap.add_argument("--demo-break", nargs=2, metavar=("MACHINE", "TICK"),
                    help="Inject tool-break on MACHINE at TICK (e.g. M02 5)")
    args = ap.parse_args()

    demo = None
    if args.demo_break:
        demo = (args.demo_break[0], int(args.demo_break[1]))

    # CLI flag > factory_secrets.ini > empty (LLM disabled)
    key = args.openai_key or config.OPENAI_API_KEY or None

    app = FactoryGUI(
        openai_key = key,
        auto_start = args.auto,
        demo_break = demo,
    )
    app.mainloop()


if __name__ == "__main__":
    main()

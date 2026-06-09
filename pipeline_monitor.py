"""
pipeline_monitor.py — Per-seed pipeline visualisation window.
=============================================================

Shows each of the 8 stages that turn a raw seed number into a running
CNC job.  One window is opened per seed; windows stay open so the user
can see the timings for every job that passed through the factory.

Stage flow
──────────
  🌱 Seed → 🧱 Solid → 📐 Features → ⚙ F&S → 📍 Toolpath
           → 📄 G-Code → 📅 Scheduler → 🏭 Machine

Each box changes colour:
  ● grey   — not yet started
  ● yellow — in progress
  ● green  — complete
  ● red    — error

Thread safety
─────────────
  progress(phase, detail) is safe to call from any thread.
  All GUI updates happen on the main thread via a queue polled
  with after(80 ms).

Integration
───────────
  factory_agent.submit_new_seed(seed, progress_cb=win.progress)
  The callback signature is:  cb(seed, phase, detail)
  but the window only uses  (phase, detail) — seed is for logging.
"""

from __future__ import annotations

import queue
import time
from typing import Optional

import tkinter as tk
from tkinter import ttk

# ── Theme (mirrors factory_gui.py) ────────────────────────────────────────────
BG     = "#0d1117"
PNL    = "#161b22"
ENTRY  = "#0f3460"
FG     = "#c9d1d9"
DIM    = "#6e7681"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
YELLOW = "#d29922"
RED    = "#f85149"
ORANGE = "#e3b341"
PURPLE = "#bc8cff"

MACHINE_ACCENT = {
    "M01": "#1f6feb", "M02": "#238636",
    "M03": "#9e6a03", "M04": "#8957e5",
}

# ── Stage definitions ─────────────────────────────────────────────────────────
# (internal key, display label, short description shown in the box)
STAGES: list[tuple[str, str, str]] = [
    ("seed",     "🌱 Seed",      "Selected"),
    ("solid",    "🧱 Solid",     "Load .npy"),
    ("features", "📐 Features",  "Extract"),
    ("fs",       "⚙ F&S",       "Feeds/Spd"),
    ("toolpath", "📍 Toolpath",  "Plan path"),
    ("gcode",    "📄 G-Code",    "Generate"),
    ("schedule", "📅 Scheduler", "Queue"),
    ("machine",  "🏭 Machine",   "Assign"),
]

STATE_COLOR: dict[str, str] = {
    "pending": DIM,
    "active":  YELLOW,
    "done":    GREEN,
    "error":   RED,
}

# Map from progress-callback phase string → stage key
_PHASE_MAP: dict[str, str] = {
    "seed":      "seed",
    "solid":     "solid",
    "features":  "features",
    "fs":        "fs",
    "toolpath":  "toolpath",
    "gcode":     "gcode",
    "schedule":  "schedule",
    "machine":   "machine",
}


# ─────────────────────────────────────────────────────────────────────────────

class PipelineMonitorWindow(tk.Toplevel):
    """
    Visual pipeline-progress window for one seed.

    Usage
    -----
        win = PipelineMonitorWindow(root, seed=42)
        # from background thread:
        win.progress("solid",    "Loading 18 faces, vol=12 450 mm³")
        win.progress("features", "3 top faces, 12 edges labelled")
        ...
        win.progress("machine",  "Assigned → M02")
    """

    def __init__(self, parent: tk.Misc, seed: int) -> None:
        super().__init__(parent)
        self._seed   = seed
        self._q:     queue.Queue              = queue.Queue()
        self._t0:    float                    = time.perf_counter()
        self._boxes: dict[str, tuple]         = {}   # key → (frame, label, dot, time_lbl)
        self._states: dict[str, str]          = {k: "pending" for k, *_ in STAGES}
        self._stage_start: dict[str, float]   = {}

        self.title(f"Pipeline  —  Seed {seed}")
        self.configure(bg=BG)
        self.geometry("820x380")
        self.minsize(700, 320)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

        self._build_ui()
        self.after(80, self._poll)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── header bar ────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#1c2128", height=32)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(
            hdr,
            text=f"🌱  Seed {self._seed}  →  Full Pipeline",
            bg="#1c2128", fg=ACCENT,
            font=("Helvetica", 11, "bold"),
        ).pack(side="left", padx=10)
        self._elapsed_lbl = tk.Label(
            hdr, text="0.0 s", bg="#1c2128", fg=DIM,
            font=("Courier", 10),
        )
        self._elapsed_lbl.pack(side="right", padx=10)

        # ── stage boxes ───────────────────────────────────────────────────
        row = tk.Frame(self, bg=BG)
        row.pack(fill="x", padx=8, pady=(10, 4))

        for i, (key, label, sub) in enumerate(STAGES):
            # stage column
            col = tk.Frame(row, bg=BG)
            col.pack(side="left", expand=True, fill="x", padx=1)

            box = tk.Frame(
                col, bg=PNL, bd=0,
                highlightbackground=DIM,
                highlightthickness=1,
            )
            box.pack(fill="x")

            lbl = tk.Label(
                box, text=label, bg=PNL, fg=DIM,
                font=("Helvetica", 8, "bold"),
                pady=5, padx=3, anchor="center",
            )
            lbl.pack(fill="x")

            sub_lbl = tk.Label(
                box, text=sub, bg=PNL, fg=DIM,
                font=("Courier", 7),
                pady=1,
            )
            sub_lbl.pack(fill="x")

            dot = tk.Label(box, text="●", bg=PNL, fg=DIM,
                           font=("Helvetica", 16))
            dot.pack(pady=2)

            t_lbl = tk.Label(box, text="", bg=PNL, fg=DIM,
                             font=("Courier", 7))
            t_lbl.pack()

            self._boxes[key] = (box, lbl, dot, t_lbl)

            # arrow between stages (not after last)
            if i < len(STAGES) - 1:
                tk.Label(row, text="▶", bg=BG, fg=DIM,
                         font=("Helvetica", 11)).pack(side="left")

        # ── detail log ────────────────────────────────────────────────────
        tk.Label(
            self, text="Pipeline log",
            bg=BG, fg=DIM, font=("Courier", 8),
        ).pack(anchor="w", padx=10, pady=(4, 0))

        log_frm = tk.Frame(self, bg=BG)
        log_frm.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self._log = tk.Text(
            log_frm, bg="#070d15", fg=FG,
            font=("Courier", 9), state="disabled",
            wrap="word", height=6,
        )
        sb = ttk.Scrollbar(log_frm, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        self._log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # text tags
        self._log.tag_config("ts",    foreground=DIM,    font=("Courier", 7))
        self._log.tag_config("phase", foreground=ACCENT, font=("Courier", 9, "bold"))
        self._log.tag_config("ok",    foreground=GREEN)
        self._log.tag_config("err",   foreground=RED)
        self._log.tag_config("body",  foreground=FG)

        # ── result banner ─────────────────────────────────────────────────
        self._result_lbl = tk.Label(
            self, text="", bg=PNL, fg=GREEN,
            font=("Helvetica", 10, "bold"), pady=5,
        )
        self._result_lbl.pack(fill="x", padx=8, pady=(0, 6))

    # ── Public API — thread-safe ──────────────────────────────────────────────

    def progress(self, seed: int, phase: str, detail: str = "") -> None:
        """
        Post a progress event.  Safe to call from any thread.

        Parameters
        ----------
        seed   : seed number (informational, for multi-seed contexts)
        phase  : one of the STAGES keys, or "error"
        detail : human-readable detail string
        """
        self._q.put((phase, detail, time.perf_counter() - self._t0))

    # ── Poll (main thread only) ───────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                phase, detail, elapsed = self._q.get_nowait()
                self._apply(phase, detail, elapsed)
        except queue.Empty:
            pass

        # keep the elapsed clock ticking until done
        if not self._is_finished():
            self._elapsed_lbl.configure(
                text=f"{time.perf_counter() - self._t0:.1f} s"
            )

        self.after(80, self._poll)

    def _apply(self, phase: str, detail: str, elapsed: float) -> None:
        ts = f"{elapsed:6.2f}s"

        if phase == "error":
            # mark the last active stage as errored
            for key, *_ in STAGES:
                if self._states[key] == "active":
                    self._states[key] = "error"
                    self._set_box(key, "error", "")
            self._log_write(ts, "error", detail, tag="err")
            self._result_lbl.configure(
                text=f"✗  Seed {self._seed}  —  pipeline error",
                fg=RED, bg=PNL,
            )
            self._elapsed_lbl.configure(text=f"{elapsed:.2f} s  ✗")
            return

        # mark any currently active stage as done before activating next
        for key, *_ in STAGES:
            if self._states[key] == "active":
                dur = elapsed - self._stage_start.get(key, elapsed)
                self._states[key] = "done"
                self._set_box(key, "done", f"{dur:.2f}s")

        # activate the new stage
        stage_key = _PHASE_MAP.get(phase)
        if stage_key and self._states.get(stage_key) == "pending":
            self._states[stage_key] = "active"
            self._stage_start[stage_key] = elapsed
            self._set_box(stage_key, "active", "")

        self._log_write(ts, phase, detail)

        # if this is the final stage "machine", close it out immediately
        if phase == "machine":
            if stage_key:
                self._states[stage_key] = "done"
                self._set_box(stage_key, "done", "")
            self._result_lbl.configure(
                text=f"✓  Seed {self._seed}  —  pipeline complete  ({elapsed:.2f} s)  |  {detail}",
                fg=GREEN, bg=PNL,
            )
            self._elapsed_lbl.configure(text=f"{elapsed:.2f} s  ✓")

    def _set_box(self, key: str, state: str, time_str: str) -> None:
        """Update the visual box for a stage."""
        if key not in self._boxes:
            return
        box, lbl, dot, t_lbl = self._boxes[key]
        color = STATE_COLOR.get(state, DIM)
        lbl.configure(fg=color)
        dot.configure(fg=color)
        box.configure(highlightbackground=color)
        if time_str:
            t_lbl.configure(text=time_str, fg=color)

    def _log_write(self, ts: str, phase: str, detail: str,
                   tag: str = "body") -> None:
        self._log.configure(state="normal")
        self._log.insert("end", f"[{ts}] ", "ts")
        self._log.insert("end", f"{phase:<10s}  ", "phase")
        self._log.insert("end", detail + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _is_finished(self) -> bool:
        return self._states.get("machine") in ("done", "error")

"""
agent_dialog.py  —  Rich real-time conversation window for the factory agents.

Layout
──────
  ┌──────────────────────────────────────────────────────────────┐
  │  💬  M02  ↔  Factory  ↔  OpenAI               [Export] [×]  │
  ├──────────────────────────────────────────────────────────────┤
  │  ┌─ conversation log ──────────────────────────────────────┐ │
  │  │  [auto-scrolling, phase-separated, streaming tokens]    │ │
  │  └─────────────────────────────────────────────────────────┘ │
  │  ┌─ thinking bar ──────────────────────────────────────────┐ │
  │  │  ⣾  OpenAI t=0.5  generating…                           │ │
  │  └─────────────────────────────────────────────────────────┘ │
  │  ┌─ solution summary ──────────────────────────────────────┐ │
  │  │  ACTION  recalculate  │  RISK  safe  │  SCORE  0.800    │ │
  │  │  Feed 3299 → 1374 mm/min   RPM 8247 → 3436              │ │
  │  └─────────────────────────────────────────────────────────┘ │
  │  [Operator context / override…]              [Send] [Clear]  │
  └──────────────────────────────────────────────────────────────┘

Key capabilities
────────────────
• phase_header()   — visual separator between conversation phases
• append()         — add a complete message (speaker-colour-coded)
• stream_start()   — open a streaming slot; returns stream_id
• stream_token()   — append one token to a live slot
• stream_end()     — close and finalise a streaming slot
• set_thinking()   — start/stop animated spinner (main-thread safe)
• set_solution()   — update the pinned summary bar
• export_log()     — dump conversation as plain text

All public methods are safe to call from the main GUI thread only.
Background threads communicate via the FactoryGUI event queue.
"""

from __future__ import annotations

import datetime
import os
import time
import uuid
from enum import Enum
from typing import Optional, TYPE_CHECKING

import tkinter as tk
from tkinter import ttk, filedialog

if TYPE_CHECKING:
    from factory_gui import GuiEvent

# ── Theme (mirrors factory_gui.py) ───────────────────────────────────────────
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
TEAL   = "#39d353"

MACHINE_ACCENT = {
    "M01": "#1f6feb", "M02": "#238636",
    "M03": "#9e6a03", "M04": "#8957e5",
}

# Speaker → (display_name, colour, left_pad)
SPEAKER_STYLE: dict[str, tuple[str, str, int]] = {
    "machine":  ("{mid}",        ACCENT,  0),
    "factory":  ("Factory",      GREEN,   0),
    "openai":   ("OpenAI",       PURPLE,  2),
    "fallback": ("Rule-based",   TEAL,    2),
    "best":     ("✓ Factory",    YELLOW,  0),
    "error":    ("⚠ Error",      RED,     0),
    "system":   ("System",       DIM,     0),
    "operator": ("Operator",     ORANGE,  0),
    "confirm":  ("✓ Confirmed",  GREEN,   0),
}

# Conversation phase separators
class Phase(str, Enum):
    MACHINE_REPORT   = "machine_report"
    FACTORY_ANALYSIS = "factory_analysis"
    LLM_QUERY        = "llm_query"
    SCORING          = "scoring"
    NEGOTIATION      = "negotiation"
    DECISION         = "decision"
    FOLLOWUP         = "followup"
    IMPLEMENTATION   = "implementation"

PHASE_LABEL = {
    Phase.MACHINE_REPORT:   "MACHINE REPORT",
    Phase.FACTORY_ANALYSIS: "FACTORY ANALYSIS",
    Phase.LLM_QUERY:        "LLM QUERY",
    Phase.SCORING:          "CANDIDATE SCORING",
    Phase.NEGOTIATION:      "INTER-MACHINE NEGOTIATION",
    Phase.DECISION:         "DECISION",
    Phase.FOLLOWUP:         "FOLLOW-UP",
    Phase.IMPLEMENTATION:   "IMPLEMENTATION",
}

# Spinner frames
_SPIN = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")


class AgentConversationWindow(tk.Toplevel):
    """
    Per-machine conversation window.
    One instance lives for the lifetime of the GUI; calling open() brings
    it to focus and resets if needed.
    """

    def __init__(self, parent, machine_id: str, factory_gui):
        super().__init__(parent)
        self._mid  = machine_id
        self._app  = factory_gui
        self._streams:  dict[str, str] = {}   # stream_id → tk mark name
        self._spin_idx: int            = 0
        self._thinking: bool           = False
        self._think_who: str           = ""

        accent = MACHINE_ACCENT.get(machine_id, ACCENT)
        self.title(f"💬  {machine_id}  ↔  Factory  ↔  OpenAI")
        self.configure(bg=BG)
        self.geometry("780x620")
        self.minsize(600, 450)
        self.resizable(True, True)

        self._build_ui(accent)
        self.after(200, self._tick_spinner)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)  # hide, don't destroy

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self, accent: str) -> None:
        # title bar
        hdr = tk.Frame(self, bg=accent, height=30)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr,
                 text=f"💬  {self._mid}  ↔  Factory  ↔  OpenAI",
                 bg=accent, fg="white",
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=8)
        tk.Button(hdr, text="Export log", bg=accent, fg="white",
                  relief="flat", font=("Helvetica", 8),
                  command=self._export).pack(side="right", padx=6)
        tk.Button(hdr, text="Clear", bg=accent, fg="white",
                  relief="flat", font=("Helvetica", 8),
                  command=self.clear).pack(side="right", padx=2)

        # main pane
        pane = tk.Frame(self, bg=BG)
        pane.pack(fill="both", expand=True)
        pane.rowconfigure(0, weight=1)
        pane.columnconfigure(0, weight=1)

        # ── conversation log ──────────────────────────────────────────────
        self._log = tk.Text(
            pane, bg="#070d15", fg=FG,
            font=("Courier", 10), wrap="word",
            state="disabled", selectbackground=ENTRY,
            insertbackground=FG, padx=6, pady=4,
        )
        sb = ttk.Scrollbar(pane, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        self._log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self._setup_tags()

        # ── thinking bar ──────────────────────────────────────────────────
        think_frame = tk.Frame(self, bg=PNL, height=22)
        think_frame.pack(fill="x")
        think_frame.pack_propagate(False)
        self._think_lbl = tk.Label(
            think_frame, text="", bg=PNL, fg=PURPLE,
            font=("Courier", 9), anchor="w",
        )
        self._think_lbl.pack(fill="x", padx=8)

        # ── solution summary ──────────────────────────────────────────────
        sol_frame = tk.LabelFrame(
            self, text="Solution", bg=PNL,
            fg=YELLOW, font=("Helvetica", 8, "bold"),
        )
        sol_frame.pack(fill="x", padx=4, pady=2)
        self._sol_action = tk.Label(
            sol_frame, text="—", bg=PNL, fg=YELLOW,
            font=("Helvetica", 10, "bold"),
        )
        self._sol_action.pack(side="left", padx=8)
        self._sol_risk = tk.Label(
            sol_frame, text="", bg=PNL, fg=DIM,
            font=("Courier", 9),
        )
        self._sol_risk.pack(side="left", padx=4)
        self._sol_score = tk.Label(
            sol_frame, text="", bg=PNL, fg=ACCENT,
            font=("Courier", 9),
        )
        self._sol_score.pack(side="left", padx=4)
        self._sol_detail = tk.Label(
            sol_frame, text="", bg=PNL, fg=FG,
            font=("Courier", 9), wraplength=480, justify="left",
        )
        self._sol_detail.pack(side="left", padx=6)

        # ── operator input ────────────────────────────────────────────────
        inp = tk.Frame(self, bg=BG)
        inp.pack(fill="x", padx=4, pady=4)
        self._entry = tk.Entry(
            inp, bg=ENTRY, fg=FG, insertbackground=FG,
            font=("Courier", 10),
            relief="flat",
        )
        self._entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._entry.bind("<Return>", self._send)
        self._entry.insert(0, "Add context or override decision…")
        self._entry.bind("<FocusIn>",
            lambda _: (self._entry.delete(0, "end")
                       if self._entry.get().startswith("Add context") else None))
        tk.Button(
            inp, text="Send", bg=MACHINE_ACCENT.get(self._mid, ACCENT),
            fg="white", relief="flat", padx=10,
            command=self._send,
        ).pack(side="right")

    def _setup_tags(self) -> None:
        t = self._log
        t.tag_config("ts",        foreground=DIM,    font=("Courier", 8))
        t.tag_config("machine",   foreground=ACCENT, font=("Courier", 10, "bold"))
        t.tag_config("factory",   foreground=GREEN,  font=("Courier", 10, "bold"))
        t.tag_config("openai",    foreground=PURPLE, font=("Courier", 10, "bold"))
        t.tag_config("fallback",  foreground=TEAL,   font=("Courier", 10, "bold"))
        t.tag_config("best",      foreground=YELLOW, font=("Courier", 10, "bold"))
        t.tag_config("error",     foreground=RED,    font=("Courier", 10, "bold"))
        t.tag_config("system",    foreground=DIM)
        t.tag_config("operator",  foreground=ORANGE, font=("Courier", 10, "bold"))
        t.tag_config("confirm",   foreground=GREEN)
        t.tag_config("body",      foreground=FG)
        t.tag_config("indent",    foreground=DIM,    lmargin1=20, lmargin2=20)
        t.tag_config("phase_sep", foreground="#30363d",
                     font=("Courier", 8), spacing1=6, spacing3=4)
        t.tag_config("stream",    foreground="#a0c0ff")
        t.tag_config("param",     foreground=ORANGE, font=("Courier", 9))

    # ── public interface ──────────────────────────────────────────────────────

    def phase_header(self, phase: Phase) -> None:
        """Insert a visual phase separator."""
        label = PHASE_LABEL.get(phase, phase.value.upper())
        bar   = "─" * 8
        self._write(f"\n{bar}  {label}  {bar}\n", "phase_sep")

    def append(self, speaker_key: str, text: str, *, indent: bool = False) -> None:
        """Add a complete message. speaker_key must be in SPEAKER_STYLE."""
        style = SPEAKER_STYLE.get(speaker_key, SPEAKER_STYLE["system"])
        name, color, pad = style
        name  = name.format(mid=self._mid)
        ts    = datetime.datetime.now().strftime("%H:%M:%S")
        pfx   = " " * pad

        self._write(f"{pfx}[{ts}] ", "ts")
        self._write(f"{pfx}{name:<12s}  ", speaker_key)

        body_tag = "indent" if indent else "body"
        # first line inline with header
        lines = text.split("\n")
        self._write(lines[0] + "\n", body_tag)
        for line in lines[1:]:
            self._write("               " + " " * pad + line + "\n", body_tag)

    def stream_start(self, stream_id: str, speaker_key: str,
                     label: str = "") -> None:
        """
        Open a streaming slot.  Subsequent stream_token() calls append to it.
        Call stream_end() when the LLM finishes.
        """
        style = SPEAKER_STYLE.get(speaker_key, SPEAKER_STYLE["openai"])
        _, color, pad = style
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        pfx  = " " * pad
        name = label or speaker_key

        self._log.configure(state="normal")
        self._write(f"{pfx}[{ts}] ", "ts")
        self._write(f"{pfx}{name:<12s}  ", speaker_key)
        # insert a uniquely named mark so we can append tokens here
        mark = f"stream_{stream_id}"
        self._log.mark_set(mark, "end")
        self._log.mark_gravity(mark, "left")
        self._streams[stream_id] = mark
        self._log.configure(state="disabled")

    def stream_token(self, stream_id: str, token: str) -> None:
        """Append one token to a live streaming slot."""
        mark = self._streams.get(stream_id)
        if mark is None:
            return
        self._log.configure(state="normal")
        self._log.insert(mark, token, "stream")
        self._log.see("end")
        self._log.configure(state="disabled")

    def stream_end(self, stream_id: str) -> None:
        """Finalise a streaming slot."""
        mark = self._streams.pop(stream_id, None)
        if mark:
            self._log.configure(state="normal")
            self._log.insert(mark, "\n", "body")
            self._log.mark_unset(mark)
            self._log.see("end")
            self._log.configure(state="disabled")

    def set_thinking(self, thinking: bool, who: str = "") -> None:
        """Show / hide the animated thinking bar."""
        self._thinking = thinking
        self._think_who = who
        if not thinking:
            self._think_lbl.configure(text="")

    def set_solution(self, action: str, risk: str, score: float,
                     reasoning: str, params: dict) -> None:
        """Update the pinned solution summary bar."""
        risk_color = {"safe": GREEN, "risky": ORANGE,
                      "catastrophic": RED}.get(risk, DIM)
        self._sol_action.configure(text=action.upper())
        self._sol_risk.configure(text=f"risk: {risk}", fg=risk_color)
        self._sol_score.configure(text=f"score: {score:.3f}")
        detail = reasoning
        if params:
            detail += "   " + "  ".join(
                f"{k}={v}" for k, v in params.items()
                if isinstance(v, (int, float, str))
            )
        self._sol_detail.configure(text=detail)

    def clear(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._streams.clear()
        self._sol_action.configure(text="—")
        self._sol_risk.configure(text="")
        self._sol_score.configure(text="")
        self._sol_detail.configure(text="")

    def open(self) -> None:
        """Bring window to front (re-show if withdrawn)."""
        self.deiconify()
        self.lift()
        self.focus_set()

    def export_log(self) -> str:
        """Return the full conversation as plain text."""
        return self._log.get("1.0", "end")

    # ── internals ─────────────────────────────────────────────────────────────

    def _write(self, text: str, tag: str = "body") -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text, tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _tick_spinner(self) -> None:
        if self._thinking:
            frame = _SPIN[self._spin_idx % len(_SPIN)]
            self._think_lbl.configure(
                text=f"  {frame}  {self._think_who}  generating…")
            self._spin_idx += 1
        self.after(140, self._tick_spinner)

    def _send(self, _=None) -> None:
        txt = self._entry.get().strip()
        if not txt or txt.startswith("Add context") or txt.startswith("Type intent"):
            return
        self._entry.delete(0, "end")
        self.append("operator", txt)

        agent = next((a for a in self._app.fa.agents
                      if a.machine_id == self._mid), None)
        if agent is None:
            return

        # ── Classify: query vs disturbance action ─────────────────────────
        import threading
        from agent_intents import (
            classify_message, extract_parameters,
            classify_action, ActionExecutor,
        )

        msg_type, intent_id = classify_message(txt)

        if msg_type == "query":
            # ── Path A: data query — direct answer, no LLM ──────────────
            params = extract_parameters(txt, intent_id)
            threading.Thread(
                target=self._answer_query,
                args=(agent, intent_id, params),
                daemon=True,
            ).start()

        else:
            action_type, action_params = classify_action(txt)

            if action_type != "unknown":
                # ── Path B: known deterministic action ──────────────────
                threading.Thread(
                    target=self._execute_action,
                    args=(agent, action_type, action_params),
                    daemon=True,
                ).start()
            else:
                # ── Path C: complex / ambiguous → full LLM pipeline ─────
                agent.inject_error(txt)
                threading.Thread(
                    target=self._app._diagnose_disturbance,
                    args=(agent, None, {}, self),
                    daemon=True,
                ).start()

    def _answer_query(self, agent, intent_id: str, parameters: dict) -> None:
        """Execute a query intent and stream the result into the chat log."""
        from agent_intents import IntentExecutor, INTENT_MAP, build_system_context
        import time

        cw = self   # conversation window

        def post(speaker, text):
            """Thread-safe append via the app's post_chat helper.
            Avoids a circular import — factory_gui imports agent_dialog,
            so agent_dialog must NOT import factory_gui at runtime."""
            self._app.post_chat(self._mid, speaker, text)

        executor = IntentExecutor(agent, self._app.fa)

        # Inject the msim reference so queue queries can read the live queue
        msim = self._app._msim.get(agent.machine_id)
        if msim is not None:
            executor._msim = msim

        result = executor.execute(intent_id, parameters)

        intent_label = INTENT_MAP[intent_id].label if intent_id in INTENT_MAP else intent_id
        post("factory", f"ℹ {intent_label}")
        post("machine", result["result"])

    def _execute_action(self, agent, action_type: str, parameters: dict) -> None:
        """Execute a deterministic action and post each chat line to the log."""
        from agent_intents import ActionExecutor, _ACTION_PATTERNS

        executor = ActionExecutor(agent, self._app.fa, app=self._app)
        msim = self._app._msim.get(agent.machine_id)
        if msim is not None:
            executor._msim = msim

        # Show what action was detected
        label = next(
            (atype for atype, _ in _ACTION_PATTERNS if atype == action_type),
            action_type
        ).replace("_", " ").title()
        self._app.post_chat(self._mid, "factory",
                            f"────  {label}  ────")

        chat_lines = executor.execute(action_type, parameters)
        for speaker, text in chat_lines:
            self._app.post_chat(self._mid, speaker, text)

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"conversation_{self._mid}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.export_log())


# ══════════════════════════════════════════════════════════════════════════════
#  FACTORY CONVERSATION WINDOW
#  Mirrors AgentConversationWindow but operates at factory level.
#  Opened via factory_gui.py's "Factory Chat" button.
# ══════════════════════════════════════════════════════════════════════════════

FACTORY_ACCENT = "#8b5cf6"   # violet — distinct from all four machine colours


class FactoryConversationWindow(tk.Toplevel):
    """
    Factory-level conversation window.

    Routing (same three-path architecture as AgentConversationWindow):
      Path A  →  factory query    →  FactoryIntentExecutor  (direct answer)
      Path B  →  factory action   →  FactoryActionExecutor  (deterministic)
      Path C  →  LLM advisory     →  FactoryActionExecutor._do_factory_ai_advice()

    The "factory" speaker ID in the chat always refers to the factory agent,
    not an individual machine.
    """

    _FACTORY_MID = "factory"   # pseudo machine_id for post_chat routing

    def __init__(self, parent, factory_gui):
        super().__init__(parent)
        self._app     = factory_gui
        self._streams: dict[str, str] = {}
        self._spin_idx: int = 0
        self._thinking: bool = False

        self.title("🏭  Factory  ↔  Fleet Overview  ↔  OpenAI")
        self.configure(bg=BG)
        self.geometry("860x660")
        self.minsize(640, 480)
        self.resizable(True, True)

        self._build_ui()
        self.after(200, self._tick_spinner)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        accent = FACTORY_ACCENT

        # title bar
        hdr = tk.Frame(self, bg=accent, height=30)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr,
                 text="🏭  Factory  ↔  Fleet Overview  ↔  OpenAI",
                 bg=accent, fg="white",
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=8)
        tk.Button(hdr, text="Export log", bg=accent, fg="white",
                  relief="flat", font=("Helvetica", 8),
                  command=self._export).pack(side="right", padx=6)
        tk.Button(hdr, text="Clear", bg=accent, fg="white",
                  relief="flat", font=("Helvetica", 8),
                  command=self.clear).pack(side="right", padx=2)

        # KPI strip
        kpi_frame = tk.Frame(self, bg=PNL, height=24)
        kpi_frame.pack(fill="x")
        kpi_frame.pack_propagate(False)
        self._kpi_lbl = tk.Label(
            kpi_frame, text="", bg=PNL, fg=DIM,
            font=("Courier", 8), anchor="w"
        )
        self._kpi_lbl.pack(fill="x", padx=8)

        # main pane
        pane = tk.Frame(self, bg=BG)
        pane.pack(fill="both", expand=True)
        pane.rowconfigure(0, weight=1)
        pane.columnconfigure(0, weight=1)

        # conversation log
        self._log = tk.Text(
            pane, bg="#070d15", fg=FG,
            font=("Courier", 10), wrap="word",
            state="disabled", selectbackground=ENTRY,
            padx=6, pady=4,
        )
        sb = ttk.Scrollbar(pane, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        self._log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self._setup_tags()

        # thinking bar
        think_frame = tk.Frame(self, bg=PNL, height=22)
        think_frame.pack(fill="x")
        think_frame.pack_propagate(False)
        self._think_lbl = tk.Label(
            think_frame, text="", bg=PNL, fg=PURPLE,
            font=("Courier", 9), anchor="w"
        )
        self._think_lbl.pack(fill="x", padx=8)

        # operator input
        inp = tk.Frame(self, bg=BG)
        inp.pack(fill="x", padx=4, pady=4)
        self._entry = tk.Entry(
            inp, bg=ENTRY, fg=FG, insertbackground=FG,
            font=("Courier", 10), relief="flat"
        )
        self._entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._entry.bind("<Return>", self._send)
        self._entry.insert(0, "Ask the factory agent…")
        self._entry.bind("<FocusIn>",
            lambda _: (self._entry.delete(0, "end")
                       if self._entry.get().startswith("Ask") else None))
        tk.Button(
            inp, text="Send", bg=FACTORY_ACCENT, fg="white",
            relief="flat", padx=10, command=self._send,
        ).pack(side="right")

        # Quick-access buttons for common queries
        btn_frame = tk.Frame(self, bg=PNL)
        btn_frame.pack(fill="x", padx=4, pady=(0, 4))
        quick = [
            ("Fleet status",      "factory status"),
            ("Active machines",   "which machines are running?"),
            ("Queue",             "what parts are in the queue?"),
            ("Idle time",         "idle time for each machine"),
            ("Tool history",      "tool usage history"),
            ("AI Advice",         "advise me on factory performance"),
        ]
        for label, cmd_text in quick:
            tk.Button(
                btn_frame, text=label, bg=PNL, fg=ACCENT,
                relief="flat", font=("Helvetica", 8),
                command=lambda t=cmd_text: self._quick_send(t),
            ).pack(side="left", padx=2, pady=2)

    def _setup_tags(self) -> None:
        t = self._log
        t.tag_config("ts",        foreground=DIM,    font=("Courier", 8))
        t.tag_config("machine",   foreground=ACCENT, font=("Courier", 10, "bold"))
        t.tag_config("factory",   foreground=GREEN,  font=("Courier", 10, "bold"))
        t.tag_config("openai",    foreground=PURPLE, font=("Courier", 10, "bold"))
        t.tag_config("fallback",  foreground=TEAL,   font=("Courier", 10, "bold"))
        t.tag_config("system",    foreground=DIM)
        t.tag_config("operator",  foreground=ORANGE, font=("Courier", 10, "bold"))
        t.tag_config("body",      foreground=FG)
        t.tag_config("indent",    foreground=DIM, lmargin1=20, lmargin2=20)
        t.tag_config("phase_sep", foreground="#30363d",
                     font=("Courier", 8), spacing1=6, spacing3=4)
        t.tag_config("stream",    foreground="#a0c0ff")

    # ── Public interface ──────────────────────────────────────────────────────

    def append(self, speaker_key: str, text: str) -> None:
        style  = SPEAKER_STYLE.get(speaker_key, SPEAKER_STYLE["system"])
        name, color, pad = style
        name   = name.format(mid="Factory")
        ts     = datetime.datetime.now().strftime("%H:%M:%S")
        pfx    = " " * pad

        self._write(f"{pfx}[{ts}] ", "ts")
        self._write(f"{pfx}{name:<12s}  ", speaker_key)
        lines  = text.split("\n")
        self._write(lines[0] + "\n", "body")
        for line in lines[1:]:
            self._write("               " + " " * pad + line + "\n", "body")

    def clear(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def open(self) -> None:
        self.deiconify()
        self.lift()
        self.focus_set()
        self._refresh_kpi()

    def set_thinking(self, thinking: bool, who: str = "") -> None:
        self._thinking = thinking
        self._think_who = who if thinking else ""
        if not thinking:
            self._think_lbl.configure(text="")

    def export_log(self) -> str:
        return self._log.get("1.0", "end")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _write(self, text: str, tag: str = "body") -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text, tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _tick_spinner(self) -> None:
        if self._thinking:
            frame = _SPIN[self._spin_idx % len(_SPIN)]
            self._think_lbl.configure(
                text=f"  {frame}  {getattr(self, '_think_who', '')}  generating…")
            self._spin_idx += 1
        self.after(140, self._tick_spinner)

    def _refresh_kpi(self) -> None:
        """Update the KPI strip with live fleet data."""
        try:
            kpis = self._app.fa.get_production_kpis()
            self._kpi_lbl.configure(
                text=(
                    f"  Machines: {kpis['machines_running']} run  "
                    f"{kpis['machines_idle']} idle  "
                    f"{kpis['machines_stopped']} stopped  │  "
                    f"Jobs: {kpis['total_jobs_completed']} done  "
                    f"{kpis['scheduler_queue_depth']} queued  "
                    f"{kpis['rework_queue_depth']} rework  │  "
                    f"Tool life avg: {kpis['avg_tool_life_pct']:.1f}%  │  "
                    f"Policy: {self._app.fa.policy}"
                )
            )
        except Exception:
            pass
        # refresh every 5 s
        self.after(5000, self._refresh_kpi)

    def _quick_send(self, text: str) -> None:
        """Inject a pre-defined query as if the operator typed it."""
        self._entry.delete(0, "end")
        self._entry.insert(0, text)
        self._send()

    def _send(self, _=None) -> None:
        txt = self._entry.get().strip()
        if not txt or txt.startswith("Ask"):
            return
        self._entry.delete(0, "end")
        self.append("operator", txt)

        # ── Three-path routing ────────────────────────────────────────────
        import threading
        from factory_intents import (
            classify_factory_message,
            extract_factory_parameters,
            classify_factory_action,
        )

        msg_type, intent_id = classify_factory_message(txt)

        if msg_type == "query":
            # Path A — direct data query
            params = extract_factory_parameters(txt, intent_id)
            threading.Thread(
                target=self._answer_factory_query,
                args=(intent_id, params),
                daemon=True,
            ).start()
        else:
            action_type, action_params = classify_factory_action(txt)
            if action_type != "unknown":
                # Path B — deterministic factory action (inc. AI advisory)
                threading.Thread(
                    target=self._execute_factory_action,
                    args=(action_type, action_params),
                    daemon=True,
                ).start()
            else:
                # Path C — unrecognised; fall through to AI advice
                action_params["question"] = txt
                threading.Thread(
                    target=self._execute_factory_action,
                    args=("factory_ai_advice", action_params),
                    daemon=True,
                ).start()

    def _answer_factory_query(self, intent_id: str, parameters: dict) -> None:
        from factory_intents import FactoryIntentExecutor, FACTORY_INTENT_MAP

        executor = FactoryIntentExecutor(self._app.fa, app=self._app)
        # Inject all _MachSim references so live G-code state is available
        for mid, msim in self._app._msim.items():
            executor._msim[mid] = msim

        result = executor.execute(intent_id, parameters)
        label  = (FACTORY_INTENT_MAP[intent_id].label
                  if intent_id in FACTORY_INTENT_MAP else intent_id)

        self._app.post_factory_chat("factory", f"ℹ {label}")
        self._app.post_factory_chat("factory", result["result"])

    def _execute_factory_action(self, action_type: str, parameters: dict) -> None:
        from factory_intents import FactoryActionExecutor

        self.set_thinking(True, "Factory AI")
        try:
            executor = FactoryActionExecutor(self._app.fa, app=self._app)
            for mid, msim in self._app._msim.items():
                executor._msim[mid] = msim

            label = action_type.replace("_", " ").title()
            self._app.post_factory_chat("factory", f"────  {label}  ────")

            chat_lines = executor.execute(action_type, parameters)
            for speaker, text in chat_lines:
                self._app.post_factory_chat(speaker, text)
        finally:
            self.set_thinking(False)

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"factory_conversation_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.export_log())

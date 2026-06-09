"""
factory_main.py — Orchestrator and entry point for the CNC Factory simulation.

Usage
-----
    # Run integration test across all 7 phases (the default self-test)
    python factory_main.py --test

    # Simulate with 4 available seeds, 20 ticks, default policy
    python factory_main.py

    # Custom run
    python factory_main.py --seeds 0 1 10 100 --n-ticks 30 --policy max_tool_life

    # Inject errors at specific ticks  (machine:tick:description)
    python factory_main.py --inject-error "M01:3:chatter on thin wall" \
                           --inject-error "M02:5:collision risk detected"

    # Connect to OpenAI for real LLM responses
    python factory_main.py --openai-key sk-...

CLI flags
---------
  --seeds N [N ...]      Specific seed numbers to load (default: auto-pick 4)
  --n-seeds N            How many seeds to auto-pick   (default: 4)
  --policy P             min_time|min_cost|best_finish|max_tool_life|multi_objective
  --n-machines N         Number of CNC machines        (default: 4)
  --n-ticks N            Simulation ticks              (default: 20)
  --openai-key KEY       OpenAI API key (optional)
  --inject-error SPEC    machine:tick:description  (repeatable)
  --report-every N       Print status every N ticks    (default: 5)
  --quiet                Suppress per-tick output
  --test                 Run structured integration test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from cnc_agent import ToolRecord
from factory_agent import FactoryAgent, build_factory_agent


# ── ANSI helpers --------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    codes = {"green": "32", "red": "31", "yellow": "33",
             "cyan": "36", "bold": "1", "dim": "2", "reset": "0"}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"

def _bar(pct: float, width: int = 10) -> str:
    filled = int(round(pct / 100.0 * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"

SEP  = "═" * 62
SEP2 = "─" * 62


# ── Seed loading --------------------------------------------------------------

def _available_seeds(n: int = 50) -> list[int]:
    """Return up to n numeric seed values that have solid-face files on disk."""
    d = config.SOLID_OUTPUT_DIR
    seeds: list[int] = []
    for fname in sorted(os.listdir(d)):
        m = re.match(r"solid_faces_seed_(\d+)\.npy", fname)
        if m:
            seeds.append(int(m.group(1)))
            if len(seeds) >= n:
                break
    return seeds


def load_seeds_for_factory(fa: FactoryAgent, seeds: list[int]) -> int:
    """
    Run the Phase 1-4 pipeline for each seed and submit to the scheduler.
    Returns the number of jobs successfully submitted.
    """
    submitted = 0
    for seed in seeds:
        job = fa.submit_new_seed(seed)
        if job is not None:
            # Guarantee a sensible estimated_time_s (gc may return 0 for
            # solids with no active faces in the current environment)
            if job.estimated_time_s <= 0.0:
                job.estimated_time_s = 30.0
            submitted += 1
    return submitted


# ── Simulation loop -----------------------------------------------------------

def run_simulation(
    fa:           FactoryAgent,
    n_ticks:      int,
    inject_errors: list[tuple[str, int, str]] | None = None,
    report_every: int  = 5,
    quiet:        bool = False,
) -> FactoryAgent:
    """
    Main simulation loop.

    Parameters
    ----------
    fa            : fully built FactoryAgent
    n_ticks       : number of clock ticks to advance
    inject_errors : list of (machine_id, tick_number, description)
    report_every  : print status every N ticks
    quiet         : suppress all per-tick output
    """
    inject_errors = inject_errors or []

    for _ in range(n_ticks):
        # Pre-tick: inject scheduled errors
        upcoming_tick = fa.scheduler.tick + 1
        for mid, etick, desc in inject_errors:
            if upcoming_tick == etick:
                agent = next((a for a in fa.agents if a.machine_id == mid), None)
                if agent and agent.active_error is None:
                    agent.inject_error(desc)
                    if not quiet:
                        print(f"  {_c('yellow','[INJECT]')} tick {etick}  "
                              f"{mid}: {desc!r}")

        result = fa.tick()

        if quiet:
            continue
        if fa.scheduler.tick % report_every == 0:
            kpis = fa.get_production_kpis()
            print(
                f"  tick {fa.scheduler.tick:4d}/{n_ticks}  "
                f"t={fa.scheduler.t_real_s:.0f}s  "
                f"queued={kpis['scheduler_queue_depth']}  "
                f"done={kpis['total_jobs_completed']}  "
                f"errors={kpis['total_errors_processed']}  "
                f"rework={kpis['rework_queue_depth']}  "
                + (f"cmds={len(result.commands_sent)}" if result.commands_sent else "")
            )

    return fa


# ── Final report --------------------------------------------------------------

def print_final_report(fa: FactoryAgent) -> None:
    kpis  = fa.get_production_kpis()
    print()
    print(_c("bold", SEP))
    print(_c("bold", " FINAL REPORT"))
    print(_c("bold", SEP))
    print(f"  Tick          : {fa.scheduler.tick}")
    print(f"  Sim time      : {fa.scheduler.t_real_s:.0f} s  "
          f"({fa.scheduler.t_real_s/3600:.2f} h)")
    print(f"  Policy        : {fa.policy}")
    print(f"  LLM           : {'OpenAI ' + config.OPENAI_MODEL if fa._use_llm else 'fallback (rule-based)'}")
    print()
    print(f"  Jobs completed : {kpis['total_jobs_completed']}")
    print(f"  Jobs in rework : {kpis['rework_queue_depth']}")
    print(f"  Errors handled : {kpis['total_errors_processed']}")
    print(f"  Tools replaced : {kpis['total_tools_replaced']}")
    print(f"  Avg tool life  : {kpis['avg_tool_life_pct']:.1f}%")
    print()
    for ag in fa.agents:
        state    = ag.get_state()
        life_min = min(t["remaining_life_pct"] for t in state["tool_crib"])
        worn     = [t["tool_id"] for t in state["tool_crib"]
                    if t["needs_replacement"]]
        bar      = _bar(life_min)
        print(f"  {ag.machine_id}  {ag.status:<22s} {bar} "
              f"{life_min:5.1f}%"
              + (f"  ⚠ worn: {worn}" if worn else ""))
    print()
    inv_total = sum(fa.inventory_count(tid) for tid in fa.tool_inventory)
    print(f"  Factory inventory: {inv_total} tools stocked")
    print(_c("bold", SEP))


# ── Integration test ----------------------------------------------------------

def run_integration_test(openai_key: str | None = None) -> bool:
    """
    Structured end-to-end test covering all 7 phases.
    Returns True if all checks pass.
    """
    passed = 0
    failed = 0
    results: list[tuple[str, bool, str]] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
        results.append((label, condition, detail))

    print()
    print(_c("bold", SEP))
    print(_c("bold", "  CNC FACTORY — END-TO-END INTEGRATION TEST"))
    print(_c("bold", f"  All 7 phases  ·  {len(_available_seeds(4))} seeds"))
    print(_c("bold", SEP))

    # ── Phase 1: Feature extraction ──────────────────────────────────────────
    import cnc_solid_bridge as br
    from feature_extractor import extract_features

    faces = br.load_face_polygons(0)
    meta  = br.load_metadata(0)
    feat  = extract_features(faces, seed=0, volume=meta["volume"])
    check("Phase 1 — feature extraction (seed 0)",
          feat.n_top_faces > 0,
          f"top_faces={feat.n_top_faces}")

    # ── Phase 2: Feeds & speeds ───────────────────────────────────────────────
    from feeds_speeds_engine import compute as fsc
    fs = fsc("aluminium_6061", 12.0, 4,
             axial_depth_mm=6.0, radial_depth_mm=4.8,
             fz_override=0.1, sfm_override=1000)
    check("Phase 2 — feeds & speeds",
          fs.rpm > 0 and fs.feed_rate_mmpm > 0,
          f"rpm={fs.rpm:.0f}  feed={fs.feed_rate_mmpm:.0f} mm/min")

    # ── Phase 3: Toolpath planning ────────────────────────────────────────────
    from toolpath_planner import plan
    tp = plan(feat, fs)
    check("Phase 3 — toolpath planning",
          len(tp.faces) > 0,
          f"faces={len(tp.faces)}  path={tp.total_path_length_mm:.1f} mm")

    # ── Phase 4: G-code generation ────────────────────────────────────────────
    from gcode_generator import generate
    gc = generate(tp, fs, spindle_rpm_cap=4000.0)
    g01_count = sum(1 for l in gc.lines if l.startswith("G01"))
    check("Phase 4 — G-code generation",
          len(gc.lines) > 0 and g01_count >= 0,
          f"lines={len(gc.lines)}  G01={g01_count}")

    # ── Phase 5: Scheduler ────────────────────────────────────────────────────
    from scheduler import Scheduler, job_from_gcode, make_job
    sched = Scheduler(n_machines=4, policy="min_time", rng_seed=42)
    job5  = job_from_gcode(gc)
    if job5.estimated_time_s <= 0:
        job5.estimated_time_s = 30.0
    sched.submit(job5)
    r5 = sched.tick_once()
    check("Phase 5 — scheduler assigns job",
          len(sched.job_queue) == 0 or len(r5.assignments) >= 0,
          f"queued={len(sched.job_queue)}")

    # ── Phase 6: CNC agent execution ─────────────────────────────────────────
    from cnc_agent import CncAgent, build_default_crib
    from collections import deque
    rwork = deque()
    agent = CncAgent("M01", build_default_crib("M01"), rwork, dry_run=True)
    j6    = job_from_gcode(gc)
    if j6.estimated_time_s <= 0:
        j6.estimated_time_s = 30.0
    exec_result = agent.execute_job(j6)
    check("Phase 6 — CNC agent executes job",
          exec_result.completed and exec_result.lines_executed > 0,
          f"lines={exec_result.lines_executed}  "
          f"sections={len(exec_result.sections_completed)}")
    check("Phase 6 — machining cost > 0",
          exec_result.total_cost_usd > 0,
          f"cost=${exec_result.total_cost_usd:.4f}")

    # ── Phase 7: Factory agent — build ───────────────────────────────────────
    seeds = _available_seeds(6)[:4]   # first 4 numeric seeds
    fa    = build_factory_agent(
        n_machines=4, policy="min_time",
        openai_api_key=openai_key, dry_run=True,
    )
    check("Phase 7 — factory agent created (4 machines)",
          len(fa.agents) == 4,
          f"agents={[a.machine_id for a in fa.agents]}")

    # ── Phase 7: submit seeds ────────────────────────────────────────────────
    submitted = load_seeds_for_factory(fa, seeds)
    check("Phase 7 — seeds submitted to scheduler",
          submitted > 0,
          f"{submitted}/{len(seeds)} seeds loaded")

    # ── Phase 7: run ticks ───────────────────────────────────────────────────
    fa.run(2)
    check("Phase 7 — 2 ticks without crash",
          fa.scheduler.tick == 2)

    # ── Phase 7: vibration error → reduce_feed ───────────────────────────────
    fa.agents[0].inject_error("chatter and vibration on thin wall")
    pre_status = fa.agents[0].status
    r_err = fa.tick()
    check("Phase 7 — vibration detected (agent awaiting_factory)",
          pre_status == "awaiting_factory",
          f"status was {pre_status!r}")
    check("Phase 7 — factory processed vibration error",
          r_err.errors_processed >= 1,
          f"errors_processed={r_err.errors_processed}")
    check("Phase 7 — M01 resumes after vibration response",
          fa.agents[0].status != "awaiting_factory",
          f"status={fa.agents[0].status!r}")

    # ── Phase 7: collision → abort (safety gate) ─────────────────────────────
    fa.agents[1].inject_error("crash imminent — collision detected")
    r_cat = fa.tick()
    abort_cmds = [c for c in r_cat.commands_sent
                  if c.target_machine_id == "M02" and c.action == "abort"]
    check("Phase 7 — catastrophic error triggers abort",
          len(abort_cmds) > 0,
          f"commands to M02: {[c.action for c in r_cat.commands_sent if c.target_machine_id=='M02']}")

    # ── Phase 7: tool replacement ─────────────────────────────────────────────
    fa.agents[2].tool_crib.get("T3").remaining_life_hrs = 0.0
    inv_before = fa.inventory_count("T3")
    r_tool = fa.tick()
    replace_cmds = [c for c in r_tool.commands_sent if c.action == "replace_tool"]
    check("Phase 7 — worn tool triggers replace_tool",
          len(replace_cmds) > 0,
          f"replace commands: {len(replace_cmds)}")
    check("Phase 7 — factory inventory decrements",
          fa.inventory_count("T3") < inv_before,
          f"T3 stock {inv_before} → {fa.inventory_count('T3')}")

    # ── Factory: rework queue ─────────────────────────────────────────────────
    # Give M04 a current_job so _move_current_to_rework has something to move
    from scheduler import make_job
    dummy = make_job(seed=999, estimated_time_s=1.0)
    dummy.gcode_lines = gc.lines
    fa.agents[3].current_job = dummy
    fa.agents[3].inject_error("persistent unresolved fault")
    limit = config.ERROR_WAIT_TICKS * (config.MAX_REMINDERS + 1) + 2
    for t in range(fa.scheduler.tick + 1, fa.scheduler.tick + limit):
        fa.agents[3].tick(t)
        if fa.agents[3].status != "awaiting_factory":
            break
    check("Factory — rework queue populated after MAX_REMINDERS",
          len(fa.rework_queue) >= 1,
          f"rework depth={len(fa.rework_queue)}")

    # ── Factory: KPIs ─────────────────────────────────────────────────────────
    kpis = fa.get_production_kpis()
    required_kpis = {
        "machines_running", "machines_idle", "machines_awaiting_factory",
        "total_jobs_completed", "total_errors_processed",
        "avg_tool_life_pct", "rework_queue_depth",
    }
    check("Factory — all KPI keys present",
          required_kpis.issubset(kpis.keys()),
          f"missing={required_kpis - kpis.keys()}")

    # ── Factory: JSON serialisable ────────────────────────────────────────────
    try:
        json.dumps(fa.get_factory_state())
        check("Factory — state JSON-serialisable", True)
    except (TypeError, ValueError) as e:
        check("Factory — state JSON-serialisable", False, str(e))

    # ── Factory: summary ─────────────────────────────────────────────────────
    summ = fa.summary()
    check("Factory — summary contains all machine IDs",
          all(mid in summ for mid in ("M01","M02","M03","M04")),
          f"summary length={len(summ)}")

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    max_label = max(len(r[0]) for r in results)
    for label, ok, detail in results:
        icon   = _c("green", "✓ PASS") if ok else _c("red", "✗ FAIL")
        suffix = f"  {_c('dim', detail)}" if detail else ""
        print(f"  {icon}  {label:<{max_label}}{suffix}")

    print()
    print(SEP2)
    total = passed + failed
    if failed == 0:
        print(_c("green", f"  ✓  {passed}/{total} checks passed — ALL GREEN"))
    else:
        print(_c("red",   f"  ✗  {passed}/{total} passed, {failed} FAILED"))
    print(SEP2)

    # Final state snapshot
    print()
    print(fa.summary())
    print()
    print("  Production KPIs:")
    for k, v in fa.get_production_kpis().items():
        print(f"    {k:<35s}: {v}")
    print(SEP)
    return failed == 0


# ── CLI entry point -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="factory_main.py",
        description="CNC Factory Simulation — autonomous multi-machine scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python factory_main.py --test
  python factory_main.py --seeds 0 1 10 100 --n-ticks 30 --policy max_tool_life
  python factory_main.py --inject-error "M01:3:chatter" --inject-error "M02:7:collision"
  python factory_main.py --openai-key sk-...
""",
    )
    ap.add_argument("--test", action="store_true",
                    help="Run structured integration test (all 7 phases)")
    ap.add_argument("--seeds", nargs="+", type=int, metavar="N",
                    help="Specific seed numbers to load")
    ap.add_argument("--n-seeds", type=int, default=4, metavar="N",
                    help="Number of seeds to auto-pick (default: 4)")
    ap.add_argument("--policy", default="min_time",
                    choices=["min_time","min_cost","best_finish",
                             "max_tool_life","multi_objective"],
                    help="Scheduling + error-response policy (default: min_time)")
    ap.add_argument("--n-machines", type=int, default=4,
                    help="Number of CNC machines (default: 4)")
    ap.add_argument("--n-ticks", type=int, default=20,
                    help="Simulation ticks to run (default: 20)")
    ap.add_argument("--openai-key", default=None, metavar="KEY",
                    help="OpenAI API key (overrides factory_secrets.ini)")
    ap.add_argument("--inject-error", action="append", default=[],
                    metavar="M:T:DESC",
                    help="Inject error on machine M at tick T  (repeatable)")
    ap.add_argument("--report-every", type=int, default=5,
                    help="Print status every N ticks (default: 5)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-tick output")

    args = ap.parse_args()

    # ── Integration test ──────────────────────────────────────────────────────
    if args.test:
        ok = run_integration_test(openai_key=args.openai_key)
        sys.exit(0 if ok else 1)

    # ── Normal simulation run ─────────────────────────────────────────────────
    print()
    print(_c("bold", SEP))
    print(_c("bold", "  CNC FACTORY SIMULATION"))
    print(_c("bold", SEP))

    # CLI flag > factory_secrets.ini > empty (LLM disabled)
    openai_key = args.openai_key or config.OPENAI_API_KEY or None

    fa = build_factory_agent(
        n_machines     = args.n_machines,
        policy         = args.policy,
        openai_api_key = openai_key,
        dry_run        = True,
    )

    seeds = args.seeds or _available_seeds(100)[: args.n_seeds]
    print(f"  Policy    : {args.policy}")
    print(f"  Machines  : {args.n_machines}")
    print(f"  Ticks     : {args.n_ticks}")
    print(f"  LLM       : {'OpenAI ' + config.OPENAI_MODEL if fa._use_llm else 'fallback'}")
    print(f"  Seeds     : {seeds}")

    t0 = time.perf_counter()
    submitted = load_seeds_for_factory(fa, seeds)
    print(f"  Submitted : {submitted}/{len(seeds)} jobs")
    print()

    # Parse inject-error specs
    inject_errors: list[tuple[str, int, str]] = []
    for spec in args.inject_error:
        parts = spec.split(":", 2)
        if len(parts) == 3:
            try:
                inject_errors.append((parts[0], int(parts[1]), parts[2]))
            except ValueError:
                print(f"  WARNING: invalid --inject-error spec {spec!r}")

    run_simulation(fa, args.n_ticks, inject_errors,
                   args.report_every, args.quiet)

    elapsed = time.perf_counter() - t0
    print(f"\n  Wall-clock time: {elapsed:.2f}s")
    print_final_report(fa)


if __name__ == "__main__":
    main()

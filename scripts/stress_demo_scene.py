#!/usr/bin/env python3
"""Headless stress-test harness for the firefighting autonomy stack on ``demo_scene``.

This drives the full stack -- natural-language mission parsing
(:func:`ember.mission.parse_mission`) -> mission FSM
(:class:`ember.executor.MissionExecutor`) -> locomotion + spray sim
(:class:`ember.sim.G1Sim`) -- on the ``demo_scene`` house, varying the robot's
START location and the operator prompt, and asserts that the autonomy reaches the
expected outcome (right fires out, optional return-home, no falls / stalls /
external pauses).

RENDER IS REQUIRED (important caveat)
-------------------------------------
The water-jet ballistics, the fire hit-test and the fire-health decrement all
happen inside the *render* thread (``G1Sim.render_once`` -> ``_update_jet``).
So every scenario MUST run with ``sim.run(render=True)`` (the default); with
rendering off the hose never connects and fires never go out. This harness is
therefore only meaningful on a box with a working (headless) GL backend; we set
``MUJOCO_GL=egl`` before importing mujoco.

Determinism
-----------
We pop ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` so mission parsing always falls
back to the deterministic rule-based parser, and we additionally pass
``use_llm=False`` everywhere. The harness also records the *parsed plan* for each
scenario so a human can see exactly how the parser mapped each prompt.

Scene variants
--------------
Only the robot START pose is varied (``dataclasses.replace`` on the frozen
``SceneSpec``); walls / fires / terrain / debris / home stay identical to the
base ``demo_scene``. Each candidate start is run through ``scenegen.validate_spec``
and skipped if it does not validate (so fires/home stay reachable). Each valid
variant is written to its own ``stress_<label>`` XML pair in the model dir; the
``--cleanup`` / ``--cleanup-only`` steps delete those generated XMLs. No JSON is
ever written into ``scenes/specs/``.

Usage
-----
    python scripts/stress_demo_scene.py                  # curated matrix
    python scripts/stress_demo_scene.py --list           # show scenarios, no run
    python scripts/stress_demo_scene.py --full           # all prompts x all starts
    python scripts/stress_demo_scene.py --starts yard_center,room_A
    python scripts/stress_demo_scene.py --prompts all_1,two_return
    python scripts/stress_demo_scene.py --timeout 240
    python scripts/stress_demo_scene.py --json /tmp/out.json
    python scripts/stress_demo_scene.py --cleanup-only   # delete stress_* XMLs

Exit code is 0 iff every run scenario passed, else 1.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# MUST be set before mujoco / ember.sim are imported so the render thread (which
# does the jet hit-test) can create a GL context headlessly.
os.environ.setdefault("MUJOCO_GL", "egl")
# Pin EGL to the discrete NVIDIA GPU (device 0 here). Without this MuJoCo's EGL
# picks the first device that initializes -- often llvmpipe (CPU software GL,
# ~60x slower) -- which both starves parallel workers AND breaks the real-time
# pacing the wall-clock-timed executor relies on. Override per-box if device 0
# is not your GPU (check `eglinfo -B`). Set EMBER_EGL_DEVICE to change/disable.
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID",
                      os.environ.get("EMBER_EGL_DEVICE", "0"))
# Force the deterministic rule-based mission parser (no cloud LLM).
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

import _bootstrap  # noqa: F401,E402  (puts src/ on sys.path)

from ember import scenegen, scenes  # noqa: E402
from ember.config import G1_MODEL_DIR  # noqa: E402
from ember.executor import MissionExecutor  # noqa: E402
from ember.mission import parse_mission  # noqa: E402
from ember.sim import G1Sim  # noqa: E402
from ember.spec import from_json  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO / "scenes" / "specs" / "demo_scene.json"
DEFAULT_JSON = "/tmp/stress_demo_scene.json"

# Health at/below this counts a fire as "out" (matches the executor's verify).
OUT_THRESHOLD = 0.05
# Settle time after launching the physics thread before parsing/executing.
SETTLE_S = 0.6
# Poll cadence of the per-scenario watch loop.
POLL_S = 0.25
# Return-home success tolerance (m).
HOME_TOL = 0.6
# A VISIT must not damage any fire below this health.
VISIT_HEALTH_FLOOR = 0.95
# Stall detector. We trust the SIM's own ``blocked`` signal (set when the robot
# is commanded to translate but does not move >5 cm over a 2 s *sim-time* window
# -- the production "stuck" definition, immune to wall-clock contention and to
# in-place rotation during the face/spray phases). If ``blocked`` stays True
# continuously for this long, the robot is genuinely wedged.
STUCK_SECONDS = 10.0
# Hard backstop: no x/y progress at all over this wall window while a mission is
# actively navigating (catches a dead/never-started sim).
NOPROGRESS_WINDOW_S = 45.0
NOPROGRESS_DISP_M = 0.3
# Message substrings that signal an autonomy failure (case-insensitive).
FAIL_KEYWORDS = ("fail", "timeout", "could not", "no path", "verify failed")

# Candidate robot starts (label, x, y, yaw[rad]); yaw faces toward the house
# interior (+x = 0.0). Some interior-room starts may not validate (debris/door
# clearance); those are skipped with a log line.
CANDIDATE_STARTS = (
    ("yard_center", -1.0, -1.4, 0.0),
    ("yard_low", -1.5, -2.6, 0.3),
    ("yard_high", -1.5, 2.6, -0.3),
    ("yard_mid", -0.2, 0.0, 0.0),
    ("room_C", 2.3, -1.0, 1.57),
    ("room_A", 2.3, 1.0, -1.57),
    ("room_D", 5.05, -1.0, 1.57),
    ("room_B", 5.05, 1.0, -1.57),
)

# Prompt scenarios. ``expect_out`` is resolved at runtime to a concrete index set
# (see :func:`resolve_expected`): "all" -> every fire; "count:N" -> the N fires
# nearest to home (same rule as mission.py); a frozenset -> those exact indices.
PROMPTS = (
    {"id": "all_1", "prompt": "put out all fires",
     "expect_out": "all", "expect_return_home": False, "involves_search": False},
    {"id": "all_2", "prompt": "extinguish every fire",
     "expect_out": "all", "expect_return_home": False, "involves_search": False},
    {"id": "all_return", "prompt": "put out all fires and return home",
     "expect_out": "all", "expect_return_home": True, "involves_search": False},
    {"id": "one", "prompt": "put out one fire",
     "expect_out": "count:1", "expect_return_home": False, "involves_search": False},
    {"id": "two", "prompt": "put out two fires",
     "expect_out": "count:2", "expect_return_home": False, "involves_search": False},
    {"id": "two_return", "prompt": "put out two fires and return home",
     "expect_out": "count:2", "expect_return_home": True, "involves_search": False},
    {"id": "three_return", "prompt": "put out three fires then return home",
     "expect_out": "count:3", "expect_return_home": True, "involves_search": False},
    {"id": "fire3", "prompt": "extinguish fire 3",
     "expect_out": frozenset({2}), "expect_return_home": False, "involves_search": False},
    {"id": "fires12", "prompt": "put out fires 1 and 2",
     "expect_out": frozenset({0, 1}), "expect_return_home": False, "involves_search": False},
    {"id": "visit2_return", "prompt": "visit fire 2 then return home",
     "expect_out": frozenset(), "expect_return_home": True, "involves_search": False,
     "is_visit": True},
    {"id": "one_map_return",
     "prompt": "put out one fire then map the area then return home",
     "expect_out": "count:1", "expect_return_home": True, "involves_search": True},
    {"id": "one_return", "prompt": "put out a fire and come back",
     "expect_out": "count:1", "expect_return_home": True, "involves_search": False},
    # A few more sensible variants for breadth.
    {"id": "all_come_back", "prompt": "extinguish all fires then come back",
     "expect_out": "all", "expect_return_home": True, "involves_search": False},
    {"id": "fire1", "prompt": "extinguish fire 1",
     "expect_out": frozenset({0}), "expect_return_home": False, "involves_search": False},
    {"id": "both_return", "prompt": "put out both fires then return home",
     "expect_out": "count:2", "expect_return_home": True, "involves_search": False},
)

# Focused prompts run from every (non yard_center) valid start in the curated matrix.
FOCUS_PROMPT_IDS = ("all_1", "two_return", "fire3")


@dataclasses.dataclass
class Scenario:
    start_label: str
    variant: object  # SceneSpec
    prompt_def: dict
    expected: frozenset  # resolved fire indices expected extinguished


# -- expected-outcome resolution ------------------------------------------- #

def nearest_to_home_order(home, fires) -> list[int]:
    """Fire indices sorted by squared distance to home (mission.py's tie-break)."""
    hx, hy = home
    return sorted(
        range(len(fires)),
        key=lambda i: (fires[i][0] - hx) ** 2 + (fires[i][1] - hy) ** 2,
    )


def resolve_expected(expect_out, order: list[int], n_fires: int) -> frozenset:
    """Turn an ``expect_out`` spec into the concrete set of extinguished indices."""
    if expect_out == "all":
        return frozenset(range(n_fires))
    if isinstance(expect_out, str) and expect_out.startswith("count:"):
        n = int(expect_out.split(":", 1)[1])
        return frozenset(order[: max(0, n)])
    return frozenset(expect_out)


# -- scene variants --------------------------------------------------------- #

def make_variant(base, label: str, x: float, y: float, yaw: float):
    """Frozen-safe copy of ``base`` with only name + start changed."""
    return dataclasses.replace(base, name=f"stress_{label}",
                               start=(float(x), float(y), float(yaw)))


def valid_variants(base) -> list[tuple[str, object]]:
    """Build + validate every candidate start; keep only the ones that validate."""
    kept: list[tuple[str, object]] = []
    for label, x, y, yaw in CANDIDATE_STARTS:
        variant = make_variant(base, label, x, y, yaw)
        try:
            scenegen.validate_spec(variant)
        except ValueError as exc:
            print(f"[skip] start {label} ({x}, {y}, {yaw}) invalid: {exc}")
            continue
        kept.append((label, variant))
    return kept


def build_variant_xml(variant) -> None:
    """Write the variant's 12/29-DOF XML pair into the model dir."""
    scenes.build(variant)


def cleanup_generated() -> list[str]:
    """Delete every generated ``stress_*`` XML in the model dir."""
    if not G1_MODEL_DIR.exists():
        return []
    deleted: list[str] = []
    for path in sorted(G1_MODEL_DIR.glob("g1_stress_*")):
        try:
            path.unlink()
            deleted.append(str(path))
        except OSError as exc:
            print(f"[cleanup] could not delete {path}: {exc}")
    return deleted


# -- single-scenario execution --------------------------------------------- #

def _xy(state) -> tuple[float, float]:
    pos = state["pos"]
    return float(pos[0]), float(pos[1])


def _watch_loop(sim, ex, timeout: float) -> dict:
    """Poll the executor + sim until done / timeout / stall / pause.

    Returns the observations collected during the run (flags + messages +
    elapsed time + final state)."""
    messages: set[str] = set()
    samples: deque[tuple[float, float, float]] = deque()
    fell = stalled = paused = timed_out = done = False
    blocked_since: float | None = None

    last_state = None
    last_phase = None
    last_task = None

    start = time.monotonic()
    deadline = start + timeout
    while True:
        status = ex.status()
        st = sim.get_state()
        now = time.monotonic()
        last_state = status.get("state")
        last_phase = st.get("approach_phase")
        last_task = status.get("task_index")

        msg = status.get("message") or ""
        if msg:
            messages.add(msg)
        if st.get("fell"):
            fell = True

        state = status.get("state")
        if status.get("done"):
            done = True
            break
        if state == "paused":
            paused = True
            break

        # Primary stall signal: the sim's own ``blocked`` flag held continuously.
        if st.get("blocked"):
            if blocked_since is None:
                blocked_since = now
            elif now - blocked_since >= STUCK_SECONDS:
                stalled = True
                break
        else:
            blocked_since = None

        # Backstop: zero net travel over a long window while actively moving.
        # Reset the baseline during phases where standing still is expected
        # (spray/verify, or any in-place spraying) so a long hose-down on one
        # fire is never mistaken for a wedge.
        if st.get("spraying") or state in ("spray", "verify"):
            samples.clear()
        else:
            x, y = _xy(st)
            samples.append((now, x, y))
            while samples and now - samples[0][0] > NOPROGRESS_WINDOW_S:
                samples.popleft()
            if samples and now - samples[0][0] >= NOPROGRESS_WINDOW_S * 0.95:
                x0, y0 = samples[0][1], samples[0][2]
                if math.hypot(x - x0, y - y0) < NOPROGRESS_DISP_M:
                    stalled = True
                    break

        if now >= deadline:
            timed_out = True
            break
        time.sleep(POLL_S)

    elapsed = time.monotonic() - start
    final_state = sim.get_state()
    return {
        "messages": messages,
        "fell": fell,
        "stalled": stalled,
        "paused": paused,
        "timed_out": timed_out,
        "done": done,
        "elapsed_s": elapsed,
        "final_state": final_state,
        "last_fsm_state": last_state,
        "last_phase": last_phase,
        "last_task_index": last_task,
        "blocked_at_end": bool(final_state.get("blocked")),
    }


def _evaluate(scn: Scenario, base, obs: dict, final_health: list[float]) -> dict:
    """Apply the PASS/FAIL rules to one finished run; returns result fields."""
    extinguished = frozenset(
        i for i, h in enumerate(final_health) if h <= OUT_THRESHOLD)
    fail: list[str] = []
    warnings: dict = {}

    if obs["fell"]:
        fail.append("fell")
    if obs["stalled"]:
        fail.append("stalled")
    if obs["paused"]:
        fail.append("paused (external override)")
    if obs["timed_out"]:
        fail.append("timed_out")

    bad_msgs = sorted(
        m for m in obs["messages"]
        if any(k in m.lower() for k in FAIL_KEYWORDS))
    if bad_msgs:
        fail.append("failure message(s): " + " | ".join(bad_msgs))

    # Extinguished-set check (relaxed for search: warn, don't fail).
    if scn.prompt_def.get("involves_search"):
        extra = sorted(extinguished - scn.expected)
        missing = sorted(scn.expected - extinguished)
        if extra:
            warnings["search_extra_extinguished"] = extra
        if missing:
            warnings["search_missing_expected"] = missing
    else:
        if extinguished != scn.expected:
            fail.append(
                f"extinguished {sorted(extinguished)} != expected "
                f"{sorted(scn.expected)}")

    # VISIT must not damage any fire.
    if scn.prompt_def.get("is_visit"):
        damaged = [i for i, h in enumerate(final_health) if h < VISIT_HEALTH_FLOOR]
        if damaged:
            fail.append(f"visit damaged fires {damaged}")
            warnings["visit_violation"] = damaged

    # Return-home check.
    home_dist = None
    if scn.prompt_def.get("expect_return_home"):
        hx, hy = base.home
        x, y = _xy(obs["final_state"])
        home_dist = math.hypot(x - hx, y - hy)
        if home_dist >= HOME_TOL:
            fail.append(f"home_dist {home_dist:.2f} m >= {HOME_TOL} m")

    return {
        "extinguished": sorted(extinguished),
        "home_dist": home_dist,
        "warnings": warnings,
        "fail_reasons": fail,
        "passed": not fail,
    }


def _run_one_payload(payload):
    """Picklable entry point for a pool worker: rebuild the Scenario and run it.

    Each scenario builds its own ``G1Sim`` (its own EGL context), so scenarios
    are independent and safe to fan out across separate processes. The variant
    XMLs are written ONCE by the parent before dispatch, so workers only read.
    """
    start_label, variant, prompt_def, expected, base, timeout = payload
    scn = Scenario(start_label, variant, prompt_def, expected)
    return run_scenario(scn, base, timeout)


def run_scenario(scn: Scenario, base, timeout: float) -> dict:
    """Run one (start, prompt) scenario end-to-end; never raises."""
    variant = scn.variant
    name_12, name_29 = scenes.spec_scene_names(variant.name)
    mission = parse_mission(scn.prompt_def["prompt"], variant, use_llm=False)
    parsed_plan = [(t.type.value, t.target) for t in mission]

    result = {
        "start_label": scn.start_label,
        "prompt_id": scn.prompt_def["id"],
        "prompt": scn.prompt_def["prompt"],
        "parsed_plan": parsed_plan,
        "expected_out": sorted(scn.expected),
        "expect_return_home": bool(scn.prompt_def.get("expect_return_home")),
        "involves_search": bool(scn.prompt_def.get("involves_search")),
    }

    sim = None
    ex = None
    thread = None
    try:
        sim = G1Sim(scene_path=str(G1_MODEL_DIR / name_12),
                    overlay_scene=str(G1_MODEL_DIR / name_29),
                    spec=variant)
        sim.heading_hold = False

        thread = threading.Thread(target=sim.run, daemon=True)
        thread.start()
        time.sleep(SETTLE_S)

        init_state = sim.get_state()
        init_health = list(sim.fire_health)

        ex = MissionExecutor(sim, mission, tick_hz=20.0)
        ex.start()
        obs = _watch_loop(sim, ex, timeout)

        final_health = list(sim.fire_health)
        result.update(_evaluate(scn, base, obs, final_health))
        result.update({
            "initial_health": [round(h, 3) for h in init_health],
            "final_health": [round(h, 3) for h in final_health],
            "fell": obs["fell"],
            "stalled": obs["stalled"],
            "paused": obs["paused"],
            "timed_out": obs["timed_out"],
            "done": obs["done"],
            "elapsed_s": round(obs["elapsed_s"], 2),
            "messages": sorted(obs["messages"]),
            "initial_pos": [round(float(c), 3) for c in init_state["pos"][:2]],
            "final_pos": [round(float(c), 3) for c in obs["final_state"]["pos"][:2]],
            "last_fsm_state": obs.get("last_fsm_state"),
            "last_phase": obs.get("last_phase"),
            "last_task_index": obs.get("last_task_index"),
            "blocked_at_end": obs.get("blocked_at_end"),
        })
    except Exception:  # one crash must not abort the whole matrix
        result.setdefault("messages", [])
        result.setdefault("warnings", {})
        result.update({
            "passed": False,
            "fail_reasons": ["exception:\n" + traceback.format_exc()],
            "final_health": result.get("final_health"),
            "extinguished": result.get("extinguished"),
        })
    finally:
        if ex is not None:
            try:
                ex.stop()
            except Exception:
                pass
        if sim is not None:
            try:
                sim.stop()
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=5.0)

    return result


# -- matrix construction ---------------------------------------------------- #

def build_matrix(valid: list[tuple[str, object]], base, args) -> list[Scenario]:
    """Resolve the (start, prompt) pairs to run, honoring CLI selection."""
    valid_labels = [label for label, _ in valid]
    variant_by_label = dict(valid)
    prompt_by_id = {p["id"]: p for p in PROMPTS}
    order = nearest_to_home_order(base.home, base.fires)
    n_fires = len(base.fires)

    starts_sel = _split_csv(args.starts)
    prompts_sel = _split_csv(args.prompts)
    for s in starts_sel:
        if s not in valid_labels:
            print(f"[warn] --starts '{s}' is not a valid start; ignoring")
    for p in prompts_sel:
        if p not in prompt_by_id:
            print(f"[warn] --prompts '{p}' is not a known prompt id; ignoring")

    pairs: list[tuple[str, str]] = []
    if args.full or starts_sel or prompts_sel:
        starts = ([s for s in starts_sel if s in valid_labels]
                  if starts_sel else valid_labels)
        pids = ([p for p in prompts_sel if p in prompt_by_id]
                if prompts_sel else [p["id"] for p in PROMPTS])
        pairs = [(s, p) for s in starts for p in pids]
    else:
        # Curated matrix: every prompt from yard_center, focus prompts elsewhere.
        anchor = "yard_center"
        if anchor in valid_labels:
            pairs += [(anchor, p["id"]) for p in PROMPTS]
        for label in valid_labels:
            if label == anchor:
                continue
            pairs += [(label, pid) for pid in FOCUS_PROMPT_IDS]

    scenarios: list[Scenario] = []
    seen: set[tuple[str, str]] = set()
    for start_label, pid in pairs:
        if (start_label, pid) in seen:
            continue
        seen.add((start_label, pid))
        pdef = prompt_by_id[pid]
        expected = resolve_expected(pdef["expect_out"], order, n_fires)
        scenarios.append(Scenario(start_label, variant_by_label[start_label],
                                  pdef, expected))
    return scenarios


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [tok.strip() for tok in value.split(",") if tok.strip()]


# -- reporting -------------------------------------------------------------- #

def print_scenario_line(result: dict) -> None:
    tag = "PASS" if result["passed"] else "FAIL"
    base = (f"[{tag}] {result['start_label']:<11} {result['prompt_id']:<14} "
            f"ext={result.get('extinguished')} "
            f"exp={result.get('expected_out')} "
            f"t={result.get('elapsed_s')}s")
    if result.get("warnings"):
        base += f" warn={result['warnings']}"
    if not result["passed"]:
        base += " :: " + " ; ".join(result["fail_reasons"])
    print(base, flush=True)


def print_summary(results: list[dict]) -> None:
    n = len(results)
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    print("\n" + "=" * 72)
    print(f"SUMMARY: {len(passed)}/{n} passed, {len(failed)} failed")
    if failed:
        print("\nFailing scenarios:")
        for r in failed:
            print(f"  - {r['start_label']}/{r['prompt_id']}: "
                  + " ; ".join(r["fail_reasons"]))
    print("=" * 72)


def print_scenario_list(scenarios: list[Scenario]) -> None:
    print(f"{len(scenarios)} scenarios:")
    for scn in scenarios:
        pd = scn.prompt_def
        print(f"  {scn.start_label:<11} {pd['id']:<14} "
              f"expect={sorted(scn.expected)} "
              f"return_home={bool(pd.get('expect_return_home'))} "
              f"search={bool(pd.get('involves_search'))}  | {pd['prompt']!r}")


# -- CLI -------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true",
                   help="run the full cross product (all prompts x all valid starts)")
    p.add_argument("--starts", default=None,
                   help="comma list of start labels to use")
    p.add_argument("--prompts", default=None,
                   help="comma list of prompt ids to use")
    p.add_argument("--timeout", type=float, default=200.0,
                   help="per-scenario timeout in seconds (default 200)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes (each its own sim/GL context); "
                        "default 1 = sequential")
    p.add_argument("--list", action="store_true",
                   help="print the resolved scenario list and exit (no running)")
    p.add_argument("--cleanup-only", action="store_true",
                   help="delete generated stress_* XMLs and exit")
    p.add_argument("--cleanup", action="store_true",
                   help="delete generated stress_* XMLs after the run completes")
    p.add_argument("--json", default=DEFAULT_JSON,
                   help=f"path for the JSON results (default {DEFAULT_JSON})")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.cleanup_only:
        deleted = cleanup_generated()
        print(f"deleted {len(deleted)} generated XML file(s)")
        for d in deleted:
            print(f"  {d}")
        return 0

    base = from_json(SPEC_PATH)
    order = nearest_to_home_order(base.home, base.fires)
    print(f"nearest-to-home fire order: {order}")

    valid = valid_variants(base)
    if not valid:
        print("no candidate start validated; nothing to run")
        return 1
    print(f"valid starts: {[label for label, _ in valid]}")

    scenarios = build_matrix(valid, base, args)
    if not scenarios:
        print("no scenarios selected")
        return 1

    if args.list:
        print_scenario_list(scenarios)
        return 0

    # Build XML only for the starts we will actually run.
    if not G1_MODEL_DIR.exists():
        print(f"model dir not found: {G1_MODEL_DIR}; cannot run scenarios")
        return 1
    needed_labels = {scn.start_label for scn in scenarios}
    variant_by_label = dict(valid)
    for label in sorted(needed_labels):
        build_variant_xml(variant_by_label[label])

    workers = max(1, min(args.workers, len(scenarios)))
    print(f"running {len(scenarios)} scenarios (timeout {args.timeout}s each, "
          f"{workers} worker(s))\n", flush=True)
    results: list[dict] = []
    if workers > 1:
        payloads = [(s.start_label, s.variant, s.prompt_def, s.expected, base,
                     args.timeout) for s in scenarios]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as exr:
            futs = [exr.submit(_run_one_payload, p) for p in payloads]
            for fut in as_completed(futs):
                result = fut.result()
                results.append(result)
                print_scenario_line(result)
        order_key = {(s.start_label, s.prompt_def["id"]): i
                     for i, s in enumerate(scenarios)}
        results.sort(key=lambda r: order_key.get(
            (r["start_label"], r["prompt_id"]), 1 << 30))
    else:
        for scn in scenarios:
            result = run_scenario(scn, base, args.timeout)
            results.append(result)
            print_scenario_line(result)

    print_summary(results)

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nresults written to {out_path}")

    if args.cleanup:
        deleted = cleanup_generated()
        print(f"cleaned up {len(deleted)} generated XML file(s)")

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

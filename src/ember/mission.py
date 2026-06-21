"""Natural-language mission parsing for the firefighter.

A ``Mission`` is an ordered queue of ``Task`` s over a small fixed verb set
(VISIT / EXTINGUISH / SEARCH / RETURN). :func:`parse_mission` turns a free-text
prompt into a validated ``Mission`` using a cloud LLM when available, falling
back to :func:`_rule_based` so the demo never hard-fails offline.

This module is the single source of the mission data types; the executor
(:mod:`ember.executor`) consumes them. Targets are *fire indices* into
``spec.fires`` (or ``None`` meaning "all"/"n/a"), clamped to the scene.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum

from .spec import SceneSpec


class TaskType(Enum):
    VISIT = "visit"            # go to a standoff near a fire, don't spray
    EXTINGUISH = "extinguish"  # approach + spray until the fire is out
    SEARCH = "search"          # coverage tour until a fire is detected
    RETURN = "return"          # go back to home


@dataclass(frozen=True)
class Task:
    type: TaskType
    target: int | None = None  # fire index for VISIT/EXTINGUISH; None = all / n/a


@dataclass(frozen=True)
class Mission:
    tasks: tuple[Task, ...]

    def __bool__(self) -> bool:
        return bool(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)


def scene_summary(spec: SceneSpec) -> str:
    """Compact NL description of the scene for the LLM system prompt."""
    fires = ", ".join(f"fire {i} at ({fx:.1f}, {fy:.1f})"
                      for i, (fx, fy) in enumerate(spec.fires))
    xmin, xmax, ymin, ymax = spec.bounds
    return (f"Scene '{spec.name}': bounds x[{xmin:.1f},{xmax:.1f}] "
            f"y[{ymin:.1f},{ymax:.1f}]; home at {spec.home}; "
            f"{len(spec.fires)} known fires: {fires or 'none'}.")


def _clamp_target(target: int | None, spec: SceneSpec) -> int | None:
    if target is None:
        return None
    n = len(spec.fires)
    if n == 0:
        return None
    return max(0, min(int(target), n - 1))


def _validate_tasks(raw_tasks: list[object], spec: SceneSpec) -> tuple[Task, ...]:
    """Coerce, clamp, and drop malformed task dicts (shared by LLM and rule paths)."""
    validated: list[Task] = []
    n_fires = len(spec.fires)

    for raw in raw_tasks:
        if isinstance(raw, Task):
            entry: dict[str, object] = {"type": raw.type.value, "target": raw.target}
        elif isinstance(raw, dict):
            entry = raw
        else:
            continue

        type_raw = entry.get("type")
        if type_raw is None:
            continue
        try:
            ttype = TaskType(str(type_raw).lower().strip())
        except ValueError:
            continue

        target = entry.get("target")
        if target is not None:
            try:
                target = int(target)
            except (TypeError, ValueError):
                if ttype in (TaskType.VISIT, TaskType.EXTINGUISH):
                    continue
                target = None

        if ttype in (TaskType.SEARCH, TaskType.RETURN):
            target = None
        elif ttype == TaskType.VISIT:
            if target is None:
                target = 0 if n_fires else None
            else:
                target = _clamp_target(target, spec)
        elif ttype == TaskType.EXTINGUISH:
            if target is not None:
                target = _clamp_target(target, spec)

        validated.append(Task(ttype, target))

    return tuple(validated)


def _fire_numbers(text: str) -> list[int]:
    """Extract 1-based fire numbers from text; return 0-based indices (deduped)."""
    one_based: list[int] = []
    for n in re.findall(r"\bfires?\s+(\d+)\b", text):
        one_based.append(int(n))
    if re.search(r"\bfires?\s+\d+", text):
        for n in re.findall(r"\band\s+(\d+)\b", text):
            one_based.append(int(n))
    seen: set[int] = set()
    indices: list[int] = []
    for n in one_based:
        if n not in seen:
            seen.add(n)
            indices.append(n - 1)
    return indices


def _all_fires_intent(text: str) -> bool:
    return bool(re.search(
        r"\b(all|every|everything)\b(?:\s+\w+){0,3}\s*fires?\b"
        r"|\bfires?\b(?:\s+\w+){0,3}\s*\b(all|every|everything)\b"
        r"|\b(all|every|everything)\b",
        text,
    ))


# Quantity words for "<N> fires" (count BEFORE the noun, vs. "fire <N>" index).
_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "single": 1,
    "two": 2, "both": 2, "couple": 2, "pair": 2,
    "three": 3, "few": 3,
    "four": 4, "five": 5,
}


def _fire_count(text: str) -> int | None:
    """How many fires the user asked for ('one fire', 'two fires', '3 fires').

    Matches a quantity that precedes the noun, so it never collides with an
    explicit index like 'fire 3' (noun first). Returns None when no count is
    given (e.g. bare 'put out the fire')."""
    m = re.search(
        r"\b(\d+|a|an|one|two|three|four|five|single|both|couple|pair|few)"
        r"\s+(?:of\s+)?fires?\b",
        text,
    )
    if not m:
        return None
    tok = m.group(1)
    if tok.isdigit():
        c = int(tok)
        return c if c > 0 else None
    return _NUM_WORDS.get(tok)


def _nearest_fires_to_home(spec: SceneSpec, n: int) -> list[int]:
    """Indices of the ``n`` fires closest to home (the LLM's tie-break rule)."""
    hx, hy = spec.home
    order = sorted(
        range(len(spec.fires)),
        key=lambda i: (spec.fires[i][0] - hx) ** 2 + (spec.fires[i][1] - hy) ** 2,
    )
    return order[: max(0, n)]


def _parse_segment(segment: str, spec: SceneSpec) -> list[dict[str, object]]:
    """Map one prompt clause to raw task dicts (before validation)."""
    seg = segment.strip()
    if not seg:
        return []

    raw: list[dict[str, object]] = []

    if re.search(r"\b(return|go back|come back|home)\b", seg):
        raw.append({"type": "return", "target": None})
        return raw

    if re.search(r"\b(search|find|look|scan|explore|patrol|map)\b", seg):
        raw.append({"type": "search", "target": None})

    nums = _fire_numbers(seg)
    verb_visit = bool(re.search(r"\b(visit|go to|inspect|check)\b", seg))
    verb_ext = bool(re.search(
        r"\b(extinguish|douse|fight|put out|hit)\b|\bput\b.*\bout\b",
        seg,
    ))

    if verb_visit and not verb_ext:
        if nums:
            for idx in nums:
                raw.append({"type": "visit", "target": idx})
        else:
            raw.append({"type": "visit", "target": 0})
    elif verb_ext:
        if nums:
            # Explicit fire indices ("fire 3", "fires 1 and 2").
            for idx in nums:
                raw.append({"type": "extinguish", "target": idx})
        elif _all_fires_intent(seg):
            raw.append({"type": "extinguish", "target": None})
        else:
            # A count ("one fire", "two fires") -> that many nearest to home.
            count = _fire_count(seg)
            if count is not None and len(spec.fires):
                for idx in _nearest_fires_to_home(spec, count):
                    raw.append({"type": "extinguish", "target": idx})
            else:
                raw.append({"type": "extinguish", "target": None})

    return raw


def _rule_based(prompt: str, spec: SceneSpec) -> Mission:
    """Keyword parser over the fixed verb set. Always available (no network)."""
    p = prompt.lower().strip()
    segments = re.split(r"\s+(?:and\s+)?then\s+", p) if p else [""]
    raw: list[dict[str, object]] = []
    for segment in segments:
        raw.extend(_parse_segment(segment, spec))

    tasks = _validate_tasks(raw, spec)
    if not tasks:
        tasks = _validate_tasks([{"type": "extinguish", "target": None}], spec)
    return Mission(tasks)


_LLM_SYSTEM = (
    "You are the mission planner for an autonomous firefighting robot.\n"
    "{scene}\n\n"
    "Translate the user's natural-language order into an ordered list of tasks.\n"
    "Use ONLY these task types:\n"
    "  extinguish — approach a fire and spray until out. "
    "target = 0-based fire index, or null meaning ALL fires.\n"
    "  visit — drive next to a fire without spraying. target = 0-based fire index.\n"
    "  search — patrol/sweep the area. Use this for 'map the space', 'explore',\n"
    "    'patrol', or going to an unspecified/random spot. target = null.\n"
    "  return — go back home / to the start position. target = null.\n\n"
    "Rules:\n"
    "- 'extinguish all fires' (or no count given) -> ONE extinguish task, target null.\n"
    "- 'extinguish one fire' / 'a fire' -> ONE extinguish task targeting the fire\n"
    "  nearest to home.\n"
    "- 'extinguish N fires' -> N extinguish tasks with N distinct targets, the ones\n"
    "  nearest to home first.\n"
    "- Keep the order of clauses joined by 'then'/'and then'.\n"
    "- Output JSON only, no markdown, shaped exactly like:\n"
    '  {{"tasks": [{{"type": "extinguish", "target": 0}}, '
    '{{"type": "return", "target": null}}]}}'
)


def _llm_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _llm_mission(prompt: str, spec: SceneSpec) -> Mission | None:
    from google import genai
    from google.genai import types

    api_key = _llm_api_key()
    if not api_key:
        return None

    model = os.environ.get("EMBER_LLM_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_LLM_SYSTEM.format(scene=scene_summary(spec)),
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    content = resp.text
    if not content:
        return None

    data = json.loads(content)
    raw_tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(raw_tasks, list):
        return None

    tasks = _validate_tasks(raw_tasks, spec)
    if not tasks:
        return None
    return Mission(tasks)


def parse_mission(prompt: str, spec: SceneSpec, *, use_llm: bool = True) -> Mission:
    """Parse ``prompt`` into a validated ``Mission``.

    Tries the Gemini LLM (when ``use_llm``, the SDK is installed, and a
    ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is set), else the rule-based
    fallback. Never raises due to LLM failures.
    """
    if use_llm and _llm_api_key():
        try:
            from google import genai  # noqa: F401 — availability gate
        except ImportError:
            pass
        else:
            try:
                mission = _llm_mission(prompt, spec)
                if mission:
                    return mission
            except Exception:
                pass
    return _rule_based(prompt, spec)

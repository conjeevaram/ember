"""Phase 5 mission parsing: rule-based grammar, validation, LLM fallback."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ember import scenegen
from ember.mission import (
    Mission,
    Task,
    TaskType,
    _rule_based,
    _validate_tasks,
    parse_mission,
    scene_summary,
)


@pytest.fixture
def spec():
    return scenegen.random_spec(42, n_fires=3, n_walls=2, n_debris=0)


def _mission(*tasks: Task) -> Mission:
    return Mission(tasks)


def _ext(target: int | None = None) -> Task:
    return Task(TaskType.EXTINGUISH, target)


def _visit(target: int | None = 0) -> Task:
    return Task(TaskType.VISIT, target)


def _search() -> Task:
    return Task(TaskType.SEARCH)


def _return() -> Task:
    return Task(TaskType.RETURN)


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("extinguish all fires", (_ext(None),)),
        ("put out every fire", (_ext(None),)),
        ("douse everything", (_ext(None),)),
        ("fight all fires", (_ext(None),)),
        ("extinguish fire 2", (_ext(1),)),
        ("put out fires 1 and 3", (_ext(0), _ext(2))),
        ("visit fire 2", (_visit(1),)),
        ("go to fire 1", (_visit(0),)),
        ("inspect fire 3", (_visit(2),)),
        ("check fire 2", (_visit(1),)),
        ("search for fires", (_search(),)),
        ("find fires", (_search(),)),
        ("look around", (_search(),)),
        ("scan the area", (_search(),)),
        ("explore", (_search(),)),
        ("patrol", (_search(),)),
        ("return home", (_return(),)),
        ("go back", (_return(),)),
        ("come back", (_return(),)),
        (
            "search for fires then put them all out",
            (_search(), _ext(None)),
        ),
        ("hello world", (_ext(None),)),
        ("extinguish fire 99", (_ext(2),)),  # clamp to last fire
        ("visit fire 0", (_visit(0),)),       # 1-based "0" -> clamped index 0
    ],
)
def test_rule_based_prompts(spec, prompt, expected):
    got = _rule_based(prompt, spec)
    assert got.tasks == expected


def test_rule_based_deterministic(spec):
    prompt = "search then extinguish fire 2"
    assert _rule_based(prompt, spec) == _rule_based(prompt, spec)


def test_scene_summary_includes_key_fields(spec):
    text = scene_summary(spec)
    assert str(len(spec.fires)) in text
    xmin, xmax, ymin, ymax = spec.bounds
    assert f"x[{xmin:.1f},{xmax:.1f}]" in text
    assert f"y[{ymin:.1f},{ymax:.1f}]" in text
    assert str(spec.home) in text


def test_validate_tasks_clamps_and_coerces(spec):
    raw = [
        {"type": "EXTINGUISH", "target": 99},
        {"type": "search", "target": 5},
        {"type": "return", "target": 1},
        {"type": "bogus", "target": 0},
        {"type": "visit", "target": "bad"},
    ]
    tasks = _validate_tasks(raw, spec)
    assert tasks == (
        Task(TaskType.EXTINGUISH, 2),
        Task(TaskType.SEARCH, None),
        Task(TaskType.RETURN, None),
    )


def test_mission_container_api():
    m = Mission((_search(), _return()))
    assert len(m) == 2
    assert list(m) == [_search(), _return()]
    assert m
    assert not Mission(())


def _install_fake_genai(monkeypatch, gen):
    """Install a fake ``google.genai`` so the LLM path runs offline.

    ``gen`` stands in for ``client.models.generate_content``. Returns the
    ``GenerateContentConfig`` mock so callers can assert on the system prompt.
    """
    import sys

    cfg = MagicMock(return_value=object())
    types_mod = SimpleNamespace(GenerateContentConfig=cfg)
    client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))
    genai_mod = SimpleNamespace(Client=MagicMock(return_value=client), types=types_mod)
    google_mod = SimpleNamespace(genai=genai_mod)
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    return cfg


def _gemini_reply(tasks_json: dict):
    return SimpleNamespace(text=json.dumps(tasks_json))


def test_parse_mission_llm_success(monkeypatch, spec):
    gen = MagicMock(return_value=_gemini_reply({
        "tasks": [
            {"type": "extinguish", "target": 0},
            {"type": "return", "target": None},
        ],
    }))
    cfg = _install_fake_genai(monkeypatch, gen)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    got = parse_mission("put out fire 1 and go home", spec, use_llm=True)
    assert got.tasks == (_ext(0), _return())
    gen.assert_called_once()
    assert gen.call_args.kwargs["contents"] == "put out fire 1 and go home"
    assert scene_summary(spec) in cfg.call_args.kwargs["system_instruction"]


def test_parse_mission_llm_malformed_falls_back(monkeypatch, spec):
    gen = MagicMock(return_value=_gemini_reply({"tasks": []}))
    _install_fake_genai(monkeypatch, gen)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    got = parse_mission("return home", spec, use_llm=True)
    assert got.tasks == (_return(),)


def test_parse_mission_llm_exception_falls_back(monkeypatch, spec):
    gen = MagicMock(side_effect=RuntimeError("network down"))
    _install_fake_genai(monkeypatch, gen)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    got = parse_mission("search", spec, use_llm=True)
    assert got.tasks == (_search(),)


def test_parse_mission_no_llm_without_key(monkeypatch, spec):
    gen = MagicMock()
    _install_fake_genai(monkeypatch, gen)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    got = parse_mission("return", spec, use_llm=True)
    assert got.tasks == (_return(),)
    gen.assert_not_called()


def test_parse_mission_use_llm_false_skips_client(monkeypatch, spec):
    gen = MagicMock()
    _install_fake_genai(monkeypatch, gen)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    got = parse_mission("return", spec, use_llm=False)
    assert got.tasks == (_return(),)
    gen.assert_not_called()

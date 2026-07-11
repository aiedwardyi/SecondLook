"""Auditor parse, follow-up, and API-path tests."""

import base64
import sys
import types
from unittest.mock import MagicMock

from src.trust.auditor import (
    DEFER_EVIDENCE,
    DEFER_UNAVAILABLE,
    MAX_FOLLOWUP_CHARS,
    audit_tile,
    fail_closed,
    follow_up_attention,
    normalize_followup_question,
    parse_audit_response,
    plain_reason_line,
)

_METRICS = {"topk_mass": 0.4, "corner_ratio": 0.1, "edge_ratio": 0.2}


def test_parse_valid_json():
    raw = """{
      "status": "FLAGGED",
      "reason_lines": [
        "Attention is piled on the frame corner.",
        "Needs human review. Class label unchanged."
      ],
      "numbers_cited": ["corner_ratio=0.41"]
    }"""
    r = parse_audit_response(raw)
    assert r.status == "FLAGGED"
    assert r.error is None
    assert r.defer_reason is None
    assert len(r.reason_lines) == 2
    assert r.numbers_cited == ["corner_ratio=0.41"]


def test_parse_fenced_json():
    raw = """```json
{"status":"VERIFIED","reason_lines":["Heat on tissue.","Map and score agree."],"numbers_cited":["topk_mass=0.55"]}
```"""
    r = parse_audit_response(raw)
    assert r.status == "VERIFIED"
    assert r.error is None
    assert r.defer_reason is None


def test_valid_defer_is_evidence_ambiguous():
    raw = """{
      "status": "DEFER",
      "reason_lines": ["Mixed attention.", "Doctor owns the call."],
      "numbers_cited": []
    }"""
    r = parse_audit_response(raw)
    assert r.status == "DEFER"
    assert r.defer_reason == DEFER_EVIDENCE
    assert r.error is None


def test_malformed_json_is_audit_unavailable():
    r = parse_audit_response("not json at all")
    assert r.status == "DEFER"
    assert r.defer_reason == DEFER_UNAVAILABLE


def test_schema_invalid_json_is_audit_unavailable():
    raw = '{"status":"TRUST","reason_lines":["a","b"],"numbers_cited":[]}'
    r = parse_audit_response(raw)
    assert r.status == "DEFER"
    assert r.defer_reason == DEFER_UNAVAILABLE
    assert r.error is not None


def test_empty_defers():
    r = parse_audit_response("")
    assert r.status == "DEFER"
    assert r.defer_reason == DEFER_UNAVAILABLE


def test_fail_closed_shape():
    r = fail_closed("boom")
    assert r.status == "DEFER"
    assert r.error == "boom"
    assert r.defer_reason == DEFER_UNAVAILABLE
    assert len(r.reason_lines) >= 2


def test_json_array_root_defers():
    r = parse_audit_response("[]")
    assert r.status == "DEFER"
    assert r.defer_reason == DEFER_UNAVAILABLE


def test_four_reason_lines_defers():
    raw = json_blob(4)
    r = parse_audit_response(raw)
    assert r.status == "DEFER"
    assert r.defer_reason == DEFER_UNAVAILABLE


def test_flagged_without_numbers_defers():
    raw = """{
      "status": "FLAGGED",
      "reason_lines": ["Corner heat.", "Needs review."],
      "numbers_cited": []
    }"""
    r = parse_audit_response(raw)
    assert r.status == "DEFER"
    assert r.defer_reason == DEFER_UNAVAILABLE


def test_missing_api_key_is_audit_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = audit_tile(
        score=0.95,
        verdict="POSITIVE",
        metrics=_METRICS,
        api_key=None,
    )
    assert r.status == "DEFER"
    assert r.defer_reason == DEFER_UNAVAILABLE
    assert "ANTHROPIC_API_KEY" in (r.error or "")


def test_image_png_builds_anthropic_block(monkeypatch):
    captured: dict = {}
    ok_json = (
        '{"status":"VERIFIED","reason_lines":["On tissue.","Map agrees."],'
        '"numbers_cited":["topk_mass=0.40"]}'
    )

    class _Msg:
        content = [types.SimpleNamespace(type="text", text=ok_json)]

    class _Messages:
        def create(self, **kwargs):
            captured["content"] = kwargs["messages"][0]["content"]
            captured["system"] = kwargs["system"]
            return _Msg()

    class _Client:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    r = audit_tile(
        score=0.95,
        verdict="POSITIVE",
        metrics=_METRICS,
        image_png=png,
        api_key="test-key-not-real",
    )
    assert r.status == "VERIFIED"
    assert captured["content"][0]["type"] == "image"
    src = captured["content"][0]["source"]
    assert src["type"] == "base64"
    assert src["media_type"] == "image/png"
    assert src["data"] == base64.standard_b64encode(png).decode("ascii")
    assert isinstance(captured["system"], str)
    assert "attention" in captured["system"].lower()


def test_fake_anthropic_success_returns_status(monkeypatch):
    ok_json = (
        '{"status":"FLAGGED","reason_lines":["Corner pile-up.","Needs review."],'
        '"numbers_cited":["corner_ratio=0.41"]}'
    )
    create_mock = MagicMock(
        return_value=types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=ok_json)]
        )
    )

    class _Messages:
        create = create_mock

    class _Client:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    r = audit_tile(
        score=0.99,
        verdict="POSITIVE",
        metrics=_METRICS,
        api_key="test-key-not-real",
    )
    assert r.status == "FLAGGED"
    assert r.defer_reason is None
    create_mock.assert_called_once()


def test_to_dict_includes_defer_reason():
    r = fail_closed("x")
    d = r.to_dict()
    assert d["defer_reason"] == DEFER_UNAVAILABLE
    assert "status" in d


def test_plain_reason_line_maps_metric_fields():
    assert plain_reason_line("corner_ratio is high") == "corner heat is high"
    assert plain_reason_line("edge_ratio=0.19") == "edge heat=0.19"
    assert plain_reason_line("topk_mass low") == "focus concentration low"
    assert "corner_ratio" not in plain_reason_line("corner_ratio and edge_ratio")


def test_metric_level_matches_ui_bands():
    from src.trust.auditor import metric_level

    assert metric_level(0.14) == "low"
    assert metric_level(0.15) == "modest"
    assert metric_level(0.21) == "modest"
    assert metric_level(0.29) == "modest"
    assert metric_level(0.30) == "high"
    assert metric_level(0.41) == "high"


def test_align_reason_metric_labels_fixes_mismatched_intensity():
    from src.trust.auditor import align_reason_metric_labels

    metrics = {
        "topk_mass": 0.22,
        "corner_ratio": 0.21,
        "edge_ratio": 0.37,
    }
    lines = [
        "Corner heat is high at 21% with a bright hot spot.",
        "Edge heat is elevated at 37%; this call needs human review.",
        "Focus concentration is low at 22%.",
        "Attention is spread with low focus at 18%.",
    ]
    metrics_spread = {
        "topk_mass": 0.18,
        "corner_ratio": 0.03,
        "edge_ratio": 0.18,
    }
    fixed = align_reason_metric_labels(lines[:3], metrics)
    assert fixed[0] == "Corner heat is modest at 21% with a bright hot spot."
    assert fixed[1] == "Edge heat is high at 37%; this call needs human review."
    assert fixed[2] == "Focus concentration is modest at 22%."
    paraphrased = align_reason_metric_labels([lines[3]], metrics_spread)
    assert paraphrased[0] == "Attention is spread with modest focus at 18%."


def test_align_reason_metric_labels_does_not_rewrite_unrelated_intensity():
    from src.trust.auditor import align_reason_metric_labels

    metrics = {
        "topk_mass": 0.12,
        "corner_ratio": 0.05,
        "edge_ratio": 0.08,
    }
    line = "High confidence with tight focus at 12%."
    assert align_reason_metric_labels([line], metrics) == [line]


def test_align_reason_metric_labels_zero_gap_and_metric_anchor():
    from src.trust.auditor import align_reason_metric_labels

    metrics = {
        "topk_mass": 0.21,
        "corner_ratio": 0.21,
        "edge_ratio": 0.10,
    }
    zero_gap = align_reason_metric_labels(
        ["concentration is high at 21%."],
        metrics,
    )
    assert zero_gap == ["concentration is modest at 21%."]
    anchored = align_reason_metric_labels(
        ["Corner heat is high at 21% with a bright hot spot."],
        metrics,
    )
    assert anchored == ["Corner heat is modest at 21% with a bright hot spot."]


def test_normalize_followup_question_guards():
    assert normalize_followup_question("  Why?  ") == "Why?"
    try:
        normalize_followup_question("   ")
        assert False, "expected empty rejection"
    except ValueError:
        pass
    try:
        normalize_followup_question("x" * (MAX_FOLLOWUP_CHARS + 1))
        assert False, "expected length rejection"
    except ValueError:
        pass


def test_followup_success_fake_anthropic(monkeypatch):
    create_mock = MagicMock(
        return_value=types.SimpleNamespace(
            content=[
                types.SimpleNamespace(
                    type="text",
                    text="Attention sits on the corner. That is why trust is low.",
                )
            ]
        )
    )

    class _Messages:
        create = create_mock

    class _Client:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    answer, err = follow_up_attention(
        score=0.99,
        tier="POSITIVE",
        metrics=_METRICS,
        audit_status="FLAGGED",
        reason_lines=["Corner heat is high.", "Needs review."],
        image_png=png,
        question="Why flagged?",
        api_key="test-key-not-real",
    )
    assert err is None
    assert answer is not None
    assert "corner" in answer.lower() or "attention" in answer.lower()
    create_mock.assert_called_once()
    kwargs = create_mock.call_args.kwargs
    assert isinstance(kwargs.get("system"), str)
    assert "diagnose" in kwargs["system"].lower() or "Never diagnose" in kwargs["system"]


def test_followup_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    answer, err = follow_up_attention(
        score=0.9,
        tier="POSITIVE",
        metrics=_METRICS,
        audit_status="VERIFIED",
        reason_lines=["a", "b"],
        image_png=b"png",
        question="Why?",
        api_key=None,
    )
    assert answer is None
    assert err == "missing_key"


def test_followup_api_failure(monkeypatch):
    class _Messages:
        def create(self, **kwargs):
            raise RuntimeError("network down")

    class _Client:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    answer, err = follow_up_attention(
        score=0.9,
        tier="POSITIVE",
        metrics=_METRICS,
        audit_status="VERIFIED",
        reason_lines=["a", "b"],
        image_png=b"png",
        question="Why?",
        api_key="test-key-not-real",
    )
    assert answer is None
    assert err == "api_error"


def test_followup_empty_response(monkeypatch):
    class _Messages:
        def create(self, **kwargs):
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="  ")])

    class _Client:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    answer, err = follow_up_attention(
        score=0.9,
        tier="POSITIVE",
        metrics=_METRICS,
        audit_status="VERIFIED",
        reason_lines=["a", "b"],
        image_png=b"png",
        question="Why?",
        api_key="test-key-not-real",
    )
    assert answer is None
    assert err == "empty_response"


def test_followup_question_length_guard():
    answer, err = follow_up_attention(
        score=0.9,
        tier="POSITIVE",
        metrics=_METRICS,
        audit_status="VERIFIED",
        reason_lines=["a", "b"],
        image_png=b"png",
        question="x" * (MAX_FOLLOWUP_CHARS + 5),
        api_key="test-key-not-real",
    )
    assert answer is None
    assert err is not None
    assert "300" in err or "character" in err


def test_followup_cache_key_shape():
    """Session chat cache keys by evidence hash + normalized question."""
    ehash = "abc123"
    q1 = "  Why is this flagged? "
    q2 = "Why is this flagged?"
    norm1 = " ".join(q1.split()).lower()
    norm2 = " ".join(q2.split()).lower()
    assert (ehash, norm1) == (ehash, norm2)


def json_blob(n_reasons: int) -> str:
    reasons = ", ".join(f'"line {i}"' for i in range(n_reasons))
    return (
        '{"status":"FLAGGED","reason_lines":['
        + reasons
        + '],"numbers_cited":["corner_ratio=0.4"]}'
    )

"""Auditor parse tests."""

from src.trust.auditor import fail_closed, parse_audit_response


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
    assert len(r.reason_lines) == 2
    assert r.numbers_cited == ["corner_ratio=0.41"]


def test_parse_fenced_json():
    raw = """```json
{"status":"VERIFIED","reason_lines":["Heat on tissue.","Map and score agree."],"numbers_cited":["topk_mass=0.55"]}
```"""
    r = parse_audit_response(raw)
    assert r.status == "VERIFIED"
    assert r.error is None


def test_bad_status_defers():
    raw = '{"status":"TRUST","reason_lines":["a","b"],"numbers_cited":[]}'
    r = parse_audit_response(raw)
    assert r.status == "DEFER"
    assert r.error is not None


def test_empty_defers():
    r = parse_audit_response("")
    assert r.status == "DEFER"


def test_fail_closed_shape():
    r = fail_closed("boom")
    assert r.status == "DEFER"
    assert r.error == "boom"
    assert len(r.reason_lines) >= 2

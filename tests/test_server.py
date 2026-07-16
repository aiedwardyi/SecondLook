"""FastAPI server tests."""

import io
import threading
import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server import audit_coalesce, cache
from server import main as server_main
from server.cache import AnalysisRecord
from server.inference import CANONICAL_TILES, InferenceResult, build_evidence_hash
from server.schemas import AuditResponse, Metrics
from src.trust.auditor import AuditResult


class FakePipeline:
    def __init__(self) -> None:
        self.models = {"before": object(), "after": object()}
        self.device = "cpu"
        self.calls = []
        self.score_calls = []

    def score(self, image_bytes: bytes, model: str) -> float:
        self.score_calls.append((image_bytes, model))
        return 0.42

    def analyze(self, image_bytes: bytes, model: str) -> InferenceResult:
        self.calls.append((image_bytes, model))
        return InferenceResult(
            score=0.91,
            tier="POSITIVE",
            metrics=Metrics(topk_mass=0.12, corner_ratio=0.08, edge_ratio=0.21),
            overlay_png=b"overlay",
            original_png=b"original",
            evidence_hash="cafebabe",
        )


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(cache, "_RECORDS", OrderedDict())
    monkeypatch.setattr(audit_coalesce, "_INFLIGHT", {})


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def client(monkeypatch, pipeline):
    monkeypatch.setattr(
        server_main.InferencePipeline,
        "load",
        classmethod(lambda cls: pipeline),
    )
    with TestClient(server_main.app) as test_client:
        yield test_client


def _put_record(
    evidence_hash: str = "evidence",
    *,
    tier: str = "POSITIVE",
    score: float = 0.91,
    audit: AuditResponse | None = None,
) -> None:
    cache.put(
        evidence_hash,
        AnalysisRecord(
            overlay_png=b"overlay",
            score=score,
            tier=tier,
            metrics={"topk_mass": 0.12, "corner_ratio": 0.08, "edge_ratio": 0.21},
            model="after",
            audit=audit,
        ),
    )


def _image_bytes(image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(buffer, format=image_format)
    return buffer.getvalue()


def test_root_and_health(client):
    root = client.get("/")
    assert root.status_code == 200
    assert "<!DOCTYPE html>" in root.text

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "models": ["after", "before"],
        "device": "cpu",
    }


def test_analyze_canonical_response_and_cache(client, pipeline, monkeypatch):
    canonical = Mock(return_value=b"canonical")
    monkeypatch.setattr(server_main, "canonical_tile_bytes", canonical)

    response = client.post(
        "/api/analyze",
        data={"model": "after", "tile_id": "positive-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "score": 0.91,
        "tier": "POSITIVE",
        "metrics": {"topk_mass": 0.12, "corner_ratio": 0.08, "edge_ratio": 0.21},
        "overlay_png": "data:image/png;base64,b3ZlcmxheQ==",
        "original_png": "data:image/png;base64,b3JpZ2luYWw=",
        "evidence_hash": "cafebabe",
    }
    canonical.assert_called_once_with("positive-1")
    assert pipeline.calls == [(b"canonical", "after")]
    record = cache.get("cafebabe")
    assert record is not None
    assert record.overlay_png == b"overlay"
    assert record.model == "after"


@pytest.mark.parametrize(
    ("tile_id", "filename"),
    [
        ("positive-1", "positive_1.png"),
        ("positive-2", "positive_2.png"),
        ("positive-3", "positive_3.png"),
        ("borderline-1", "unclear_1.png"),
        ("borderline-2", "unclear_2.png"),
        ("borderline-3", "unclear_3.png"),
        ("negative-1", "negative_1.png"),
        ("negative-2", "negative_2.png"),
        ("negative-3", "negative_3.png"),
    ],
)
def test_canonical_tile_mapping(tile_id, filename):
    expected_dir = Path(__file__).resolve().parents[1] / "gallery" / "before"
    assert CANONICAL_TILES[tile_id] == expected_dir / filename
    assert CANONICAL_TILES[tile_id].is_file()


def test_analyze_rejects_unknown_tile(client, pipeline):
    response = client.post(
        "/api/analyze",
        data={"model": "after", "tile_id": "missing"},
    )

    assert response.status_code == 422
    assert "unknown tile_id" in response.json()["detail"]
    assert pipeline.calls == []


def test_tile_original_serves_canonical_bytes(client):
    response = client.get("/api/tile/positive-1")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert response.content == CANONICAL_TILES["positive-1"].read_bytes()


def test_tile_original_unknown(client):
    response = client.get("/api/tile/missing")
    assert response.status_code == 404


def test_score_canonical_tile(client, pipeline, monkeypatch):
    tile_bytes = b"canonical-tile-bytes"
    monkeypatch.setattr(server_main, "canonical_tile_bytes", lambda tile_id: tile_bytes)

    response = client.post(
        "/api/score",
        data={"model": "after", "tile_id": "positive-1"},
    )

    assert response.status_code == 200
    assert response.json() == {"score": 0.42}
    assert pipeline.score_calls == [(tile_bytes, "after")]
    assert pipeline.calls == []


def test_score_rejects_unknown_tile(client, pipeline):
    response = client.post(
        "/api/score",
        data={"model": "after", "tile_id": "missing"},
    )
    assert response.status_code == 422
    assert "unknown tile_id" in response.json()["detail"]
    assert pipeline.score_calls == []


def test_analyze_rejects_upload_type(client):
    response = client.post(
        "/api/analyze",
        data={"model": "after"},
        files={"file": ("tile.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "unsupported image type"


@pytest.mark.parametrize(
    ("image_format", "content_type", "filename"),
    [
        ("JPEG", "image/jpeg", "tile.jpg"),
        ("PNG", "image/png", "tile.png"),
        ("TIFF", "image/tiff", "tile.tiff"),
    ],
)
def test_analyze_passes_valid_upload_bytes_unchanged(
    client,
    pipeline,
    image_format,
    content_type,
    filename,
):
    image_bytes = _image_bytes(image_format)

    response = client.post(
        "/api/analyze",
        data={"model": "after"},
        files={"file": (filename, image_bytes, content_type)},
    )

    assert response.status_code == 200
    assert pipeline.calls == [(image_bytes, "after")]


def test_analyze_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(server_main, "MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr(server_main, "MAX_UPLOAD_MB", 0)

    response = client.post(
        "/api/analyze",
        data={"model": "after"},
        files={"file": ("tile.png", b"12345", "image/png")},
    )

    assert response.status_code == 413


def test_analyze_rejects_invalid_image(client):
    response = client.post(
        "/api/analyze",
        data={"model": "after"},
        files={"file": ("tile.png", b"not a png", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid image"


def test_analyze_rejects_corrupt_image(client, pipeline):
    corrupt_png = _image_bytes("PNG")[:-12]

    response = client.post(
        "/api/analyze",
        data={"model": "after"},
        files={"file": ("tile.png", corrupt_png, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid image"
    assert pipeline.calls == []


def test_audit_requires_cached_evidence(client):
    response = client.post("/api/audit", json={"evidence_hash": "missing"})

    assert response.status_code == 404
    assert response.json()["detail"] == "evidence not found"


def test_audit_requires_positive_call_tier(client, monkeypatch):
    _put_record(tier="NEGATIVE")
    audit = Mock()
    monkeypatch.setattr(server_main, "audit_tile", audit)

    response = client.post("/api/audit", json={"evidence_hash": "evidence"})

    assert response.status_code == 409
    assert response.json()["detail"] == "audit requires a POSITIVE call tier"
    audit.assert_not_called()


def test_audit_allows_call_tier_positive_when_stored_uncertain(client, monkeypatch):
    _put_record(tier="UNCERTAIN", score=0.65)
    audit = Mock(
        return_value=AuditResult(
            status="VERIFIED",
            reason_lines=["Attention stays on tissue."],
            numbers_cited=["corner_ratio=0.08"],
        )
    )
    monkeypatch.setattr(server_main, "audit_tile", audit)

    response = client.post(
        "/api/audit",
        json={"evidence_hash": "evidence", "call_tier": "POSITIVE"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "VERIFIED"
    audit.assert_called_once_with(
        score=0.65,
        verdict="POSITIVE",
        metrics={"topk_mass": 0.12, "corner_ratio": 0.08, "edge_ratio": 0.21},
        image_png=b"overlay",
        model_id="after",
    )


def test_audit_rejects_call_tier_positive_when_stored_negative(client, monkeypatch):
    _put_record(tier="NEGATIVE", score=0.05)
    audit = Mock()
    monkeypatch.setattr(server_main, "audit_tile", audit)

    response = client.post(
        "/api/audit",
        json={"evidence_hash": "evidence", "call_tier": "POSITIVE"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "audit requires a POSITIVE call tier"
    audit.assert_not_called()


def test_audit_fail_closed_omits_internal_error(client, monkeypatch):
    _put_record()
    monkeypatch.setattr(
        server_main,
        "audit_tile",
        Mock(
            return_value=AuditResult(
                status="DEFER",
                reason_lines=["corner_ratio unavailable"],
                numbers_cited=[],
                error="ANTHROPIC_API_KEY missing",
                defer_reason="AUDIT_UNAVAILABLE",
            )
        ),
    )

    response = client.post("/api/audit", json={"evidence_hash": "evidence"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "DEFER",
        "reason_lines": ["corner heat unavailable"],
        "numbers_cited": [],
        "defer_reason": "AUDIT_UNAVAILABLE",
    }
    assert "ANTHROPIC_API_KEY" not in response.text


def test_audit_reuses_cached_result(client, monkeypatch):
    _put_record()
    audit = Mock(
        return_value=AuditResult(
            status="VERIFIED",
            reason_lines=["Attention stays on tissue."],
            numbers_cited=["corner_ratio=0.08"],
        )
    )
    monkeypatch.setattr(server_main, "audit_tile", audit)

    first = client.post("/api/audit", json={"evidence_hash": "evidence"})
    second = client.post("/api/audit", json={"evidence_hash": "evidence"})

    assert first.status_code == 200
    assert second.json() == first.json()
    audit.assert_called_once_with(
        score=0.91,
        verdict="POSITIVE",
        metrics={"topk_mass": 0.12, "corner_ratio": 0.08, "edge_ratio": 0.21},
        image_png=b"overlay",
        model_id="after",
    )


def test_audit_coalesces_concurrent_same_hash(client, monkeypatch):
    _put_record()
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def slow_audit(**_kwargs):
        calls["n"] += 1
        started.set()
        assert release.wait(timeout=5)
        return AuditResult(
            status="VERIFIED",
            reason_lines=["Attention stays on tissue.", "Map agrees."],
            numbers_cited=["corner_ratio=0.08"],
        )

    monkeypatch.setattr(server_main, "audit_tile", slow_audit)
    results: list = []
    errors: list = []

    def hit():
        try:
            results.append(client.post("/api/audit", json={"evidence_hash": "evidence"}))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=hit)
    t2 = threading.Thread(target=hit)
    t1.start()
    assert started.wait(timeout=5)
    t2.start()
    time.sleep(0.15)
    release.set()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert errors == []
    assert len(results) == 2
    assert all(r.status_code == 200 for r in results)
    assert results[0].json()["status"] == "VERIFIED"
    assert results[1].json() == results[0].json()
    assert calls["n"] == 1


def test_audit_timeout_keeps_final_cached_result(client, monkeypatch):
    _put_record()

    async def race(*_a, **_k):
        cache.set_audit(
            "evidence",
            AuditResponse(
                status="VERIFIED",
                reason_lines=["Attention stays on tissue.", "Map agrees."],
                numbers_cited=["corner_ratio=0.08"],
            ),
            call_tier="POSITIVE",
        )
        raise TimeoutError()

    monkeypatch.setattr(server_main.audit_coalesce, "run_once_async", race)
    response = client.post("/api/audit", json={"evidence_hash": "evidence"})
    assert response.status_code == 200
    assert response.json()["status"] == "VERIFIED"


def test_audit_uses_accepted_snapshot_if_evicted(client, monkeypatch):
    _put_record()

    def audit_after_evict(**_kwargs):
        monkeypatch.setattr(cache, "_RECORDS", OrderedDict())
        return AuditResult(
            status="VERIFIED",
            reason_lines=["Attention stays on tissue.", "Map agrees."],
            numbers_cited=["corner_ratio=0.08"],
        )

    monkeypatch.setattr(server_main, "audit_tile", audit_after_evict)
    response = client.post("/api/audit", json={"evidence_hash": "evidence"})
    assert response.status_code == 200
    assert response.json()["status"] == "VERIFIED"


def test_audit_stores_defer_for_ask_but_does_not_reuse_on_audit(client, monkeypatch):
    _put_record()
    audit = Mock(
        return_value=AuditResult(
            status="DEFER",
            reason_lines=["Evidence unclear."],
            numbers_cited=[],
            defer_reason="EVIDENCE_AMBIGUOUS",
        )
    )
    monkeypatch.setattr(server_main, "audit_tile", audit)
    follow_up = Mock(return_value=("Score and map leave the call open.", None))
    monkeypatch.setattr(server_main, "follow_up_attention", follow_up)

    first = client.post("/api/audit", json={"evidence_hash": "evidence"})
    second = client.post("/api/audit", json={"evidence_hash": "evidence"})
    ask = client.post(
        "/api/ask",
        json={"evidence_hash": "evidence", "question": "Why is this unresolved?"},
    )

    assert first.status_code == 200
    assert first.json()["status"] == "DEFER"
    assert second.json()["status"] == "DEFER"
    assert audit.call_count == 2
    stored = cache.get("evidence").audit
    assert stored is not None
    assert stored.status == "DEFER"
    assert ask.status_code == 200
    assert ask.json() == {"answer": "Score and map leave the call open.", "error": None}
    follow_up.assert_called_once()


def test_cache_evicts_oldest_past_max(monkeypatch):
    monkeypatch.setattr(cache, "MAX_RECORDS", 2)
    monkeypatch.setattr(cache, "_RECORDS", OrderedDict())
    for i in range(3):
        cache.put(
            f"h{i}",
            AnalysisRecord(
                overlay_png=b"o",
                score=0.1 * i,
                tier="POSITIVE",
                metrics={"topk_mass": 0.1, "corner_ratio": 0.1, "edge_ratio": 0.1},
                model="after",
            ),
        )
    assert cache.get("h0") is None
    assert cache.get("h1") is not None
    assert cache.get("h2") is not None


def test_evidence_hash_includes_image_and_is_full_digest():
    metrics = Metrics(topk_mass=0.12, corner_ratio=0.08, edge_ratio=0.21)
    a = build_evidence_hash("after", 0.91, metrics, b"tile-a")
    b = build_evidence_hash("after", 0.91, metrics, b"tile-b")
    same = build_evidence_hash("after", 0.91, metrics, b"tile-a")
    assert len(a) == 64
    assert a != b
    assert a == same


def test_ask_requires_cached_audit(client, monkeypatch):
    _put_record()
    follow_up = Mock()
    monkeypatch.setattr(server_main, "follow_up_attention", follow_up)

    response = client.post(
        "/api/ask",
        json={"evidence_hash": "evidence", "question": "Why?"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "attention audit not found"
    follow_up.assert_not_called()


def test_ask_accepts_300_chars_and_rejects_more(client, monkeypatch):
    _put_record(
        audit=AuditResponse(
            status="VERIFIED",
            reason_lines=["Attention stays on tissue."],
            numbers_cited=["corner_ratio=0.08"],
        )
    )
    follow_up = Mock(return_value=("The attention stays on tissue.", None))
    monkeypatch.setattr(server_main, "follow_up_attention", follow_up)

    accepted = client.post(
        "/api/ask",
        json={"evidence_hash": "evidence", "question": "x" * 300},
    )
    rejected = client.post(
        "/api/ask",
        json={"evidence_hash": "evidence", "question": "x" * 301},
    )

    assert accepted.status_code == 200
    assert accepted.json() == {"answer": "The attention stays on tissue.", "error": None}
    assert rejected.status_code == 422
    follow_up.assert_called_once_with(
        score=0.91,
        tier="POSITIVE",
        metrics={"topk_mass": 0.12, "corner_ratio": 0.08, "edge_ratio": 0.21},
        audit_status="VERIFIED",
        reason_lines=["Attention stays on tissue."],
        image_png=b"overlay",
        question="x" * 300,
    )


def test_ask_uses_audit_call_tier_when_stored_tier_is_uncertain(client, monkeypatch):
    _put_record(tier="UNCERTAIN", score=0.65)
    monkeypatch.setattr(
        server_main,
        "audit_tile",
        Mock(
            return_value=AuditResult(
                status="VERIFIED",
                reason_lines=["Attention stays on tissue.", "Map agrees enough."],
                numbers_cited=["corner_ratio=0.08"],
            )
        ),
    )
    follow_up = Mock(return_value=("Lifted band is positive for this check.", None))
    monkeypatch.setattr(server_main, "follow_up_attention", follow_up)

    audit = client.post(
        "/api/audit",
        json={"evidence_hash": "evidence", "call_tier": "POSITIVE"},
    )
    ask = client.post(
        "/api/ask",
        json={"evidence_hash": "evidence", "question": "Why verified?"},
    )

    assert audit.status_code == 200
    assert ask.status_code == 200
    follow_up.assert_called_once_with(
        score=0.65,
        tier="POSITIVE",
        metrics={"topk_mass": 0.12, "corner_ratio": 0.08, "edge_ratio": 0.21},
        audit_status="VERIFIED",
        reason_lines=["Attention stays on tissue.", "Map agrees enough."],
        image_png=b"overlay",
        question="Why verified?",
    )

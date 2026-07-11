"""Thread-safe in-memory analysis cache."""

from dataclasses import dataclass, replace
from threading import RLock

from server.schemas import AuditResponse, ModelId, Tier


@dataclass(frozen=True)
class AnalysisRecord:
    overlay_png: bytes
    score: float
    tier: Tier
    metrics: dict[str, float]
    model: ModelId
    audit: AuditResponse | None = None


_LOCK = RLock()
_RECORDS: dict[str, AnalysisRecord] = {}


def _copy_audit(audit: AuditResponse | None) -> AuditResponse | None:
    return audit.model_copy(deep=True) if audit is not None else None


def _copy_record(record: AnalysisRecord) -> AnalysisRecord:
    return AnalysisRecord(
        overlay_png=record.overlay_png,
        score=record.score,
        tier=record.tier,
        metrics=dict(record.metrics),
        model=record.model,
        audit=_copy_audit(record.audit),
    )


def get(evidence_hash: str) -> AnalysisRecord | None:
    with _LOCK:
        record = _RECORDS.get(evidence_hash)
        return _copy_record(record) if record is not None else None


def put(evidence_hash: str, record: AnalysisRecord) -> None:
    with _LOCK:
        existing = _RECORDS.get(evidence_hash)
        if existing is not None and existing.audit is not None and record.audit is None:
            record = replace(record, audit=_copy_audit(existing.audit))
        _RECORDS[evidence_hash] = _copy_record(record)


def set_audit(evidence_hash: str, audit: AuditResponse) -> bool:
    with _LOCK:
        record = _RECORDS.get(evidence_hash)
        if record is None:
            return False
        _RECORDS[evidence_hash] = replace(record, audit=_copy_audit(audit))
        return True

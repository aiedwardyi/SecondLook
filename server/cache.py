"""Thread-safe in-memory analysis cache."""

from collections import OrderedDict
from dataclasses import dataclass, replace
from threading import RLock

from server.schemas import AuditResponse, ModelId, Tier

# Bounds PNG-heavy records.
MAX_RECORDS = 48


@dataclass(frozen=True)
class AnalysisRecord:
    overlay_png: bytes
    score: float
    tier: Tier
    metrics: dict[str, float]
    model: ModelId
    audit: AuditResponse | None = None
    # Tier used when the audit was run (may differ from analyze tier via call_tier).
    audit_call_tier: Tier | None = None


_LOCK = RLock()
_RECORDS: OrderedDict[str, AnalysisRecord] = OrderedDict()


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
        audit_call_tier=record.audit_call_tier,
    )


def _is_final_audit(audit: AuditResponse | None) -> bool:
    return audit is not None and audit.status != "DEFER"


def get(evidence_hash: str) -> AnalysisRecord | None:
    with _LOCK:
        record = _RECORDS.get(evidence_hash)
        if record is None:
            return None
        _RECORDS.move_to_end(evidence_hash)
        return _copy_record(record)


def put(evidence_hash: str, record: AnalysisRecord) -> None:
    with _LOCK:
        existing = _RECORDS.get(evidence_hash)
        # Keep a final audit across re-analyze; DEFER is not sticky.
        if (
            existing is not None
            and _is_final_audit(existing.audit)
            and record.audit is None
        ):
            record = replace(
                record,
                audit=_copy_audit(existing.audit),
                audit_call_tier=existing.audit_call_tier,
            )
        if evidence_hash in _RECORDS:
            del _RECORDS[evidence_hash]
        _RECORDS[evidence_hash] = _copy_record(record)
        while len(_RECORDS) > MAX_RECORDS:
            _RECORDS.popitem(last=False)


def set_audit(
    evidence_hash: str,
    audit: AuditResponse,
    *,
    call_tier: Tier | None = None,
) -> bool:
    """Store audit for ask lookup. call_tier is the band used when auditing."""
    with _LOCK:
        record = _RECORDS.get(evidence_hash)
        if record is None:
            return False
        _RECORDS[evidence_hash] = replace(
            record,
            audit=_copy_audit(audit),
            audit_call_tier=call_tier if call_tier is not None else record.audit_call_tier,
        )
        _RECORDS.move_to_end(evidence_hash)
        return True


def set_audit_soft(
    evidence_hash: str,
    audit: AuditResponse,
    *,
    call_tier: Tier | None = None,
) -> AuditResponse | None:
    """Store audit unless a final audit is already present; return that final if so."""
    with _LOCK:
        record = _RECORDS.get(evidence_hash)
        if record is None:
            return None
        if _is_final_audit(record.audit):
            return _copy_audit(record.audit)
        _RECORDS[evidence_hash] = replace(
            record,
            audit=_copy_audit(audit),
            audit_call_tier=call_tier if call_tier is not None else record.audit_call_tier,
        )
        _RECORDS.move_to_end(evidence_hash)
        return None

"""Claude attention auditor for detector trust checks."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, replace
from typing import Any

DEFAULT_MODEL = "claude-opus-4-8"
VALID_STATUS = frozenset({"VERIFIED", "FLAGGED", "DEFER"})
DEFER_EVIDENCE = "EVIDENCE_AMBIGUOUS"
DEFER_UNAVAILABLE = "AUDIT_UNAVAILABLE"
MAX_FOLLOWUP_CHARS = 300

# Intensity bands - keep in sync with server/static/index.html METRIC_LEVEL.
METRIC_LEVEL_MODEST = 0.15
METRIC_LEVEL_HIGH = 0.30
_INTENSITY_WORDS = ("low", "modest", "high", "elevated", "moderate", "mild", "strong")

_FIELD_TO_PLAIN = (
    ("corner_ratio", "corner heat"),
    ("edge_ratio", "edge heat"),
    ("topk_mass", "focus concentration"),
)

SYSTEM_PROMPT = """You audit a medical AI detector's attention - not the patient as a pathologist.

Job: decide whether the detector's call can be trusted based on WHERE it looked.
Primary judgment uses the score, the attention metrics, and the heatmap image.
Outputs only one of: VERIFIED, FLAGGED, DEFER.

Status rules:
- Read metrics first (focus concentration, corner heat, edge heat). The heatmap confirms.
- FLAGGED: high confidence but attention piled on corner/edge/frame, or map fights the score.
- VERIFIED: attention on tissue/field structures; map and score agree enough to stand.
- DEFER: borderline score, messy/mixed attention, or numbers vs image conflict. Doctor owns the call.
- Never flip the class label. You only change trust (needs review), not + vs -.
- On FLAGGED say the call needs human review. Do not talk about "auto-accept" or autonomy.
- Never claim the detector is correct or that the diagnosis is settled.
- Never diagnose the slide or the patient.

Do not invent clinical memory (hallucinated experience):
- Never say you "have seen" this case, these patients, or similar private slides.
- Never claim you trained on private data or this project's tiles.
- Never imply a personal case log.
- You cannot verify a private case list - do not fake one.

General pattern knowledge is OK only with hedges (optional, secondary):
- OK: "This pattern is often confused with follicle-like structures."
- OK: "Attention is on the corner, not the tissue field."
- Not OK: "I've seen many BCCs like this."
- Never diagnose the patient. Never lead with nest/palisading histology as proof.

If a claim is not grounded in the provided score, metrics, and image, do not make it.
Prefer DEFER or a short uncertainty line over invention.

Language for reason_lines:
- Plain clinical-review wording only (attention, corner heat, edge heat, tissue, review).
- Never print internal field names (no topk_mass, corner_ratio, edge_ratio).
- Do not lead with ML jargon. Describe only model attention and trust.
- Use hyphens only, never em dashes or en dashes.
- Cite at least one plain metric number in reason_lines when status is VERIFIED or FLAGGED.
- When you name intensity for a metric, use exactly the low/modest/high label given with that metric.
  Bands: low under 15%, modest 15% up to under 30%, high 30% and above.
- Example style: "Corner heat is high at 41%." not "corner_ratio=0.41".

Return ONLY a JSON object with keys:
  status: "VERIFIED" | "FLAGGED" | "DEFER"
  reason_lines: array of 2-3 short plain strings (no internal field names)
  numbers_cited: array of strings like "corner_ratio=0.41" (machine keys OK here only)
"""

FOLLOWUP_SYSTEM = """You answer a short follow-up about one attention-trust check.

Rules:
- Answer only about the model's attention and this trust check.
- Never diagnose the slide or the patient.
- Never change the detector class.
- Never change VERIFIED / FLAGGED / DEFER.
- Use 2-3 short plain sentences.
- Never print internal field names (no topk_mass, corner_ratio, edge_ratio).
- Use hyphens only, never em dashes or en dashes.
- When you name intensity for a metric, use the low/modest/high label supplied with that metric.
- If the evidence cannot answer the question, say so plainly.
- No API keys, no system details, no raw errors.
"""


@dataclass(frozen=True)
class AuditResult:
    status: str
    reason_lines: list[str]
    numbers_cited: list[str]
    raw: str | None = None
    error: str | None = None
    defer_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_lines": list(self.reason_lines),
            "numbers_cited": list(self.numbers_cited),
            "error": self.error,
            "defer_reason": self.defer_reason,
        }


def fail_closed(reason: str, *, raw: str | None = None) -> AuditResult:
    return AuditResult(
        status="DEFER",
        reason_lines=[
            "Could not finish the trust check.",
            "Please review this tile yourself.",
        ],
        numbers_cited=[],
        raw=raw,
        error=reason,
        defer_reason=DEFER_UNAVAILABLE,
    )


def plain_reason_line(text: str) -> str:
    """Map internal metric field names to plain review wording for display."""
    out = str(text)
    for field, plain in _FIELD_TO_PLAIN:
        out = re.sub(re.escape(field), plain, out, flags=re.IGNORECASE)
    return out.replace("\u2014", "-").replace("\u2013", "-")


def metric_level(frac: float) -> str:
    """low / modest / high - same bands as the trust UI metrics cards."""
    x = float(frac)
    if not (x >= 0) or x < METRIC_LEVEL_MODEST:
        return "low"
    if x < METRIC_LEVEL_HIGH:
        return "modest"
    return "high"


def format_metrics_block(metrics: dict[str, Any]) -> str:
    keys = ("topk_mass", "corner_ratio", "edge_ratio")
    plain_by_key = dict(_FIELD_TO_PLAIN)
    lines = []
    for k in keys:
        if k not in metrics:
            raise KeyError(f"missing metric {k!r}")
        value = float(metrics[k])
        plain = plain_by_key[k]
        level = metric_level(value)
        pct = int(round(value * 100))
        lines.append(f"{plain}: {pct}% ({level})  [{k}={value:.4f}]")
    return "\n".join(lines)


def align_reason_metric_labels(reasons: list[str], metrics: dict[str, Any]) -> list[str]:
    """Force intensity words near a metric % to match UI bands.

    Catches free paraphrases, e.g. "low focus at 18%" not only "focus concentration is low at 18%".
    """
    pct_to_level: dict[int, str] = {}
    for field, _plain in _FIELD_TO_PLAIN:
        if field not in metrics:
            continue
        value = float(metrics[field])
        pct_to_level[int(round(value * 100))] = metric_level(value)

    word_alt = "|".join(_INTENSITY_WORDS)
    aligned: list[str] = []
    for line in reasons:
        fixed = str(line)
        for pct, level in pct_to_level.items():
            # "low focus at 18%" / "is high at 21%" / "elevated at 37%"
            fixed = re.sub(
                rf"(?i)\b({word_alt})\b((?:\s+\w+){{0,4}})\s+at\s+(?:about\s+)?{pct}\s*%",
                rf"{level}\2 at {pct}%",
                fixed,
            )
            # "at 18% low focus" / "at about 21% high"
            fixed = re.sub(
                rf"(?i)(at\s+(?:about\s+)?{pct}\s*%\s+)({word_alt})\b",
                rf"\1{level}",
                fixed,
            )
        aligned.append(fixed)
    return aligned


def build_user_prompt(
    *,
    score: float,
    verdict: str,
    metrics: dict[str, Any],
    model_id: str = "unknown",
) -> str:
    block = format_metrics_block(metrics)
    return (
        f"Model: {model_id}\n"
        f"Detector score: {float(score):.4f}\n"
        f"Detector tier: {verdict}\n"
        f"Attention metrics (use the given low/modest/high labels when you describe them):\n{block}\n"
        "The attached image is the Grad-CAM overlay (where the model looked).\n"
        "Audit trust only. Return JSON only.\n"
        "In reason_lines use plain wording only - never print topk_mass, corner_ratio, or edge_ratio."
    )


def parse_audit_response(text: str) -> AuditResult:
    if not text or not str(text).strip():
        return fail_closed("empty response", raw=text)
    raw = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    blob = fenced.group(1) if fenced else raw
    try:
        start = blob.index("{")
        end = blob.rindex("}") + 1
        data = json.loads(blob[start:end])
    except (ValueError, json.JSONDecodeError) as exc:
        return fail_closed(f"json parse: {exc}", raw=raw)
    if not isinstance(data, dict):
        return fail_closed("json root must be an object", raw=raw)
    status = str(data.get("status", "")).upper().strip()
    if status not in VALID_STATUS:
        return fail_closed(f"bad status {status!r}", raw=raw)
    reasons = data.get("reason_lines") or []
    if not isinstance(reasons, list):
        return fail_closed("reason_lines not a list", raw=raw)
    reasons = [str(r).strip() for r in reasons if str(r).strip()]
    if not (2 <= len(reasons) <= 3):
        return fail_closed("reason_lines must be 2-3 items", raw=raw)
    cited = data.get("numbers_cited") or []
    if not isinstance(cited, list):
        return fail_closed("numbers_cited not a list", raw=raw)
    cited = [str(c).strip() for c in cited if str(c).strip()]
    if status in {"VERIFIED", "FLAGGED"} and not cited:
        return fail_closed("numbers_cited required for VERIFIED/FLAGGED", raw=raw)
    defer_reason = DEFER_EVIDENCE if status == "DEFER" else None
    return AuditResult(
        status=status,
        reason_lines=reasons,
        numbers_cited=cited,
        raw=raw,
        error=None,
        defer_reason=defer_reason,
    )


def _image_block(media_type: str, b64: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64,
        },
    }


def audit_tile(
    *,
    score: float,
    verdict: str,
    metrics: dict[str, Any],
    image_path: str | None = None,
    image_png: bytes | None = None,
    model_id: str = "unknown",
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> AuditResult:
    try:
        user_text = build_user_prompt(
            score=score, verdict=verdict, metrics=metrics, model_id=model_id
        )
    except (KeyError, TypeError, ValueError) as exc:
        return fail_closed(f"metrics: {exc}")

    key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return fail_closed("ANTHROPIC_API_KEY missing")

    try:
        import anthropic
    except ImportError as exc:
        return fail_closed(f"anthropic import: {exc}")

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if image_png is not None:
        if not image_png:
            return fail_closed("image_png is empty")
        b64 = base64.standard_b64encode(image_png).decode("ascii")
        content.insert(0, _image_block("image/png", b64))
    elif image_path:
        try:
            media_type, b64 = _load_image_b64(image_path)
            content.insert(0, _image_block(media_type, b64))
        except OSError as exc:
            return fail_closed(f"image: {exc}")

    try:
        client = anthropic.Anthropic(api_key=key, timeout=60.0)
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )
    except Exception as exc:
        return fail_closed(f"api: {exc}")

    result = parse_audit_response(text)
    if result.error:
        return result
    return replace(
        result,
        reason_lines=align_reason_metric_labels(result.reason_lines, metrics),
    )


def normalize_followup_question(question: str) -> str:
    """Trim and validate a bounded follow-up question."""
    text = str(question or "").strip()
    if not text:
        raise ValueError("empty question")
    if len(text) > MAX_FOLLOWUP_CHARS:
        raise ValueError(f"question over {MAX_FOLLOWUP_CHARS} characters")
    return text


def build_followup_prompt(
    *,
    score: float,
    tier: str,
    metrics: dict[str, Any],
    audit_status: str,
    reason_lines: list[str],
    question: str,
) -> str:
    block = format_metrics_block(metrics)
    reasons = "\n".join(f"- {plain_reason_line(r)}" for r in reason_lines[:3])
    return (
        f"Detector score: {float(score):.4f}\n"
        f"Detector tier: {tier}\n"
        f"Attention metrics:\n{block}\n"
        f"Audit status: {audit_status}\n"
        f"Audit reasons:\n{reasons}\n"
        "The attached image is the Grad-CAM overlay for this case.\n"
        f"Reviewer question: {question}\n"
        "Answer in 2-3 short plain sentences about attention and trust only."
    )


def follow_up_attention(
    *,
    score: float,
    tier: str,
    metrics: dict[str, Any],
    audit_status: str,
    reason_lines: list[str],
    image_png: bytes,
    question: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[str | None, str | None]:
    """Bounded Claude follow-up. Returns (answer, error_code). Never raises."""
    try:
        q = normalize_followup_question(question)
    except ValueError as exc:
        return None, str(exc)

    try:
        user_text = build_followup_prompt(
            score=score,
            tier=tier,
            metrics=metrics,
            audit_status=audit_status,
            reason_lines=list(reason_lines or []),
            question=q,
        )
    except (KeyError, TypeError, ValueError):
        return None, "bad_context"

    key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None, "missing_key"

    try:
        import anthropic
    except ImportError:
        return None, "import_error"

    if not image_png:
        return None, "empty_image"

    b64 = base64.standard_b64encode(image_png).decode("ascii")
    content: list[dict[str, Any]] = [
        _image_block("image/png", b64),
        {"type": "text", "text": user_text},
    ]

    try:
        client = anthropic.Anthropic(api_key=key, timeout=60.0)
        msg = client.messages.create(
            model=model,
            max_tokens=220,
            system=FOLLOWUP_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        ).strip()
    except Exception:
        return None, "api_error"

    if not text:
        return None, "empty_response"
    cleaned = plain_reason_line(text)
    return align_reason_metric_labels([cleaned], metrics)[0], None


def _load_image_b64(path: str) -> tuple[str, str]:
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        raise OSError(f"not a file: {path}")
    suffix = p.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        media = "image/jpeg"
    elif suffix == ".png":
        media = "image/png"
    else:
        raise OSError(f"unsupported image type: {suffix}")
    return media, base64.standard_b64encode(p.read_bytes()).decode("ascii")

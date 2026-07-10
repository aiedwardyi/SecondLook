"""Claude attention auditor for detector trust checks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "claude-opus-4-8"
VALID_STATUS = frozenset({"VERIFIED", "FLAGGED", "DEFER"})

SYSTEM_PROMPT = """You audit a medical AI detector's attention - not the patient as a pathologist.

Job: decide whether the detector's call can be trusted based on WHERE it looked.
Primary judgment uses the score, the attention metrics, and the heatmap image.
Outputs only one of: VERIFIED, FLAGGED, DEFER.

Status rules:
- Read metrics first (topk_mass, corner_ratio, edge_ratio). The heatmap confirms.
- FLAGGED: high confidence but attention piled on corner/edge/frame, or map fights the score.
- VERIFIED: attention on tissue/field structures; map and score agree enough to stand.
- DEFER: borderline score, messy/mixed attention, or numbers vs image conflict. Doctor owns the call.
- Never flip the class label. You only change trust (needs review), not + vs -.
- On FLAGGED say the call needs human review. Do not talk about "auto-accept" or autonomy.

Do not invent clinical memory (hallucinated experience):
- Never say you "have seen" this case, these patients, or similar private slides.
- Never claim you trained on Heidelberg data or this project's tiles.
- Never imply a personal case log ("of the BCCs I've reviewed...").
- You cannot verify a private case list - do not fake one.

General pattern knowledge is OK only with hedges (optional, secondary):
- OK: "This pattern is often confused with follicle-like structures."
- OK: "Attention is on the corner, not the tissue field."
- Not OK: "I've seen many BCCs like this."
- Never diagnose the patient. Never lead with nest/palisading histology as proof.

If a claim is not grounded in the provided score, metrics, and image, do not make it.
Prefer DEFER or a short uncertainty line over invention.

Language:
- Plain doctor words (attention, corner, tissue, review).
- Do not lead with ML jargon (topk, CAM mass) - you may still cite metric values.
- Cite at least one metric number in reason_lines when status is VERIFIED or FLAGGED.

Return ONLY a JSON object with keys:
  status: "VERIFIED" | "FLAGGED" | "DEFER"
  reason_lines: array of 2-3 short plain strings
  numbers_cited: array of strings like "corner_ratio=0.41"
"""


@dataclass(frozen=True)
class AuditResult:
    status: str
    reason_lines: list[str]
    numbers_cited: list[str]
    raw: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_lines": list(self.reason_lines),
            "numbers_cited": list(self.numbers_cited),
            "error": self.error,
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
    )


def format_metrics_block(metrics: dict[str, Any]) -> str:
    keys = ("topk_mass", "corner_ratio", "edge_ratio")
    lines = []
    for k in keys:
        if k not in metrics:
            raise KeyError(f"missing metric {k!r}")
        lines.append(f"{k}={float(metrics[k]):.4f}")
    return "\n".join(lines)


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
        f"Attention metrics:\n{block}\n"
        "The attached image is the Grad-CAM overlay (where the model looked).\n"
        "Audit trust only. Return JSON only."
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
    return AuditResult(
        status=status,
        reason_lines=reasons,
        numbers_cited=cited,
        raw=raw,
        error=None,
    )


def audit_tile(
    *,
    score: float,
    verdict: str,
    metrics: dict[str, Any],
    image_path: str | None = None,
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
    if image_path:
        try:
            media_type, b64 = _load_image_b64(image_path)
            content.insert(
                0,
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                },
            )
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

    return parse_audit_response(text)


def _load_image_b64(path: str) -> tuple[str, str]:
    import base64
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

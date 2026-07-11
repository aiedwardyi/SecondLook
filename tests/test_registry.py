"""Model registry resolve tests."""

import pytest

from src.models.registry import resolve


def test_resolve_before():
    spec = resolve("before")
    assert spec.label == "before"
    assert spec.padding_mode == "zeros"
    assert "bcc_before" in spec.path.replace("\\", "/")


def test_resolve_after():
    spec = resolve("after")
    assert spec.label == "after"
    assert spec.padding_mode == "reflect"


def test_resolve_unknown_model_id():
    with pytest.raises(KeyError, match="unknown model id"):
        resolve("not-a-model")

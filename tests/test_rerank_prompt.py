from __future__ import annotations

import numpy as np
import pytest

from meow_embed import server
from meow_embed.server import rerank_scores, validate_rerank_prompt_args


class _StubCrossEncoder:
    def __init__(self) -> None:
        self.predict_kwargs: dict[str, object] | None = None

    def predict(self, pairs: list[list[str]], **kwargs: object) -> np.ndarray:
        self.predict_kwargs = kwargs
        return np.array([0.1, 0.9], dtype=np.float32)


def test_validate_rerank_prompt_args_both_prompt_fields() -> None:
    with pytest.raises(ValueError, match="not both"):
        validate_rerank_prompt_args(
            is_cross_encoder=True,
            prompt="custom",
            prompt_name="ecommerce",
        )


def test_validate_rerank_prompt_args_flag_reranker() -> None:
    with pytest.raises(ValueError, match="CrossEncoder"):
        validate_rerank_prompt_args(
            is_cross_encoder=False,
            prompt="custom",
            prompt_name=None,
        )


def test_rerank_scores_cross_encoder_passes_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "CrossEncoder", _StubCrossEncoder)
    model = _StubCrossEncoder()
    pairs = [["query", "doc a"], ["query", "doc b"]]
    instruction = "Rate relevance for this marketplace search."

    scores = rerank_scores(model, pairs, prompt=instruction)

    np.testing.assert_array_equal(scores, np.array([0.1, 0.9], dtype=np.float32))
    assert model.predict_kwargs == {
        "convert_to_numpy": True,
        "prompt": instruction,
    }


def test_rerank_scores_cross_encoder_passes_prompt_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "CrossEncoder", _StubCrossEncoder)
    model = _StubCrossEncoder()

    rerank_scores(model, [["q", "d"]], prompt_name="ecommerce")

    assert model.predict_kwargs == {
        "convert_to_numpy": True,
        "prompt_name": "ecommerce",
    }

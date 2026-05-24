"""Rerank request typing for MeowEmbedClient.rerank / arerank."""

from __future__ import annotations

from typing import assert_type

from meow_embed.types import RerankActivationFn, RerankRequestDict


def test_rerank_request_dict_activation_fn() -> None:
    payload: RerankRequestDict = {
        "reranker_model_id": "DiTy/cross-encoder-russian-msmarco",
        "query": "запрос",
        "docs": ["док 1", "док 2"],
        "activation_fn": "identity",
    }
    assert_type(payload["activation_fn"], RerankActivationFn | None)


def test_rerank_request_dict_prompt() -> None:
    payload: RerankRequestDict = {
        "reranker_model_id": "Qwen/Qwen3-Reranker-0.6B",
        "query": "ноутбук",
        "docs": ["ноутбук Lenovo", "зарядка для ноутбука"],
        "prompt": "Определи релевантность названия товара запросу.",
    }
    assert_type(payload["prompt"], str | None)

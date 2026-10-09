"""Simple embedding return types, optional fields, validation, and offline caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, assert_type, cast

import httpx
import numpy as np
import pytest

from meow_embed import EmbedCache, MeowEmbedClient
from meow_embed.parsing import assemble_parsed_response
from meow_embed.types import (
    BGEM3Embeddings,
    DenseEmbeddings,
    EmbedInput,
    EmbedOneRequestPayload,
    EmbedRequestPayload,
    ParsedEmbedOne,
    ParsedEmbedResponse,
    SparseEmbedding,
    SparseEmbeddings,
)

MODEL_COMBINATIONS = [
    pytest.param(True, False, False, id="dense"),
    pytest.param(False, True, False, id="sparse"),
    pytest.param(False, False, True, id="bge"),
    pytest.param(True, True, False, id="dense-sparse"),
    pytest.param(True, False, True, id="dense-bge"),
    pytest.param(False, True, True, id="sparse-bge"),
    pytest.param(True, True, True, id="all"),
]


def _payload(
    texts: Sequence[EmbedInput], dense: bool, sparse: bool, bge: bool
) -> EmbedRequestPayload:
    return {
        "texts": texts,
        "dense_model_id": "d" if dense else None,
        "sparse_model_id": "s" if sparse else None,
        "bge_model_id": "b" if bge else None,
    }


def _one_payload(dense: bool, sparse: bool, bge: bool) -> EmbedOneRequestPayload:
    return {
        "text": "a",
        "dense_model_id": "d" if dense else None,
        "sparse_model_id": "s" if sparse else None,
        "bge_model_id": "b" if bge else None,
    }


def _value(text: EmbedInput) -> float:
    assert isinstance(text, str)
    return float(ord(text))


def _fake_parsed_embed(payload: EmbedRequestPayload) -> ParsedEmbedResponse:
    texts = payload["texts"]
    vectors = np.asarray([[_value(text), _value(text) + 1] for text in texts], dtype=np.float32)

    def sparse_items() -> list[SparseEmbedding]:
        return [
            SparseEmbedding(
                dim=4,
                indices=np.array([0], dtype=np.uint32),
                values=np.array([_value(text)], dtype=np.float32),
            )
            for text in texts
        ]

    dense_id = payload.get("dense_model_id")
    sparse_id = payload.get("sparse_model_id")
    bge_id = payload.get("bge_model_id")
    return assemble_parsed_response(
        texts_count=len(texts),
        dense=DenseEmbeddings(model_id=dense_id, vectors=vectors) if dense_id else None,
        sparse=SparseEmbeddings(model_id=sparse_id, items=sparse_items()) if sparse_id else None,
        bge_m3=BGEM3Embeddings(
            model_id=bge_id,
            dense=DenseEmbeddings(model_id=bge_id, vectors=vectors.copy()),
            sparse=SparseEmbeddings(model_id=bge_id, items=sparse_items()),
            colbert=[np.full((2, 2), _value(text), dtype=np.float32) for text in texts],
        ) if bge_id else None,
    )


class _EmbedHarnessClient(MeowEmbedClient):
    def _embed_remote(self, payload: EmbedRequestPayload) -> ParsedEmbedResponse:
        return _fake_parsed_embed(payload)

    async def _aembed_remote(self, payload: EmbedRequestPayload) -> ParsedEmbedResponse:
        return _fake_parsed_embed(payload)


def _assert_batch(
    result: ParsedEmbedResponse, texts: Sequence[EmbedInput], dense: bool, sparse: bool, bge: bool
) -> None:
    assert isinstance(result, ParsedEmbedResponse)
    assert result.texts_count == len(texts)
    assert (result.dense is not None) == dense
    assert (result.sparse is not None) == sparse
    assert (result.bgeM3 is not None) == bge
    expected = np.asarray([[_value(text), _value(text) + 1] for text in texts], dtype=np.float32)
    if result.dense is not None:
        assert result.dense.model_id == "d"
        assert result.dense.vectors.shape == (len(texts), 2)
        assert result.dense.vectors.dtype == np.float32
        np.testing.assert_array_equal(result.dense.vectors, expected)
    sparse_results = []
    if result.sparse is not None:
        assert result.sparse.model_id == "s"
        sparse_results.append(result.sparse)
    if result.bgeM3 is not None:
        assert result.bgeM3.model_id == "b"
        np.testing.assert_array_equal(result.bgeM3.dense.vectors, expected)
        assert len(result.bgeM3.colbert) == len(texts)
        for matrix, text in zip(result.bgeM3.colbert, texts, strict=True):
            assert matrix.shape == (2, 2)
            assert matrix.dtype == np.float32
            np.testing.assert_array_equal(matrix, np.full((2, 2), _value(text)))
        sparse_results.append(result.bgeM3.sparse)
    for embeddings in sparse_results:
        assert len(embeddings.items) == len(texts)
        for item, text in zip(embeddings.items, texts, strict=True):
            assert item.dim == 4
            assert item.indices.dtype == np.uint32
            assert item.values.dtype == np.float32
            np.testing.assert_array_equal(item.indices, [0])
            np.testing.assert_array_equal(item.values, [_value(text)])
    assert isinstance(result.client_timings, dict)


def _assert_one(result: ParsedEmbedOne, dense: bool, sparse: bool, bge: bool) -> None:
    assert isinstance(result, ParsedEmbedOne)
    assert (result.dense is not None) == dense
    assert (result.sparse is not None) == sparse
    assert (result.bgeM3 is not None) == bge
    if result.dense is not None:
        assert result.dense.model_id == "d"
        assert result.dense.vector.shape == (2,)
        np.testing.assert_array_equal(result.dense.vector, [97, 98])
    if result.sparse is not None:
        assert result.sparse.model_id == "s"
        assert result.sparse.item.dim == 4
        np.testing.assert_array_equal(result.sparse.item.indices, [0])
        np.testing.assert_array_equal(result.sparse.item.values, [97])
    if result.bgeM3 is not None:
        assert result.bgeM3.model_id == "b"
        assert result.bgeM3.dense.vector.shape == (2,)
        np.testing.assert_array_equal(result.bgeM3.dense.vector, [97, 98])
        np.testing.assert_array_equal(result.bgeM3.sparse.item.values, [97])
        assert result.bgeM3.colbert.shape == (2, 2)
        np.testing.assert_array_equal(result.bgeM3.colbert, np.full((2, 2), 97))
    assert isinstance(result.client_timings, dict)


@pytest.mark.parametrize("dense,sparse,bge", MODEL_COMBINATIONS)
def test_embed_and_embed_one_simple_types(dense: bool, sparse: bool, bge: bool) -> None:
    with httpx.Client(base_url="http://embed-harness.invalid") as http:
        meow = _EmbedHarnessClient(client=http)
        payload = _payload(("a", "b"), dense, sparse, bge)
        result = meow.embed(payload, use_cache=False)
        assert_type(result, ParsedEmbedResponse)
        _assert_batch(result, payload["texts"], dense, sparse, bge)
        assert result.server_timings is None
        single = meow.embed_one(_one_payload(dense, sparse, bge), use_cache=False)
        assert_type(single, ParsedEmbedOne)
        _assert_one(single, dense, sparse, bge)
        assert single.server_timings is None


@pytest.mark.anyio
@pytest.mark.parametrize("dense,sparse,bge", MODEL_COMBINATIONS)
async def test_aembed_and_aembed_one_simple_types(dense: bool, sparse: bool, bge: bool) -> None:
    async with httpx.AsyncClient(base_url="http://embed-harness.invalid") as http:
        meow = _EmbedHarnessClient(aclient=http)
        payload = _payload(("a", "b"), dense, sparse, bge)
        result = await meow.aembed(payload, use_cache=False)
        assert_type(result, ParsedEmbedResponse)
        _assert_batch(result, payload["texts"], dense, sparse, bge)
        assert result.server_timings is None
        single = await meow.aembed_one(_one_payload(dense, sparse, bge), use_cache=False)
        assert_type(single, ParsedEmbedOne)
        _assert_one(single, dense, sparse, bge)
        assert single.server_timings is None


@pytest.mark.anyio
@pytest.mark.parametrize("dense,sparse,bge", MODEL_COMBINATIONS)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_cache_combinations_misses_duplicates_order_and_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    dense: bool, sparse: bool, bge: bool, asynchronous: bool,
) -> None:
    cache = EmbedCache.open(tmp_path / "cache")
    calls: list[list[EmbedInput]] = []

    def remote(payload: EmbedRequestPayload) -> ParsedEmbedResponse:
        calls.append(list(payload["texts"]))
        return _fake_parsed_embed(payload)

    async def aremote(payload: EmbedRequestPayload) -> ParsedEmbedResponse:
        return remote(payload)

    try:
        with httpx.Client(base_url="http://embed-harness.invalid") as http:
            async with httpx.AsyncClient(base_url="http://embed-harness.invalid") as ahttp:
                meow = _EmbedHarnessClient(client=http, aclient=ahttp, cache=cache)
                monkeypatch.setattr(meow, "_embed_remote", remote)
                monkeypatch.setattr(meow, "_aembed_remote", aremote)

                async def embed(texts: Sequence[EmbedInput]) -> ParsedEmbedResponse:
                    payload = _payload(texts, dense, sparse, bge)
                    return await meow.aembed(payload) if asynchronous else meow.embed(payload)

                first = await embed(["a", "b"])
                _assert_batch(first, ["a", "b"], dense, sparse, bge)
                assert calls == [["a", "b"]]
                mixed_texts = ["b", "c", "a", "b"]
                mixed = await embed(mixed_texts)
                _assert_batch(mixed, mixed_texts, dense, sparse, bge)
                assert calls == [["a", "b"], ["c"]]

                def fail_remote(payload: EmbedRequestPayload) -> ParsedEmbedResponse:
                    raise AssertionError(f"Cache hit unexpectedly called remote: {payload}")

                async def fail_aremote(payload: EmbedRequestPayload) -> ParsedEmbedResponse:
                    return fail_remote(payload)

                monkeypatch.setattr(meow, "_embed_remote", fail_remote)
                monkeypatch.setattr(meow, "_aembed_remote", fail_aremote)
                hit_texts = ["c", "b", "a", "a"]
                hit = await embed(hit_texts)
                _assert_batch(hit, hit_texts, dense, sparse, bge)
                assert hit.server_timings is None
                one_payload = _one_payload(dense, sparse, bge)
                single = await meow.aembed_one(one_payload) if asynchronous else meow.embed_one(one_payload)
                _assert_one(single, dense, sparse, bge)
                assert single.server_timings is None
                assert calls == [["a", "b"], ["c"]]
    finally:
        cache.close()


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("missing", ["dense", "sparse", "bgeM3"])
async def test_cache_rejects_missing_expected_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, missing: str
) -> None:
    cache = EmbedCache.open(tmp_path / "cache")

    def remote(payload: EmbedRequestPayload) -> ParsedEmbedResponse:
        result = _fake_parsed_embed(payload)
        setattr(result, missing, None)
        return result

    async def aremote(payload: EmbedRequestPayload) -> ParsedEmbedResponse:
        return remote(payload)

    try:
        with httpx.Client(base_url="http://embed-harness.invalid") as http:
            async with httpx.AsyncClient(base_url="http://embed-harness.invalid") as ahttp:
                meow = _EmbedHarnessClient(client=http, aclient=ahttp, cache=cache)
                monkeypatch.setattr(meow, "_embed_remote", remote)
                monkeypatch.setattr(meow, "_aembed_remote", aremote)
                payload = _payload(["a"], True, True, True)
                with pytest.raises(ValueError, match="expected .*embeddings"):
                    if asynchronous:
                        await meow.aembed(payload)
                    else:
                        meow.embed(payload)
    finally:
        cache.close()


@pytest.mark.anyio
@pytest.mark.parametrize("dense,sparse,bge", MODEL_COMBINATIONS)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_cache_single_miss_then_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dense: bool,
    sparse: bool,
    bge: bool,
    asynchronous: bool,
) -> None:
    cache = EmbedCache.open(tmp_path / "cache")
    calls: list[list[EmbedInput]] = []

    def remote(payload: EmbedRequestPayload) -> ParsedEmbedResponse:
        calls.append(list(payload["texts"]))
        return _fake_parsed_embed(payload)

    async def aremote(payload: EmbedRequestPayload) -> ParsedEmbedResponse:
        return remote(payload)

    try:
        with httpx.Client(base_url="http://embed-harness.invalid") as http:
            async with httpx.AsyncClient(base_url="http://embed-harness.invalid") as ahttp:
                meow = _EmbedHarnessClient(client=http, aclient=ahttp, cache=cache)
                monkeypatch.setattr(meow, "_embed_remote", remote)
                monkeypatch.setattr(meow, "_aembed_remote", aremote)
                payload = _one_payload(dense, sparse, bge)
                for _ in range(2):
                    if asynchronous:
                        result = await meow.aembed_one(payload)
                    else:
                        result = meow.embed_one(payload)
                    assert_type(result, ParsedEmbedOne)
                    _assert_one(result, dense, sparse, bge)
                assert calls == [["a"]]
    finally:
        cache.close()


@pytest.mark.parametrize("missing", ["texts_count", "dense", "sparse", "bgeM3"])
def test_embed_one_rejects_invalid_batch_size(missing: str) -> None:
    result = _fake_parsed_embed(_payload(["a", "b"], True, True, True))
    if missing != "texts_count":
        result.texts_count = 1
        if missing != "dense":
            assert result.dense is not None
            result.dense.vectors = result.dense.vectors[:1]
        if missing != "sparse":
            assert result.sparse is not None
            result.sparse.items = result.sparse.items[:1]
        if missing != "bgeM3":
            assert result.bgeM3 is not None
            result.bgeM3.dense.vectors = result.bgeM3.dense.vectors[:1]
            result.bgeM3.sparse.items = result.bgeM3.sparse.items[:1]
            result.bgeM3.colbert = result.bgeM3.colbert[:1]
    with pytest.raises(ValueError, match="embed_one requires"):
        MeowEmbedClient._parsed_embed_batch_to_one(result)


INVALID_PAYLOADS = [
    pytest.param({"dense_model_id": "d"}, "At least one text must be provided", id="missing-texts"),
    pytest.param({"texts": [], "dense_model_id": "d"}, "At least one text must be provided", id="empty-texts"),
    pytest.param({"texts": ["a"]}, "At least one model must be provided", id="missing-models"),
    pytest.param({"texts": ["a"], "dense_model_id": None, "sparse_model_id": None, "bge_model_id": None}, "At least one model must be provided", id="null-models"),
]


@pytest.mark.parametrize("payload,message", INVALID_PAYLOADS)
def test_embed_validation(payload: dict[str, Any], message: str) -> None:
    with httpx.Client(base_url="http://embed-harness.invalid") as http:
        meow = _EmbedHarnessClient(client=http)
        with pytest.raises(ValueError, match=message):
            meow.embed(cast(EmbedRequestPayload, payload), use_cache=False)


@pytest.mark.anyio
@pytest.mark.parametrize("payload,message", INVALID_PAYLOADS)
async def test_aembed_validation(payload: dict[str, Any], message: str) -> None:
    async with httpx.AsyncClient(base_url="http://embed-harness.invalid") as http:
        meow = _EmbedHarnessClient(aclient=http)
        with pytest.raises(ValueError, match=message):
            await meow.aembed(cast(EmbedRequestPayload, payload), use_cache=False)


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_embed_one_raises_when_text_missing(asynchronous: bool) -> None:
    with httpx.Client(base_url="http://embed-harness.invalid") as http:
        async with httpx.AsyncClient(base_url="http://embed-harness.invalid") as ahttp:
            meow = _EmbedHarnessClient(client=http, aclient=ahttp)
            payload = cast(EmbedOneRequestPayload, {"dense_model_id": "d"})
            with pytest.raises(ValueError, match="text must be provided"):
                if asynchronous:
                    await meow.aembed_one(payload, use_cache=False)
                else:
                    meow.embed_one(payload, use_cache=False)


def test_client_requires_http_client() -> None:
    with pytest.raises(ValueError, match="client or aclient"):
        MeowEmbedClient()

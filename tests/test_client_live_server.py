import os
import sys
from pathlib import Path

import httpx
import lmdb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meow_embed import EmbedCache, MeowEmbedClient
from meow_embed.types import (
    EmbedRequestPayload,
    ParsedEmbedOne,
    ParsedEmbedResponse,
)


def _assert_embed_timings_present(
    *,
    server_timings: dict[str, float] | None,
    client_timings: dict[str, float],
    total_key: str,
) -> None:
    assert isinstance(client_timings, dict)
    assert total_key in client_timings
    assert "remote_request_ms" in client_timings or "cache_prepare_ms" in client_timings
    if server_timings is not None:
        assert isinstance(server_timings, dict)


@pytest.mark.anyio
async def test_client_models_and_embed_live_server() -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        meow = MeowEmbedClient(aclient=httpx_aclient)

        models = await meow.amodels()
        assert "models" in models
        assert len(models["models"]) > 0

        available_model_ids = {model["id"] for model in models["models"]}
        dense_model_id = "sergeyzh/BERTA"
        sparse_model_id = (
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        )

        assert dense_model_id in available_model_ids
        assert sparse_model_id in available_model_ids

        result = await meow.aembed(
            {
                "texts": ["hello world", "server integration test"],
                "dense_model_id": dense_model_id,
                "sparse_model_id": sparse_model_id,
            }
        )
        assert isinstance(result, ParsedEmbedResponse)

        assert result.texts_count == 2
        assert result.bgeM3 is None
        assert result.dense is not None
        assert result.dense.model_id == dense_model_id
        assert result.dense.vectors.shape[0] == 2
        assert result.dense.vectors.ndim == 2
        assert result.sparse is not None
        assert result.sparse.model_id == sparse_model_id
        assert len(result.sparse.items) == 2
        assert result.sparse.items[0].indices.size == result.sparse.items[0].values.size
        _assert_embed_timings_present(
            server_timings=result.server_timings,
            client_timings=result.client_timings,
            total_key="aembed_total_ms",
        )


@pytest.mark.anyio
async def test_client_aembed_one_live_server() -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        meow = MeowEmbedClient(aclient=httpx_aclient)

        models = await meow.amodels()
        available_model_ids = {model["id"] for model in models["models"]}
        dense_model_id = "sergeyzh/BERTA"
        sparse_model_id = (
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        )

        assert dense_model_id in available_model_ids
        assert sparse_model_id in available_model_ids

        one = await meow.aembed_one(
            {
                "text": "single string embed_one",
                "dense_model_id": dense_model_id,
                "sparse_model_id": sparse_model_id,
            }
        )
        assert isinstance(one, ParsedEmbedOne)
        assert one.dense is not None
        assert one.sparse is not None
        assert one.bgeM3 is None
        assert one.dense.vector.ndim == 1
        assert one.sparse.item.indices.size == one.sparse.item.values.size
        _assert_embed_timings_present(
            server_timings=one.server_timings,
            client_timings=one.client_timings,
            total_key="aembed_total_ms",
        )


@pytest.mark.anyio
async def test_client_embed_cache_hit_skips_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    dense_model_id = "sergeyzh/BERTA"
    sparse_model_id = (
        "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    )
    payload: EmbedRequestPayload = {
        "texts": ["cache me", "cache me too"],
        "dense_model_id": dense_model_id,
        "sparse_model_id": sparse_model_id,
    }

    cache = EmbedCache.open(tmp_path / "client-cache.lmdb")
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
            meow = MeowEmbedClient(aclient=httpx_aclient, cache=cache)

            first = await meow.aembed(payload)
            assert isinstance(first, ParsedEmbedResponse)

            async def _fail_if_remote_called(payload_arg: object) -> object:
                raise AssertionError(
                    f"Expected cache hit, but remote was called with: {payload_arg}"
                )

            monkeypatch.setattr(meow, "_aembed_remote", _fail_if_remote_called)
            second = await meow.aembed(payload)
            assert isinstance(second, ParsedEmbedResponse)

            assert second.texts_count == first.texts_count
            assert first.dense is not None
            assert first.sparse is not None
            assert first.bgeM3 is None
            assert second.dense is not None
            assert second.sparse is not None
            assert second.bgeM3 is None
            assert second.dense.vectors.shape == first.dense.vectors.shape
            assert np.array_equal(first.dense.vectors, second.dense.vectors)
            assert len(second.sparse.items) == len(first.sparse.items)
            for first_item, second_item in zip(
                first.sparse.items, second.sparse.items, strict=True
            ):
                assert first_item.dim == second_item.dim
                assert np.array_equal(first_item.indices, second_item.indices)
                assert np.array_equal(first_item.values, second_item.values)
            assert second.server_timings is None
            _assert_embed_timings_present(
                server_timings=second.server_timings,
                client_timings=second.client_timings,
                total_key="aembed_total_ms",
            )
    finally:
        cache.close()


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_client_accepts_external_lmdb_environment(
    tmp_path: Path, anyio_backend: str
) -> None:
    assert anyio_backend == "asyncio"
    cache_dir = tmp_path / "external-cache.lmdb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_env = lmdb.open(str(cache_dir), map_size=2 * 1024 * 1024 * 1024)
    try:
        cache = EmbedCache(env=cache_env)
        base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
            meow = MeowEmbedClient(aclient=httpx_aclient, cache=cache)
            assert meow.cache is cache
            assert meow.cache is not None
            assert meow.cache.env is cache_env
    finally:
        cache_env.close()


@pytest.mark.anyio
async def test_client_rerank_live_server() -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    reranker_model_id = "BAAI/bge-reranker-v2-m3"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        meow = MeowEmbedClient(aclient=httpx_aclient)
        models = await meow.amodels()
        available_model_ids = {
            model["id"] for model in models["models"] if model["type"] == "reranker"
        }
        if reranker_model_id not in available_model_ids:
            pytest.skip(f"{reranker_model_id} is not loaded on server")

        reranked = await meow.arerank(
            {
                "reranker_model_id": reranker_model_id,
                "queries": ["what is panda?", "capital of france"],
                "docs": [
                    "The giant panda is a bear species endemic to China.",
                    "Paris is the capital city of France.",
                ],
            }
        )

    assert reranked.model_id == reranker_model_id
    assert reranked.shape == (2, 2)
    assert len(reranked.scores) == 2
    assert all(len(row) == 2 for row in reranked.scores)
    assert all(isinstance(score, float) for row in reranked.scores for score in row)


@pytest.mark.anyio
async def test_client_embed_bge_m3_live_server() -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    bge_model_id = "BAAI/bge-m3"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        meow = MeowEmbedClient(aclient=httpx_aclient)
        models = await meow.amodels()
        bge_models = [model for model in models["models"] if model["type"] == "bgeM3"]
        available_model_ids = {
            model["id"] for model in models["models"] if model["type"] == "bgeM3"
        }
        if bge_model_id not in available_model_ids:
            pytest.skip(f"{bge_model_id} is not loaded on server")
        bge_model_info = next(
            model for model in bge_models if model["id"] == bge_model_id
        )
        assert bge_model_info.get("dense_dimensions") is not None
        assert bge_model_info.get("sparse_dimensions") is not None
        assert bge_model_info.get("batch_size") is not None

        raw_response = await httpx_aclient.post(
            "/embed",
            json={
                "texts": ["What is BGE M3?", "Definition of BM25"],
                "bge_model_id": bge_model_id,
            },
        )
        raw_response.raise_for_status()
        raw_payload = raw_response.json()
        assert raw_payload["texts_count"] == 2
        assert raw_payload["bgeM3"]["model_id"] == bge_model_id
        assert raw_payload["bgeM3"]["dense"]["model_id"] == bge_model_id
        assert raw_payload["bgeM3"]["sparse"]["model_id"] == bge_model_id
        assert len(raw_payload["bgeM3"]["colbert"]) == 2

        parsed = await meow.aembed(
            {
                "texts": ["What is BGE M3?", "Definition of BM25"],
                "bge_model_id": bge_model_id,
            }
        )
        assert isinstance(parsed, ParsedEmbedResponse)
        assert parsed.texts_count == 2
        assert parsed.dense is None
        assert parsed.sparse is None
        assert parsed.bgeM3 is not None
        assert parsed.bgeM3.model_id == bge_model_id
        assert parsed.bgeM3.dense.vectors.shape[0] == 2
        assert len(parsed.bgeM3.sparse.items) == 2
        assert len(parsed.bgeM3.colbert) == 2


@pytest.mark.anyio
async def test_client_embed_bge_m3_cache_hit_skips_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    bge_model_id = "BAAI/bge-m3"
    payload: EmbedRequestPayload = {
        "texts": ["cache bge one", "cache bge two"],
        "bge_model_id": bge_model_id,
    }
    cache = EmbedCache.open(tmp_path / "client-cache.lmdb")
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
            meow = MeowEmbedClient(aclient=httpx_aclient, cache=cache)
            models = await meow.amodels()
            available_model_ids = {
                model["id"] for model in models["models"] if model["type"] == "bgeM3"
            }
            if bge_model_id not in available_model_ids:
                pytest.skip(f"{bge_model_id} is not loaded on server")

            first = await meow.aembed(payload)
            assert isinstance(first, ParsedEmbedResponse)

            async def _fail_if_remote_called(payload_arg: object) -> object:
                raise AssertionError(
                    f"Expected cache hit, but remote was called with: {payload_arg}"
                )

            monkeypatch.setattr(meow, "_aembed_remote", _fail_if_remote_called)
            second = await meow.aembed(payload)
            assert isinstance(second, ParsedEmbedResponse)
            assert first.bgeM3 is not None
            assert second.bgeM3 is not None
            assert first.dense is None
            assert first.sparse is None
            assert second.dense is None
            assert second.sparse is None
            assert np.array_equal(first.bgeM3.dense.vectors, second.bgeM3.dense.vectors)
            assert len(first.bgeM3.sparse.items) == len(second.bgeM3.sparse.items)
            for first_item, second_item in zip(
                first.bgeM3.sparse.items, second.bgeM3.sparse.items, strict=True
            ):
                assert first_item.dim == second_item.dim
                assert np.array_equal(first_item.indices, second_item.indices)
                assert np.array_equal(first_item.values, second_item.values)
            assert len(first.bgeM3.colbert) == len(second.bgeM3.colbert)
            for first_row, second_row in zip(
                first.bgeM3.colbert, second.bgeM3.colbert, strict=True
            ):
                assert np.array_equal(first_row, second_row)
    finally:
        cache.close()


@pytest.mark.anyio
async def test_sync_api_requires_sync_client() -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        meow = MeowEmbedClient(aclient=httpx_aclient)
        with pytest.raises(RuntimeError, match="client is not configured"):
            meow.models()


@pytest.mark.anyio
async def test_client_models_and_embed_sync_live_server() -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    dense_model_id = "sergeyzh/BERTA"
    sparse_model_id = (
        "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    )
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
            meow = MeowEmbedClient(sync_httpx_aclient, httpx_aclient)

            models = meow.models()
            assert "models" in models
            assert len(models["models"]) > 0

            available_model_ids = {model["id"] for model in models["models"]}
            assert dense_model_id in available_model_ids
            assert sparse_model_id in available_model_ids

            result = meow.embed(
                {
                    "texts": ["hello world", "server integration test"],
                    "dense_model_id": dense_model_id,
                    "sparse_model_id": sparse_model_id,
                }
            )
            assert isinstance(result, ParsedEmbedResponse)

            assert result.texts_count == 2
            assert result.bgeM3 is None
            assert result.dense is not None
            assert result.dense.model_id == dense_model_id
            assert result.dense.vectors.shape[0] == 2
            assert result.dense.vectors.ndim == 2
            assert result.sparse is not None
            assert result.sparse.model_id == sparse_model_id
            assert len(result.sparse.items) == 2
            assert (
                result.sparse.items[0].indices.size
                == result.sparse.items[0].values.size
            )
            _assert_embed_timings_present(
                server_timings=result.server_timings,
                client_timings=result.client_timings,
                total_key="embed_total_ms",
            )


@pytest.mark.anyio
async def test_client_embed_sync_cache_hit_skips_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    dense_model_id = "sergeyzh/BERTA"
    sparse_model_id = (
        "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    )
    payload: EmbedRequestPayload = {
        "texts": ["cache me sync", "cache me sync too"],
        "dense_model_id": dense_model_id,
        "sparse_model_id": sparse_model_id,
    }

    cache = EmbedCache.open(tmp_path / "client-cache-sync.lmdb")
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
            with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
                meow = MeowEmbedClient(
                    sync_httpx_aclient,
                    httpx_aclient,
                    cache=cache,
                )

                first = meow.embed(payload)
                assert isinstance(first, ParsedEmbedResponse)

                def _fail_if_remote_called(payload_arg: object) -> object:
                    raise AssertionError(
                        f"Expected cache hit, but remote was called with: {payload_arg}"
                    )

                monkeypatch.setattr(meow, "_embed_remote", _fail_if_remote_called)

                second = meow.embed(payload)
                assert isinstance(second, ParsedEmbedResponse)

                assert second.texts_count == first.texts_count
                assert first.dense is not None
                assert first.sparse is not None
                assert first.bgeM3 is None
                assert second.dense is not None
                assert second.sparse is not None
                assert second.bgeM3 is None
                assert second.dense.vectors.shape == first.dense.vectors.shape
                assert np.array_equal(first.dense.vectors, second.dense.vectors)
                assert len(second.sparse.items) == len(first.sparse.items)
                for first_item, second_item in zip(
                    first.sparse.items, second.sparse.items, strict=True
                ):
                    assert first_item.dim == second_item.dim
                    assert np.array_equal(first_item.indices, second_item.indices)
                    assert np.array_equal(first_item.values, second_item.values)
    finally:
        cache.close()


@pytest.mark.anyio
async def test_client_sync_rerank_live_server() -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    reranker_model_id = "BAAI/bge-reranker-v2-m3"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
            meow = MeowEmbedClient(sync_httpx_aclient, httpx_aclient)
            models = meow.models()
            available_model_ids = {
                model["id"] for model in models["models"] if model["type"] == "reranker"
            }
            if reranker_model_id not in available_model_ids:
                pytest.skip(f"{reranker_model_id} is not loaded on server")

            reranked = meow.rerank(
                {
                    "reranker_model_id": reranker_model_id,
                    "queries": ["what is panda?", "capital of france"],
                    "docs": [
                        "The giant panda is a bear species endemic to China.",
                        "Paris is the capital city of France.",
                    ],
                }
            )

    assert reranked.model_id == reranker_model_id
    assert reranked.shape == (2, 2)
    assert len(reranked.scores) == 2
    assert all(len(row) == 2 for row in reranked.scores)
    assert all(isinstance(score, float) for row in reranked.scores for score in row)


@pytest.mark.anyio
async def test_client_embed_bge_m3_sync_live_server() -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    bge_model_id = "BAAI/bge-m3"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
        with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
            meow = MeowEmbedClient(sync_httpx_aclient, httpx_aclient)
            models = meow.models()
            available_model_ids = {
                model["id"] for model in models["models"] if model["type"] == "bgeM3"
            }
            if bge_model_id not in available_model_ids:
                pytest.skip(f"{bge_model_id} is not loaded on server")

            parsed = meow.embed(
                {
                    "texts": ["What is BGE M3?", "Definition of BM25"],
                    "bge_model_id": bge_model_id,
                }
            )
            assert isinstance(parsed, ParsedEmbedResponse)
            assert parsed.texts_count == 2
            assert parsed.dense is None
            assert parsed.sparse is None
            assert parsed.bgeM3 is not None
            assert parsed.bgeM3.model_id == bge_model_id
            assert parsed.bgeM3.dense.vectors.shape[0] == 2
            assert len(parsed.bgeM3.sparse.items) == 2
            assert len(parsed.bgeM3.colbert) == 2


@pytest.mark.anyio
async def test_client_embed_bge_m3_sync_cache_hit_skips_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = os.getenv("MEOW_EMBED_BASE_URL", "http://127.0.0.1:8067")
    bge_model_id = "BAAI/bge-m3"
    payload: EmbedRequestPayload = {
        "texts": ["cache bge sync one", "cache bge sync two"],
        "bge_model_id": bge_model_id,
    }
    cache = EmbedCache.open(tmp_path / "client-cache-bge-sync.lmdb")
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as httpx_aclient:
            with httpx.Client(base_url=base_url, timeout=30.0) as sync_httpx_aclient:
                meow = MeowEmbedClient(
                    sync_httpx_aclient,
                    httpx_aclient,
                    cache=cache,
                )
                models = meow.models()
                available_model_ids = {
                    model["id"]
                    for model in models["models"]
                    if model["type"] == "bgeM3"
                }
                if bge_model_id not in available_model_ids:
                    pytest.skip(f"{bge_model_id} is not loaded on server")

                first = meow.embed(payload)
                assert isinstance(first, ParsedEmbedResponse)

                def _fail_if_remote_called(payload_arg: object) -> object:
                    raise AssertionError(
                        f"Expected cache hit, but remote was called with: {payload_arg}"
                    )

                monkeypatch.setattr(meow, "_embed_remote", _fail_if_remote_called)

                second = meow.embed(payload)
                assert isinstance(second, ParsedEmbedResponse)
                assert first.bgeM3 is not None
                assert second.bgeM3 is not None
                assert first.dense is None
                assert first.sparse is None
                assert second.dense is None
                assert second.sparse is None
                assert np.array_equal(
                    first.bgeM3.dense.vectors, second.bgeM3.dense.vectors
                )
                assert len(first.bgeM3.sparse.items) == len(second.bgeM3.sparse.items)
                for first_item, second_item in zip(
                    first.bgeM3.sparse.items, second.bgeM3.sparse.items, strict=True
                ):
                    assert first_item.dim == second_item.dim
                    assert np.array_equal(first_item.indices, second_item.indices)
                    assert np.array_equal(first_item.values, second_item.values)
                assert len(first.bgeM3.colbert) == len(second.bgeM3.colbert)
                for first_row, second_row in zip(
                    first.bgeM3.colbert, second.bgeM3.colbert, strict=True
                ):
                    assert np.array_equal(first_row, second_row)
    finally:
        cache.close()

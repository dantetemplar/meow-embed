from __future__ import annotations

import asyncio
import gzip
import json
import time
from contextvars import ContextVar
from typing import cast, overload

import httpx
import numpy as np

import meow_embed.types as t
from meow_embed.cache import EmbedCache, EmbedCacheProgress
from meow_embed.parsing import decode_embed_response
from meow_embed.media import normalize_embed_payload

timing_timeline = ContextVar[dict[str, float] | None]("timing_timeline", default=None)


class MeowEmbedClient:
    @staticmethod
    def _current_timeline_ts(timings: dict[str, float]) -> float:
        if not timings:
            return 0.0
        last_key = next(reversed(timings))
        return timings[last_key]

    @classmethod
    def _append_timeline_timestamp(
        cls, key: str, timestamp: float | None = None
    ) -> None:
        timings = timing_timeline.get()
        if timings is None:
            return
        timings[key] = time.perf_counter() if timestamp is None else timestamp

    @staticmethod
    def _timeline_snapshot() -> dict[str, float]:
        timings = timing_timeline.get()
        return {} if timings is None else dict(timings)

    @staticmethod
    def _parse_server_timings(headers: httpx.Headers) -> dict[str, float] | None:
        raw_header = headers.get("X-Timing-Context-Raw")
        if not raw_header:
            return None
        try:
            parsed = json.loads(raw_header)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        points = [(k, v) for k, v in parsed.items() if isinstance(v, (int, float))]
        if not points:
            return None
        return {f"{key}_ms": float(value) * 1000.0 for key, value in points}

    @staticmethod
    def _server_total_ms(server_timings: dict[str, float] | None) -> float | None:
        if not server_timings:
            return None
        values = list(server_timings.values())
        if not values:
            return None
        return max(0.0, values[-1] - values[0])

    def __init__(
        self,
        client: httpx.Client | None = None,
        aclient: httpx.AsyncClient | None = None,
        *,
        use_cache: bool = False,
        cache: EmbedCache | None = None,
    ) -> None:
        # Caller owns lifecycle and configuration (base_url, timeout, headers).
        self._client = client
        self._aclient = aclient
        if self._client is None and self._aclient is None:
            raise ValueError("Either client or aclient must be provided, may be both.")
        if cache is None and use_cache:
            cache = EmbedCache.open()
        self._cache = cache

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError(
                "client is not configured; pass client=httpx.Client(...) to MeowEmbedClient(...)."
            )
        return self._client

    @property
    def aclient(self) -> httpx.AsyncClient:
        if self._aclient is None:
            raise RuntimeError(
                "aclient is not configured; pass aclient=httpx.AsyncClient(...) to MeowEmbedClient(...)."
            )
        return self._aclient

    @property
    def cache(self) -> EmbedCache | None:
        return self._cache

    async def amodels(self) -> t.ModelsResponseDict:
        response = await self.aclient.get(
            "/models", headers={"Accept-Encoding": "gzip"}
        )
        response.raise_for_status()
        return response.json()

    def models(self) -> t.ModelsResponseDict:
        response = self.client.get("/models", headers={"Accept-Encoding": "gzip"})
        response.raise_for_status()
        return response.json()

    @overload
    def embed(
        self,
        payload: t.DenseSparseBGEM3EmbedRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> t.ParsedEmbedResponseDenseSparseBGEM3: ...

    @overload
    def embed(
        self, payload: t.DenseSparseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseDenseSparse: ...

    @overload
    def embed(
        self, payload: t.DenseBGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseDenseBGEM3: ...

    @overload
    def embed(
        self, payload: t.SparseBGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseSparseBGEM3: ...

    @overload
    def embed(
        self, payload: t.DenseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseDense: ...

    @overload
    def embed(
        self, payload: t.SparseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseSparse: ...

    @overload
    def embed(
        self, payload: t.BGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseBGEM3: ...

    def embed(
        self, payload: t.EmbedRequestPayload, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseVariant:
        timeline_token = timing_timeline.set({})
        try:
            payload = normalize_embed_payload(payload)
            self._validate_embed_payload(payload)
            self._append_timeline_timestamp("payload_validate_ms")

            should_use_cache = (
                (self._cache is not None) if use_cache is None else use_cache
            )
            if not should_use_cache or self._cache is None:
                parsed = self._embed_remote(payload)
                parsed.client_timings["embed_total_ms"] = self._current_timeline_ts(
                    parsed.client_timings
                )
                return parsed

            prepared = self._cache.prepare(payload)
            self._append_timeline_timestamp("cache_prepare_ms")
            server_timings: dict[str, float] | None = None
            if prepared.misses:
                miss_payload = self._payload_for_missing_in_cache(prepared)
                self._append_timeline_timestamp("missing_payload_build_ms")
                remote = self._embed_remote(miss_payload)
                server_timings = remote.server_timings
                if remote.texts_count != len(prepared.misses):
                    raise ValueError(
                        "Cache merge failed: mismatch in misses and returned items."
                    )
                self._cache.merge_remote(prepared, remote)
                self._append_timeline_timestamp("cache_merge_remote_ms")
            parsed = self._cache.finalize(prepared)
            parsed.server_timings = server_timings
            parsed.client_timings = self._timeline_snapshot()
            self._append_timeline_timestamp("cache_finalize_ms")
            parsed.client_timings = self._timeline_snapshot()
            parsed.client_timings["embed_total_ms"] = self._current_timeline_ts(
                parsed.client_timings
            )
            return parsed
        finally:
            timing_timeline.reset(timeline_token)

    def _embed_remote(
        self,
        payload: t.EmbedRequestPayload,
    ) -> t.ParsedEmbedResponseVariant:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        self._append_timeline_timestamp("payload_compress_ms")
        response = self.client.post(
            "/embed",
            content=gzipped_payload,
            headers={
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        self._append_timeline_timestamp("remote_request_ms")
        response.raise_for_status()
        server_timings = self._parse_server_timings(response.headers)
        raw = cast(t.EmbedResponseDict, response.json())
        self._append_timeline_timestamp("remote_json_decode_ms")
        parsed = decode_embed_response(
            raw,
            payload,
            server_timings=server_timings,
            client_timings=self._timeline_snapshot(),
        )
        self._append_timeline_timestamp("response_parse_ms")
        parsed.client_timings = self._timeline_snapshot()
        return parsed

    @classmethod
    def _validate_embed_payload(cls, payload: t.EmbedRequestPayload) -> None:
        if not payload.get("texts"):
            raise ValueError("At least one text must be provided.")
        if (
            not payload.get("dense_model_id")
            and not payload.get("sparse_model_id")
            and not payload.get("bge_model_id")
        ):
            raise ValueError("At least one model must be provided.")

    def _payload_for_missing_in_cache(
        self, prepared: EmbedCacheProgress
    ) -> t.EmbedRequestPayload:
        miss_payload = dict(prepared.payload)
        miss_payload["texts"] = [prepared.texts[idx] for idx in prepared.misses]
        return cast(t.EmbedRequestPayload, miss_payload)

    @overload
    async def aembed(
        self,
        payload: t.DenseSparseBGEM3EmbedRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> t.ParsedEmbedResponseDenseSparseBGEM3: ...

    @overload
    async def aembed(
        self, payload: t.DenseSparseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseDenseSparse: ...

    @overload
    async def aembed(
        self, payload: t.DenseBGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseDenseBGEM3: ...

    @overload
    async def aembed(
        self, payload: t.SparseBGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseSparseBGEM3: ...

    @overload
    async def aembed(
        self, payload: t.DenseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseDense: ...

    @overload
    async def aembed(
        self, payload: t.SparseEmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseSparse: ...

    @overload
    async def aembed(
        self, payload: t.BGEM3EmbedRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseBGEM3: ...

    async def aembed(
        self, payload: t.EmbedRequestPayload, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedResponseVariant:
        timeline_token = timing_timeline.set({})
        try:
            payload = await asyncio.to_thread(normalize_embed_payload, payload)
            self._validate_embed_payload(payload)
            self._append_timeline_timestamp("payload_validate_ms")

            should_use_cache = (
                (self._cache is not None) if use_cache is None else use_cache
            )
            if not should_use_cache or self._cache is None:
                parsed = await self._aembed_remote(payload)
                parsed.client_timings["aembed_total_ms"] = self._current_timeline_ts(
                    parsed.client_timings
                )
                return parsed

            prepared = self._cache.prepare(payload)
            self._append_timeline_timestamp("cache_prepare_ms")
            server_timings: dict[str, float] | None = None
            if prepared.misses:
                miss_payload = self._payload_for_missing_in_cache(prepared)
                self._append_timeline_timestamp("missing_payload_build_ms")
                remote = await self._aembed_remote(miss_payload)
                server_timings = remote.server_timings
                if remote.texts_count != len(prepared.misses):
                    raise ValueError(
                        "Cache merge failed: mismatch in misses and returned items."
                    )
                self._cache.merge_remote(prepared, remote)
                self._append_timeline_timestamp("cache_merge_remote_ms")
            parsed = self._cache.finalize(prepared)
            parsed.server_timings = server_timings
            parsed.client_timings = self._timeline_snapshot()
            self._append_timeline_timestamp("cache_finalize_ms")
            parsed.client_timings = self._timeline_snapshot()
            parsed.client_timings["aembed_total_ms"] = self._current_timeline_ts(
                parsed.client_timings
            )
            return parsed
        finally:
            timing_timeline.reset(timeline_token)

    async def _aembed_remote(
        self,
        payload: t.EmbedRequestPayload,
    ) -> t.ParsedEmbedResponseVariant:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        self._append_timeline_timestamp("payload_compress_ms")
        response = await self.aclient.post(
            "/embed",
            content=gzipped_payload,
            headers={
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        self._append_timeline_timestamp("remote_request_ms")
        response.raise_for_status()
        server_timings = self._parse_server_timings(response.headers)
        raw = cast(t.EmbedResponseDict, response.json())
        self._append_timeline_timestamp("remote_json_decode_ms")
        parsed = decode_embed_response(
            raw,
            payload,
            server_timings=server_timings,
            client_timings=self._timeline_snapshot(),
        )
        self._append_timeline_timestamp("response_parse_ms")
        parsed.client_timings = self._timeline_snapshot()
        return parsed

    @overload
    def embed_one(
        self,
        payload: t.DenseSparseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> t.ParsedEmbedOneDenseSparseBGEM3: ...

    @overload
    def embed_one(
        self,
        payload: t.DenseSparseEmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> t.ParsedEmbedOneDenseSparse: ...

    @overload
    def embed_one(
        self, payload: t.DenseBGEM3EmbedOneRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneDenseBGEM3: ...

    @overload
    def embed_one(
        self,
        payload: t.SparseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> t.ParsedEmbedOneSparseBGEM3: ...

    @overload
    def embed_one(
        self, payload: t.DenseEmbedOneRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneDense: ...

    @overload
    def embed_one(
        self, payload: t.SparseEmbedOneRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneSparse: ...

    @overload
    def embed_one(
        self, payload: t.BGEM3EmbedOneRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneBGEM3: ...

    def embed_one(
        self, payload: t.EmbedOneRequestPayload, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneVariant:
        batch = self._embed_one_payload_as_embed_many_payload(payload)
        parsed = self.embed(cast(t.EmbedRequestPayload, batch), use_cache=use_cache)
        return self._parsed_embed_batch_to_one(parsed)

    @overload
    async def aembed_one(
        self,
        payload: t.DenseSparseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> t.ParsedEmbedOneDenseSparseBGEM3: ...

    @overload
    async def aembed_one(
        self,
        payload: t.DenseSparseEmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> t.ParsedEmbedOneDenseSparse: ...

    @overload
    async def aembed_one(
        self, payload: t.DenseBGEM3EmbedOneRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneDenseBGEM3: ...

    @overload
    async def aembed_one(
        self,
        payload: t.SparseBGEM3EmbedOneRequestDict,
        *,
        use_cache: bool | None = None,
    ) -> t.ParsedEmbedOneSparseBGEM3: ...

    @overload
    async def aembed_one(
        self, payload: t.DenseEmbedOneRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneDense: ...

    @overload
    async def aembed_one(
        self, payload: t.SparseEmbedOneRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneSparse: ...

    @overload
    async def aembed_one(
        self, payload: t.BGEM3EmbedOneRequestDict, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneBGEM3: ...

    async def aembed_one(
        self, payload: t.EmbedOneRequestPayload, *, use_cache: bool | None = None
    ) -> t.ParsedEmbedOneVariant:
        batch = self._embed_one_payload_as_embed_many_payload(payload)
        parsed = await self.aembed(
            cast(t.EmbedRequestPayload, batch), use_cache=use_cache
        )
        return self._parsed_embed_batch_to_one(parsed)

    @classmethod
    def _embed_one_payload_as_embed_many_payload(
        cls, payload: t.EmbedOneRequestPayload
    ) -> t.EmbedRequestPayload:
        if "text" not in payload:
            raise ValueError("text must be provided.")
        text = payload["text"]
        batch = dict(cast(dict[str, object], payload))
        del batch["text"]
        batch["texts"] = [text]
        return cast(t.EmbedRequestPayload, batch)

    @classmethod
    def _parsed_embed_batch_to_one(
        cls, response: t.ParsedEmbedResponseVariant
    ) -> t.ParsedEmbedOneVariant:
        common_kwargs = {
            "server_timings": response.server_timings,
            "client_timings": dict(response.client_timings),
        }

        def _dense_emb_to_vector(dense: t.DenseEmbeddings) -> t.DenseEmbeddingVector:
            if dense.vectors.shape[0] != 1:
                raise ValueError(
                    "embed_one requires batch size 1 for dense embeddings."
                )
            return t.DenseEmbeddingVector(
                model_id=dense.model_id,
                vector=np.asarray(dense.vectors[0], dtype=np.float32),
            )

        def _sparse_emb_to_one(sparse: t.SparseEmbeddings) -> t.SparseEmbeddingsOne:
            if len(sparse.items) != 1:
                raise ValueError(
                    "embed_one requires batch size 1 for sparse embeddings."
                )
            return t.SparseEmbeddingsOne(model_id=sparse.model_id, item=sparse.items[0])

        def _bge_m3_emb_to_one(bge: t.BGEM3Embeddings) -> t.BGEM3EmbeddingOne:
            if (
                bge.dense.vectors.shape[0] != 1
                or len(bge.sparse.items) != 1
                or len(bge.colbert) != 1
            ):
                raise ValueError(
                    "embed_one requires batch size 1 for BGE-M3 embeddings."
                )
            return t.BGEM3EmbeddingOne(
                model_id=bge.model_id,
                dense=t.DenseEmbeddingVector(
                    model_id=bge.dense.model_id,
                    vector=np.asarray(bge.dense.vectors[0], dtype=np.float32),
                ),
                sparse=t.SparseEmbeddingsOne(
                    model_id=bge.sparse.model_id, item=bge.sparse.items[0]
                ),
                colbert=np.asarray(bge.colbert[0], dtype=np.float32),
            )

        if response.texts_count != 1:
            raise ValueError("embed_one requires texts_count == 1.")
        if isinstance(response, t.ParsedEmbedResponseDenseSparseBGEM3):
            return t.ParsedEmbedOneDenseSparseBGEM3(
                **common_kwargs,
                dense=_dense_emb_to_vector(response.dense),
                sparse=_sparse_emb_to_one(response.sparse),
                bgeM3=_bge_m3_emb_to_one(response.bgeM3),
            )
        if isinstance(response, t.ParsedEmbedResponseDenseSparse):
            return t.ParsedEmbedOneDenseSparse(
                **common_kwargs,
                dense=_dense_emb_to_vector(response.dense),
                sparse=_sparse_emb_to_one(response.sparse),
            )
        if isinstance(response, t.ParsedEmbedResponseDenseBGEM3):
            return t.ParsedEmbedOneDenseBGEM3(
                **common_kwargs,
                dense=_dense_emb_to_vector(response.dense),
                bgeM3=_bge_m3_emb_to_one(response.bgeM3),
            )
        if isinstance(response, t.ParsedEmbedResponseSparseBGEM3):
            return t.ParsedEmbedOneSparseBGEM3(
                **common_kwargs,
                sparse=_sparse_emb_to_one(response.sparse),
                bgeM3=_bge_m3_emb_to_one(response.bgeM3),
            )
        if isinstance(response, t.ParsedEmbedResponseDense):
            return t.ParsedEmbedOneDense(
                **common_kwargs, dense=_dense_emb_to_vector(response.dense)
            )
        if isinstance(response, t.ParsedEmbedResponseSparse):
            return t.ParsedEmbedOneSparse(
                **common_kwargs, sparse=_sparse_emb_to_one(response.sparse)
            )
        if isinstance(response, t.ParsedEmbedResponseBGEM3):
            return t.ParsedEmbedOneBGEM3(
                **common_kwargs, bgeM3=_bge_m3_emb_to_one(response.bgeM3)
            )
        raise AssertionError("Unreachable embed response variant.")

    def rerank(self, payload: t.RerankRequestDict) -> t.ParsedRerankResponse:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = self.client.post(
            "/rerank",
            content=gzipped_payload,
            headers={
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        response.raise_for_status()
        raw = cast(t.RerankResponseDict, response.json())
        shape_values = raw["shape"]
        if len(shape_values) != 2:
            raise ValueError(
                f"Rerank response shape must have 2 items, got: {shape_values!r}"
            )
        return t.ParsedRerankResponse(
            model_id=raw["model_id"],
            shape=(int(shape_values[0]), int(shape_values[1])),
            scores=raw["scores"],
        )

    async def arerank(self, payload: t.RerankRequestDict) -> t.ParsedRerankResponse:
        gzipped_payload = gzip.compress(json.dumps(payload).encode("utf-8"))
        response = await self.aclient.post(
            "/rerank",
            content=gzipped_payload,
            headers={
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )
        response.raise_for_status()
        raw = cast(t.RerankResponseDict, response.json())
        shape_values = raw["shape"]
        if len(shape_values) != 2:
            raise ValueError(
                f"Rerank response shape must have 2 items, got: {shape_values!r}"
            )
        return t.ParsedRerankResponse(
            model_id=raw["model_id"],
            shape=(int(shape_values[0]), int(shape_values[1])),
            scores=raw["scores"],
        )

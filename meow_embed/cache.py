from __future__ import annotations

import hashlib
import json
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import lmdb
import numpy as np

from meow_embed.media import normalize_embed_payload
from meow_embed.parsing import (
    all_present,
    assemble_parsed_response,
    roundtrip_float32_float16_float32,
)
from meow_embed.types import (
    BGEM3Embeddings,
    DenseEmbeddings,
    EmbedInput,
    EmbedRequestPayload,
    Float32Array,
    ParsedEmbedResponseBGEM3,
    ParsedEmbedResponseDense,
    ParsedEmbedResponseDenseBGEM3,
    ParsedEmbedResponseDenseSparse,
    ParsedEmbedResponseDenseSparseBGEM3,
    ParsedEmbedResponseSparse,
    ParsedEmbedResponseSparseBGEM3,
    ParsedEmbedResponseVariant,
    SparseEmbedding,
    SparseEmbeddings,
    UInt32Array,
)

_DEFAULT_CACHE_PATH = Path.home() / ".cache" / "meow-embed" / "client-cache.lmdb"
_DEFAULT_MAP_SIZE = 2 * 1024 * 1024 * 1024


StreamKind = Literal["dense", "sparse", "bge_dense", "bge_sparse", "bge_colbert"]


def _hash_chunks(*chunks: bytes) -> bytes:
    h = hashlib.blake2b(digest_size=32)
    for i, chunk in enumerate(chunks):
        if i > 0:
            h.update(b"\x00")
        h.update(chunk)
    return h.digest()


class _Codec[V](ABC):
    @abstractmethod
    def encode(self, value: V) -> bytes | None: ...

    @abstractmethod
    def decode(self, raw: bytes | None) -> V | None: ...


class _F16VectorCodec(_Codec[Float32Array]):
    __slots__ = ("expected_dim",)

    def __init__(self, expected_dim: int | None) -> None:
        self.expected_dim = expected_dim

    def encode(self, value: Float32Array) -> bytes:
        return np.asarray(value, dtype=np.float16).tobytes()

    def decode(self, raw: bytes | None) -> Float32Array | None:
        if raw is None:
            return None
        if self.expected_dim is not None:
            vector = np.frombuffer(
                raw, dtype=np.float16, count=self.expected_dim
            ).astype(np.float32)
            if vector.size != self.expected_dim:
                return None
            return cast(Float32Array, vector)
        return cast(
            Float32Array, np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        )


class _SparseCodec(_Codec[SparseEmbedding]):
    def encode(self, value: SparseEmbedding) -> bytes | None:
        indices = np.asarray(value.indices, dtype=np.uint32)
        values = np.asarray(value.values, dtype=np.float32)
        if indices.size != values.size:
            return None
        header = struct.pack("<II", int(value.dim), int(indices.size))
        return header + indices.tobytes() + values.tobytes()

    def decode(self, raw: bytes | None) -> SparseEmbedding | None:
        if raw is None or len(raw) < 8:
            return None
        dim, nnz = struct.unpack("<II", raw[:8])
        idx_end = 8 + (nnz * 4)
        val_end = idx_end + (nnz * 4)
        if len(raw) != val_end:
            return None
        indices = cast(
            UInt32Array, np.frombuffer(raw[8:idx_end], dtype=np.uint32).copy()
        )
        values = cast(
            Float32Array, np.frombuffer(raw[idx_end:val_end], dtype=np.float32).copy()
        )
        return SparseEmbedding(dim=int(dim), indices=indices, values=values)


class _F16MatrixCodec(_Codec[Float32Array]):
    def encode(self, value: Float32Array) -> bytes | None:
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.ndim != 2:
            return None
        rows, cols = matrix.shape
        header = struct.pack("<II", int(rows), int(cols))
        return header + matrix.astype(np.float16).tobytes()

    def decode(self, raw: bytes | None) -> Float32Array | None:
        if raw is None or len(raw) < 8:
            return None
        rows, cols = struct.unpack("<II", raw[:8])
        expected_bytes = int(rows) * int(cols) * 2
        if len(raw) != (8 + expected_bytes):
            return None
        item = np.frombuffer(raw[8:], dtype=np.float16).astype(np.float32)
        if item.size != int(rows) * int(cols):
            return None
        return cast(Float32Array, item.reshape((int(rows), int(cols))))


class _StreamHandler[V](ABC):
    kind: StreamKind

    def __init__(self, model_id: str, codec: _Codec[V]) -> None:
        self.model_id = model_id
        self.codec = codec

    @abstractmethod
    def make_keys(self, texts: Sequence[str]) -> list[bytes]: ...

    @abstractmethod
    def extract(self, remote: ParsedEmbedResponseVariant, miss_row: int) -> V: ...


_DENSE_VARIANTS = (
    ParsedEmbedResponseDense,
    ParsedEmbedResponseDenseSparse,
    ParsedEmbedResponseDenseBGEM3,
    ParsedEmbedResponseDenseSparseBGEM3,
)
_SPARSE_VARIANTS = (
    ParsedEmbedResponseSparse,
    ParsedEmbedResponseDenseSparse,
    ParsedEmbedResponseSparseBGEM3,
    ParsedEmbedResponseDenseSparseBGEM3,
)
_BGE_VARIANTS = (
    ParsedEmbedResponseBGEM3,
    ParsedEmbedResponseDenseBGEM3,
    ParsedEmbedResponseSparseBGEM3,
    ParsedEmbedResponseDenseSparseBGEM3,
)


def _expect_dense_variant(remote: ParsedEmbedResponseVariant) -> None:
    if not isinstance(remote, _DENSE_VARIANTS):
        raise ValueError("Dense cache expected dense embeddings in server response.")


def _expect_sparse_variant(remote: ParsedEmbedResponseVariant) -> None:
    if not isinstance(remote, _SPARSE_VARIANTS):
        raise ValueError("Sparse cache expected sparse embeddings in server response.")


def _expect_bge_variant(remote: ParsedEmbedResponseVariant) -> None:
    if not isinstance(remote, _BGE_VARIANTS):
        raise ValueError("BGE-M3 cache expected bgeM3 embeddings in server response.")


DenseRemote = (
    ParsedEmbedResponseDense
    | ParsedEmbedResponseDenseSparse
    | ParsedEmbedResponseDenseBGEM3
    | ParsedEmbedResponseDenseSparseBGEM3
)
SparseRemote = (
    ParsedEmbedResponseSparse
    | ParsedEmbedResponseDenseSparse
    | ParsedEmbedResponseSparseBGEM3
    | ParsedEmbedResponseDenseSparseBGEM3
)
BgeRemote = (
    ParsedEmbedResponseBGEM3
    | ParsedEmbedResponseDenseBGEM3
    | ParsedEmbedResponseSparseBGEM3
    | ParsedEmbedResponseDenseSparseBGEM3
)


def _as_dense_variant(remote: ParsedEmbedResponseVariant) -> DenseRemote:
    _expect_dense_variant(remote)
    return cast(DenseRemote, remote)


def _as_sparse_variant(remote: ParsedEmbedResponseVariant) -> SparseRemote:
    _expect_sparse_variant(remote)
    return cast(SparseRemote, remote)


def _as_bge_variant(remote: ParsedEmbedResponseVariant) -> BgeRemote:
    _expect_bge_variant(remote)
    return cast(BgeRemote, remote)


class _DenseHandler(_StreamHandler[Float32Array]):
    kind = "dense"

    def __init__(
        self,
        model_id: str,
        truncate_dim: int | None,
        dense_prompt: str,
        dense_task: Literal["query", "document"] | None,
    ) -> None:
        super().__init__(model_id, _F16VectorCodec(truncate_dim))
        self.truncate_dim = truncate_dim
        self.dense_prompt = dense_prompt
        self.dense_task = dense_task

    def make_keys(self, texts: Sequence[str]) -> list[bytes]:
        dim = 0 if self.truncate_dim is None else self.truncate_dim
        task = "" if self.dense_task is None else self.dense_task
        model_bytes = self.model_id.encode("utf-8")
        dim_bytes = dim.to_bytes(4, "little", signed=False)
        prompt_bytes = self.dense_prompt.encode("utf-8")
        task_bytes = task.encode("utf-8")
        return [
            _hash_chunks(
                b"dense",
                model_bytes,
                dim_bytes,
                b"dense_prompt",
                prompt_bytes,
                b"dense_task",
                task_bytes,
                text.encode("utf-8"),
            )
            for text in texts
        ]

    def extract(
        self, remote: ParsedEmbedResponseVariant, miss_row: int
    ) -> Float32Array:
        remote_dense = _as_dense_variant(remote)
        return cast(
            Float32Array,
            np.asarray(remote_dense.dense.vectors[miss_row], dtype=np.float32),
        )


class _SparseHandler(_StreamHandler[SparseEmbedding]):
    kind = "sparse"

    def __init__(
        self,
        model_id: str,
        max_active_dims: int | None,
        pruning_ratio: float | None,
        sparse_task: Literal["query", "document"] | None,
    ) -> None:
        super().__init__(model_id, _SparseCodec())
        self.max_active_dims = max_active_dims
        self.pruning_ratio = pruning_ratio
        self.sparse_task = sparse_task

    def make_keys(self, texts: Sequence[str]) -> list[bytes]:
        mad_bytes = (
            b""
            if self.max_active_dims is None
            else self.max_active_dims.to_bytes(4, "little", signed=False)
        )
        pr_bytes = (
            b""
            if self.pruning_ratio is None
            else str(self.pruning_ratio).encode("utf-8")
        )
        task = "" if self.sparse_task is None else self.sparse_task
        model_bytes = self.model_id.encode("utf-8")
        task_bytes = task.encode("utf-8")
        return [
            _hash_chunks(
                b"sparse",
                model_bytes,
                b"mad",
                mad_bytes,
                b"pr",
                pr_bytes,
                b"sparse_task",
                task_bytes,
                text.encode("utf-8"),
            )
            for text in texts
        ]

    def extract(
        self, remote: ParsedEmbedResponseVariant, miss_row: int
    ) -> SparseEmbedding:
        remote_sparse = _as_sparse_variant(remote)
        return remote_sparse.sparse.items[miss_row]


class _BgeHandler[V](_StreamHandler[V]):
    kind: StreamKind

    def __init__(
        self,
        *,
        kind: Literal["bge_dense", "bge_sparse", "bge_colbert"],
        model_id: str,
        key_tag: bytes,
        codec: _Codec[V],
        extractor: Callable[[BgeRemote, int], V],
    ) -> None:
        super().__init__(model_id, codec)
        self.kind = kind
        self._key_tag = key_tag
        self._extractor = extractor

    def make_keys(self, texts: Sequence[str]) -> list[bytes]:
        model_bytes = self.model_id.encode("utf-8")
        return [
            _hash_chunks(self._key_tag, model_bytes, text.encode("utf-8"))
            for text in texts
        ]

    def extract(self, remote: ParsedEmbedResponseVariant, miss_row: int) -> V:
        remote_bge = _as_bge_variant(remote)
        return self._extractor(remote_bge, miss_row)


@dataclass(slots=True)
class _StreamSlot:
    handler: _StreamHandler[Any]
    keys: list[bytes]
    items: list[Any]


@dataclass(slots=True)
class EmbedCacheProgress:
    payload: EmbedRequestPayload
    texts: list[EmbedInput]
    misses: list[int]
    streams: list[_StreamSlot]


def _build_handlers(payload: EmbedRequestPayload) -> list[_StreamHandler[Any]]:
    handlers: list[_StreamHandler[Any]] = []
    dense_model_id = payload.get("dense_model_id")
    if dense_model_id is not None:
        handlers.append(
            _DenseHandler(
                model_id=dense_model_id,
                truncate_dim=payload.get("dense_truncate_dim"),
                dense_prompt=payload.get("dense_prompt") or "",
                dense_task=payload.get("dense_task"),
            )
        )
    sparse_model_id = payload.get("sparse_model_id")
    if sparse_model_id is not None:
        handlers.append(
            _SparseHandler(
                model_id=sparse_model_id,
                max_active_dims=payload.get("sparse_max_active_dims"),
                pruning_ratio=payload.get("sparse_pruning_ratio"),
                sparse_task=payload.get("sparse_task"),
            )
        )
    bge_model_id = payload.get("bge_model_id")
    if bge_model_id is not None:
        handlers.append(
            _BgeHandler(
                kind="bge_dense",
                model_id=bge_model_id,
                key_tag=b"bge_dense",
                codec=_F16VectorCodec(expected_dim=None),
                extractor=lambda remote, i: cast(
                    Float32Array,
                    np.asarray(remote.bgeM3.dense.vectors[i], dtype=np.float32),
                ),
            )
        )
        handlers.append(
            _BgeHandler(
                kind="bge_sparse",
                model_id=bge_model_id,
                key_tag=b"bge_sparse",
                codec=_SparseCodec(),
                extractor=lambda remote, i: remote.bgeM3.sparse.items[i],
            )
        )
        handlers.append(
            _BgeHandler(
                kind="bge_colbert",
                model_id=bge_model_id,
                key_tag=b"bge_colbert",
                codec=_F16MatrixCodec(),
                extractor=lambda remote, i: cast(
                    Float32Array,
                    np.asarray(remote.bgeM3.colbert[i], dtype=np.float32),
                ),
            )
        )
    return handlers


class EmbedCache:
    def __init__(self, env: lmdb.Environment) -> None:
        self._env = env

    @property
    def env(self) -> lmdb.Environment:
        return self._env

    @classmethod
    def open(
        cls,
        path: str | Path | None = None,
        *,
        map_size: int = _DEFAULT_MAP_SIZE,
    ) -> "EmbedCache":
        resolved = Path(path).expanduser() if path is not None else _DEFAULT_CACHE_PATH
        resolved.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(str(resolved), map_size=map_size)
        return cls(env)

    def close(self) -> None:
        self._env.close()

    def _load(self, keys: list[bytes]) -> list[bytes | None]:
        results: list[bytes | None] = [None for _ in keys]
        if not keys:
            return results
        try:
            with self._env.begin(write=False) as txn:
                for i, key in enumerate(keys):
                    raw = txn.get(key)
                    results[i] = bytes(raw) if raw is not None else None
        except Exception:
            return [None for _ in keys]
        return results

    def _save(self, items: list[tuple[bytes, bytes]]) -> None:
        if not items:
            return
        try:
            with self._env.begin(write=True) as txn:
                for key, value in items:
                    txn.put(key, value)
        except Exception:
            return

    def prepare(self, payload: EmbedRequestPayload) -> EmbedCacheProgress:
        payload = normalize_embed_payload(payload)
        texts = list(payload.get("texts", []))
        handlers = _build_handlers(payload)
        key_inputs = [
            item
            if isinstance(item, str)
            else "multimodal:" + json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in texts
        ]

        streams: list[_StreamSlot] = []
        miss_set: set[int] = set()

        for handler in handlers:
            keys = handler.make_keys(key_inputs)
            for idx, item in enumerate(texts):
                if isinstance(item, dict):
                    keys[idx] = _hash_chunks(b"multimodal", keys[idx])
            raw_values = self._load(keys)
            items: list[Any] = [handler.codec.decode(raw) for raw in raw_values]
            for idx, item in enumerate(items):
                if item is None:
                    miss_set.add(idx)
            streams.append(_StreamSlot(handler=handler, keys=keys, items=items))

        misses = sorted(miss_set)
        return EmbedCacheProgress(
            payload=payload, texts=texts, misses=misses, streams=streams
        )

    def merge_remote(
        self, progress: EmbedCacheProgress, remote: ParsedEmbedResponseVariant
    ) -> None:
        for slot in progress.streams:
            to_store: list[tuple[bytes, bytes]] = []
            for miss_row, idx in enumerate(progress.misses):
                value = slot.handler.extract(remote, miss_row)
                slot.items[idx] = value
                blob = slot.handler.codec.encode(value)
                if blob is not None:
                    to_store.append((slot.keys[idx], blob))
            self._save(to_store)

    def finalize(
        self,
        progress: EmbedCacheProgress,
        *,
        server_timings: dict[str, float] | None = None,
        client_timings: dict[str, float] | None = None,
    ) -> ParsedEmbedResponseVariant:
        dense_em: DenseEmbeddings | None = None
        sparse_em: SparseEmbeddings | None = None
        bge_dense_vec: Float32Array | None = None
        bge_sparse_items: list[SparseEmbedding] | None = None
        bge_colbert_items: list[Float32Array] | None = None
        bge_model_id: str | None = None

        for slot in progress.streams:
            if not all_present(slot.items):
                raise ValueError(
                    f"{slot.handler.kind} cache merge failed: unresolved items remain."
                )
            kind = slot.handler.kind
            if kind == "dense":
                vectors = roundtrip_float32_float16_float32(np.vstack(slot.items))
                dense_em = DenseEmbeddings(
                    model_id=slot.handler.model_id, vectors=vectors
                )
            elif kind == "sparse":
                sparse_em = SparseEmbeddings(
                    model_id=slot.handler.model_id, items=list(slot.items)
                )
            elif kind == "bge_dense":
                bge_dense_vec = roundtrip_float32_float16_float32(np.vstack(slot.items))
                bge_model_id = slot.handler.model_id
            elif kind == "bge_sparse":
                bge_sparse_items = list(slot.items)
                bge_model_id = slot.handler.model_id
            elif kind == "bge_colbert":
                bge_colbert_items = [
                    roundtrip_float32_float16_float32(item) for item in slot.items
                ]
                bge_model_id = slot.handler.model_id

        bge_em: BGEM3Embeddings | None = None
        if bge_model_id is not None:
            if (
                bge_dense_vec is None
                or bge_sparse_items is None
                or bge_colbert_items is None
            ):
                raise ValueError("BGE-M3 cache merge failed: unresolved buffers.")
            bge_em = BGEM3Embeddings(
                model_id=bge_model_id,
                dense=DenseEmbeddings(model_id=bge_model_id, vectors=bge_dense_vec),
                sparse=SparseEmbeddings(model_id=bge_model_id, items=bge_sparse_items),
                colbert=bge_colbert_items,
            )

        return assemble_parsed_response(
            texts_count=len(progress.texts),
            dense=dense_em,
            sparse=sparse_em,
            bge_m3=bge_em,
            server_timings=server_timings,
            client_timings={} if client_timings is None else client_timings,
        )

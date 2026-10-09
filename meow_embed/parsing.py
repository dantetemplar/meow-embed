from __future__ import annotations

import base64
from typing import TypeGuard, cast

import numpy as np

from meow_embed.types import (
    BGEM3Embeddings,
    BGEM3ResponseDict,
    ColbertItemResponseDict,
    DenseEmbeddings,
    DenseResponseDict,
    EmbedRequestPayload,
    EmbedResponseDict,
    Float32Array,
    ParsedEmbedResponse,
    SparseEmbedding,
    SparseEmbeddings,
    SparseItemResponseDict,
    SparseResponseDict,
    UInt32Array,
)


def decode_dense(dense: DenseResponseDict) -> DenseEmbeddings:
    raw = base64.b64decode(dense["data"])
    vectors = np.frombuffer(raw, dtype=np.float32)
    shape = tuple(dense["shape"])
    expected = shape[0] * shape[1]
    if vectors.size != expected:
        raise ValueError(
            f"Dense vector size mismatch: got {vectors.size}, expected {expected}."
        )
    return DenseEmbeddings(model_id=dense["model_id"], vectors=vectors.reshape(shape))


def decode_sparse(sparse: SparseResponseDict) -> SparseEmbeddings:
    def _decode_sparse_item(item: SparseItemResponseDict) -> SparseEmbedding:
        indices = cast(
            UInt32Array,
            np.frombuffer(base64.b64decode(item["indices"]), dtype=np.uint32),
        )
        values = cast(
            Float32Array,
            np.frombuffer(base64.b64decode(item["values"]), dtype=np.float32),
        )
        if indices.size != item["nnz"]:
            raise ValueError(
                f"Sparse indices size mismatch: got {indices.size}, expected {item['nnz']}."
            )
        if values.size != item["nnz"]:
            raise ValueError(
                f"Sparse values size mismatch: got {values.size}, expected {item['nnz']}."
            )
        return SparseEmbedding(dim=item["dim"], indices=indices, values=values)

    return SparseEmbeddings(
        model_id=sparse["model_id"],
        items=[_decode_sparse_item(item) for item in sparse["items"]],
    )


def decode_colbert_item(item: ColbertItemResponseDict) -> Float32Array:
    raw = base64.b64decode(item["data"])
    vectors = np.frombuffer(raw, dtype=np.float32)
    shape = tuple(item["shape"])
    expected = shape[0] * shape[1]
    if vectors.size != expected:
        raise ValueError(
            f"Colbert vector size mismatch: got {vectors.size}, expected {expected}."
        )
    return cast(Float32Array, vectors.reshape(shape))


def decode_bge_m3(bge_m3: BGEM3ResponseDict) -> BGEM3Embeddings:
    return BGEM3Embeddings(
        model_id=bge_m3["model_id"],
        dense=decode_dense(bge_m3["dense"]),
        sparse=decode_sparse(bge_m3["sparse"]),
        colbert=[decode_colbert_item(item) for item in bge_m3["colbert"]],
    )


def roundtrip_float32_float16_float32(vectors: np.ndarray) -> Float32Array:
    """Align with float16 LMDB storage so network and cache hits return the same vector values."""
    return cast(
        Float32Array,
        np.asarray(vectors, dtype=np.float32).astype(np.float16).astype(np.float32),
    )


def all_present[T](items: list[T | None] | None) -> TypeGuard[list[T]]:
    if items is None:
        return False
    for item in items:
        if item is None:
            return False
    return True


def assemble_parsed_response(
    *,
    texts_count: int,
    dense: DenseEmbeddings | None = None,
    sparse: SparseEmbeddings | None = None,
    bge_m3: BGEM3Embeddings | None = None,
    server_timings: dict[str, float] | None = None,
    client_timings: dict[str, float] | None = None,
) -> ParsedEmbedResponse:
    if dense is None and sparse is None and bge_m3 is None:
        raise ValueError("At least one of dense, sparse, or bgeM3 must be returned.")
    return ParsedEmbedResponse(
        texts_count=texts_count,
        dense=dense,
        sparse=sparse,
        bgeM3=bge_m3,
        server_timings=server_timings,
        client_timings={} if client_timings is None else client_timings,
    )


def decode_embed_response(
    raw: EmbedResponseDict,
    payload: EmbedRequestPayload,
    *,
    server_timings: dict[str, float] | None = None,
    client_timings: dict[str, float] | None = None,
) -> ParsedEmbedResponse:
    texts_len = len(payload.get("texts", []))
    texts_count = raw["texts_count"]

    dense: DenseEmbeddings | None = None
    sparse: SparseEmbeddings | None = None
    bge_m3: BGEM3Embeddings | None = None

    if payload.get("dense_model_id") is not None:
        if raw["dense"] is None:
            raise ValueError("Dense remote embed failed: unresolved vectors remain.")
        dense = decode_dense(raw["dense"])
        if len(dense.vectors) != texts_len:
            raise ValueError("Dense remote embed failed: mismatch in texts count.")

    if payload.get("sparse_model_id") is not None:
        if raw["sparse"] is None:
            raise ValueError("Sparse remote embed failed: unresolved items remain.")
        sparse = decode_sparse(raw["sparse"])
        if len(sparse.items) != texts_len:
            raise ValueError("Sparse remote embed failed: mismatch in texts count.")

    if payload.get("bge_model_id") is not None:
        if raw["bgeM3"] is None:
            raise ValueError("BGE-M3 remote embed failed: unresolved vectors remain.")
        bge_m3 = decode_bge_m3(raw["bgeM3"])
        if (
            len(bge_m3.dense.vectors) != texts_len
            or len(bge_m3.sparse.items) != texts_len
            or len(bge_m3.colbert) != texts_len
        ):
            raise ValueError("BGE-M3 remote embed failed: mismatch in texts count.")

    return assemble_parsed_response(
        texts_count=texts_count,
        dense=dense,
        sparse=sparse,
        bge_m3=bge_m3,
        server_timings=server_timings,
        client_timings={} if client_timings is None else client_timings,
    )

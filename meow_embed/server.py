import base64
import gzip
import inspect
import json
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
import torch
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.routing import APIRoute
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.sparse_encoder import SparseEncoder

from meow_embed._metadata import __description__, __version__

logger = logging.getLogger("uvicorn.error")


@dataclass(slots=True)
class ModelInstanceConfig:
    type: Literal["dense", "sparse", "reranker", "crossEncoder", "bgeM3"]
    model_id: str
    kwargs: dict[str, Any]


@dataclass(slots=True)
class ModelConfig:
    models: list[ModelInstanceConfig]


class EmbedRequest(BaseModel):
    dense_model_id: str | None = None
    dense_truncate_dim: int | None = None
    dense_prompt: str = Field(
        default="",
        description="Prefix prepended to each text for SentenceTransformer.encode (`prompt=`). Empty string means no prefix. Only used for dense models.",
    )
    dense_task: Literal["query", "document"] | None = Field(
        default=None,
        description='SentenceTransformer route: "query" or "document" (encode_query / encode_document); omit for encode().',
    )
    sparse_model_id: str | None = None
    sparse_max_active_dims: int | None = Field(default=None, gt=0)
    sparse_pruning_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    bge_model_id: str | None = None
    sparse_task: Literal["query", "document"] | None = Field(
        default=None,
        description='SparseEncoder route: "query" or "document" (encode_query / encode_document); omit for encode().',
    )

    texts: list[str]

    def __repr__(self) -> str:
        return f"EmbedRequest(dense_model_id={self.dense_model_id}, dense_truncate_dim={self.dense_truncate_dim}, dense_prompt={self.dense_prompt!r}, dense_task={self.dense_task}, sparse_model_id={self.sparse_model_id}, sparse_max_active_dims={self.sparse_max_active_dims}, sparse_pruning_ratio={self.sparse_pruning_ratio}, bge_model_id={self.bge_model_id}, sparse_task={self.sparse_task}, texts={len(self.texts)})"


class RerankRequest(BaseModel):
    reranker_model_id: str
    query: str | None = None
    queries: list[str] | None = None
    docs: list[str]


@dataclass
class TimingContext:
    request_start: float | None = None

    route_handler_start: float | None = None

    request_gunzip_start: float | None = None
    request_gunzip_end: float | None = None

    route_dense_embed_start: float | None = None
    route_dense_embed_end: float | None = None
    route_dense_payload_start: float | None = None
    route_dense_payload_end: float | None = None

    route_sparse_embed_start: float | None = None
    route_sparse_embed_end: float | None = None
    route_sparse_prune_start: float | None = None
    route_sparse_prune_end: float | None = None
    route_sparse_payload_start: float | None = None
    route_sparse_payload_end: float | None = None

    route_bge_embed_start: float | None = None
    route_bge_embed_end: float | None = None
    route_bge_payload_start: float | None = None
    route_bge_payload_end: float | None = None

    route_handler_end: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {"raw": {}, "pretty": ""}
        previous_time = None
        pretty = []
        for key, value in self.__dict__.items():
            if value is not None:
                payload["raw"][key] = value
                if previous_time is None:
                    previous_time = value
                difference = (value - previous_time) * 1000
                if difference > 0.5:
                    prty = f"{key}: +{difference:.0f}ms"
                else:
                    prty = f"{key}: ~{difference:.0f}ms"
                pretty.append(prty)
                previous_time = value
        payload["pretty"] = "\t".join(pretty)
        return payload


timing_context = ContextVar[TimingContext]("timing_context", default=None)  # pyright: ignore[reportArgumentType]


class EmbedResponse(BaseModel):
    class Dense(BaseModel):
        model_id: str
        shape: tuple[int, int]
        dtype: Literal["float32"]
        encoding: Literal["base64"]
        data: str

    class Sparse(BaseModel):
        class Item(BaseModel):
            dim: int
            nnz: int
            indices_dtype: Literal["uint32"]
            values_dtype: Literal["float32"]
            indices: str
            values: str

        model_id: str
        items: list[Item]
        encoding: Literal["base64"]

    class BGEM3(BaseModel):
        class ColbertItem(BaseModel):
            shape: tuple[int, int]
            dtype: Literal["float32"]
            encoding: Literal["base64"]
            data: str

        model_id: str
        dense: "EmbedResponse.Dense"
        sparse: "EmbedResponse.Sparse"
        colbert: list[ColbertItem]

    texts_count: int
    dense: Dense | None = None
    sparse: Sparse | None = None
    bgeM3: BGEM3 | None = None


class ModelsResponse(BaseModel):
    class Model(BaseModel):
        type: Literal["dense", "sparse", "reranker", "crossEncoder", "bgeM3"]
        id: str
        device: str
        dimensions: int | None
        dense_dimensions: int | None = None
        sparse_dimensions: int | None = None
        colbert_dimensions: int | None = None
        batch_size: int | None

    models: list[Model]


class RerankResponse(BaseModel):
    model_id: str
    shape: tuple[int, int]
    scores: list[list[float]]


def dense_dimensions(model: Any) -> int | None:
    return model.get_sentence_embedding_dimension()


def sparse_dimensions(model: Any) -> int | None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size
    return None


def bge_dense_dimensions(model: Any) -> int | None:
    for attr in ("output_dim", "dense_dim", "embedding_dim"):
        value = getattr(model, attr, None)
        if isinstance(value, int):
            return value
    nested_model = getattr(model, "model", None)
    nested_config = getattr(nested_model, "config", None)
    hidden_size = getattr(nested_config, "hidden_size", None)
    if isinstance(hidden_size, int):
        return hidden_size
    return None


def bge_sparse_dimensions(model: Any) -> int | None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size
    return None


def bge_colbert_dimensions(model: Any) -> int | None:
    for attr in ("colbert_dim",):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    # In default BGE-M3 config colbert dim equals hidden size (1024).
    return bge_dense_dimensions(model)


def default_encode_batch_size(model: Any) -> int:
    default = inspect.signature(model.encode).parameters["batch_size"].default
    if not isinstance(default, int):
        raise RuntimeError(f"Unexpected encode batch_size default: {default!r}")
    return default


def default_reranker_batch_size(model: Any) -> int | None:
    # FlagReranker / CrossEncoder store inference batch size on the instance (__init__);
    # compute_score / predict only take sentence_pairs and **kwargs.
    batch_size = getattr(model, "batch_size", None)
    return batch_size if isinstance(batch_size, int) else None


def rerank_scores(model: Any, pairs: list[list[str]]) -> Any:
    if isinstance(model, CrossEncoder):
        return model.predict(pairs, convert_to_numpy=True)
    return model.compute_score(pairs, normalize=False)


def default_bge_batch_size(model: Any) -> int | None:
    batch_size = getattr(model, "batch_size", None)
    if isinstance(batch_size, int):
        return batch_size
    default = inspect.signature(model.encode).parameters["batch_size"].default
    return default if isinstance(default, int) else None


def model_device(model: Any) -> str:
    """Human-readable device(s) for sentence-transformers or FlagEmbedding wrappers."""
    target_devices = getattr(model, "target_devices", None)
    if target_devices:
        return ", ".join(str(d) for d in target_devices)
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    inner = getattr(model, "model", None)
    if inner is not None:
        try:
            return str(next(inner.parameters()).device)
        except (StopIteration, TypeError):
            pass
    return "unknown"


def encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def prune_sparse_embedding(embeddings: Any, pruning_ratio: float) -> Any:
    if embeddings.is_sparse:
        dense = embeddings.to_dense()
    else:
        dense = embeddings

    max_values = dense.max(dim=1, keepdim=True).values
    threshold = max_values * pruning_ratio
    pruned_dense = dense * (dense > threshold)
    return pruned_dense.to_sparse()


def serialize_sparse_embeddings(vectors: Any) -> list[EmbedResponse.Sparse.Item]:
    if not isinstance(vectors, torch.Tensor):
        raise RuntimeError(
            f"Unsupported sparse embeddings type: {type(vectors)!r}. Expected torch.Tensor."
        )

    tensor = vectors
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if not tensor.is_sparse:
        tensor = tensor.to_sparse()
    tensor = tensor.coalesce()

    indices = tensor.indices().cpu().numpy()
    values = tensor.values().cpu().numpy().astype(np.float32, copy=False)
    batch = int(tensor.shape[0])
    dim = int(tensor.shape[1]) if tensor.ndim > 1 else 0

    rows_indices: list[list[int]] = [[] for _ in range(batch)]
    rows_values: list[list[float]] = [[] for _ in range(batch)]
    for col in range(indices.shape[1]):
        row_id = int(indices[0, col])
        feature_id = int(indices[1, col])
        rows_indices[row_id].append(feature_id)
        rows_values[row_id].append(float(values[col]))

    rows = []
    for row_id in range(batch):
        row_indices = np.asarray(rows_indices[row_id], dtype=np.uint32)
        row_values = np.asarray(rows_values[row_id], dtype=np.float32)
        rows.append(
            EmbedResponse.Sparse.Item(
                dim=dim,
                nnz=int(row_indices.shape[0]),
                indices_dtype="uint32",
                values_dtype="float32",
                indices=encode_bytes(row_indices.tobytes()),
                values=encode_bytes(row_values.tobytes()),
            )
        )
    return rows


def serialize_bge_lexical_weights(
    lexical_weights: list[dict[int, float]] | list[dict[str, float]],
    dim: int,
) -> list[EmbedResponse.Sparse.Item]:
    rows = []
    for item in lexical_weights:
        sorted_items = sorted(item.items(), key=lambda kv: int(kv[0]))
        row_indices = np.asarray(
            [int(index) for index, _ in sorted_items], dtype=np.uint32
        )
        row_values = np.asarray(
            [float(value) for _, value in sorted_items], dtype=np.float32
        )
        rows.append(
            EmbedResponse.Sparse.Item(
                dim=dim,
                nnz=int(row_indices.shape[0]),
                indices_dtype="uint32",
                values_dtype="float32",
                indices=encode_bytes(row_indices.tobytes()),
                values=encode_bytes(row_values.tobytes()),
            )
        )
    return rows


def serialize_bge_colbert_embeddings(
    vectors: Any,
) -> list[EmbedResponse.BGEM3.ColbertItem]:
    if isinstance(vectors, np.ndarray):
        if vectors.ndim == 3:
            array_items = [vectors[i] for i in range(vectors.shape[0])]
        elif vectors.ndim == 2:
            array_items = [vectors]
        else:
            raise RuntimeError(
                f"Unsupported bge colbert embeddings ndim: {vectors.ndim}"
            )
    elif isinstance(vectors, list):
        array_items = vectors
    else:
        raise RuntimeError(
            f"Unsupported bge colbert embeddings type: {type(vectors)!r}"
        )

    items = []
    for array_item in array_items:
        array = np.asarray(array_item, dtype=np.float32)
        if array.ndim != 2:
            raise RuntimeError(f"Unsupported bge colbert item ndim: {array.ndim}")
        items.append(
            EmbedResponse.BGEM3.ColbertItem(
                shape=(int(array.shape[0]), int(array.shape[1])),
                dtype="float32",
                encoding="base64",
                data=encode_bytes(array.tobytes()),
            )
        )
    return items


def build_app(config: ModelConfig) -> FastAPI:
    class UngzipRequest(Request):
        async def body(self) -> bytes:
            if not hasattr(self, "_body"):
                body = await super().body()
                if "gzip" in self.headers.getlist("Content-Encoding"):
                    timing_context.get().request_gunzip_start = time.monotonic()
                    body = gzip.decompress(body)
                    timing_context.get().request_gunzip_end = time.monotonic()
                self._body = body
            return self._body

    class CustomRoute(APIRoute):
        def get_route_handler(self) -> Callable:
            original_route_handler = super().get_route_handler()

            async def custom_route_handler(request: Request) -> Response:
                context_token = timing_context.set(
                    TimingContext(request_start=time.monotonic())
                )

                try:
                    request = UngzipRequest(request.scope, request.receive)
                    timing_context.get().route_handler_start = time.monotonic()
                    response = await original_route_handler(request)
                    timing_context.get().route_handler_end = time.monotonic()
                    timing_dict = timing_context.get().as_dict()
                    response.headers["X-Timing-Context"] = timing_dict["pretty"]
                    response.headers["X-Timing-Context-Raw"] = json.dumps(
                        timing_dict["raw"]
                    )
                    return response

                finally:
                    ctx = timing_context.get()
                    if ctx is not None:
                        logger.info(
                            "Timing context:\n%s",
                            ctx.as_dict()["pretty"].replace("\t", "\n"),
                        )
                    else:
                        logger.info("No timing context")
                    timing_context.reset(context_token)

            return custom_route_handler

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.dense_models = {}
        app.state.sparse_models = {}
        app.state.reranker_models = {}
        app.state.cross_encoder_models = {}
        app.state.bge_m3_models = {}

        startup_started_at = time.monotonic()
        logger.info("Loading %d model(s)...", len(config.models))
        for model in config.models:
            model_load_started_at = time.monotonic()
            logger.info("Loading %s model: %s", model.type, model.model_id)
            if model.type == "dense":
                app.state.dense_models[model.model_id] = SentenceTransformer(
                    model.model_id, **model.kwargs
                )
            elif model.type == "sparse":
                app.state.sparse_models[model.model_id] = SparseEncoder(
                    model.model_id, **model.kwargs
                )
            elif model.type == "reranker":
                app.state.reranker_models[model.model_id] = FlagReranker(
                    model.model_id, **model.kwargs
                )
            elif model.type == "crossEncoder":
                app.state.cross_encoder_models[model.model_id] = CrossEncoder(
                    model.model_id, **model.kwargs
                )
            else:
                app.state.bge_m3_models[model.model_id] = BGEM3FlagModel(
                    model.model_id, **model.kwargs
                )
            logger.info(
                "Loaded %s model: %s in %.1fs",
                model.type,
                model.model_id,
                time.monotonic() - model_load_started_at,
            )
        logger.info(
            "Startup complete. Loaded %d model(s) in %.1fs.",
            len(config.models),
            time.monotonic() - startup_started_at,
        )

        yield

        app.state.dense_models = {}
        app.state.sparse_models = {}
        app.state.reranker_models = {}
        app.state.cross_encoder_models = {}
        app.state.bge_m3_models = {}

    app = FastAPI(
        title="meow-embed",
        description=__description__,
        version=__version__,
        lifespan=lifespan,
    )
    app.router.route_class = CustomRoute

    @app.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/models")
    async def models() -> ModelsResponse:
        items = []
        for model_id, model in app.state.dense_models.items():
            items.append(
                ModelsResponse.Model(
                    type="dense",
                    id=model_id,
                    device=model_device(model),
                    dimensions=dense_dimensions(model),
                    dense_dimensions=dense_dimensions(model),
                    batch_size=default_encode_batch_size(model),
                )
            )
        for model_id, model in app.state.sparse_models.items():
            items.append(
                ModelsResponse.Model(
                    type="sparse",
                    id=model_id,
                    device=model_device(model),
                    dimensions=sparse_dimensions(model),
                    sparse_dimensions=sparse_dimensions(model),
                    batch_size=default_encode_batch_size(model),
                )
            )
        for model_id, model in app.state.reranker_models.items():
            items.append(
                ModelsResponse.Model(
                    type="reranker",
                    id=model_id,
                    device=model_device(model),
                    dimensions=None,
                    batch_size=default_reranker_batch_size(model),
                )
            )
        for model_id, model in app.state.cross_encoder_models.items():
            items.append(
                ModelsResponse.Model(
                    type="reranker",
                    id=model_id,
                    device=model_device(model),
                    dimensions=None,
                    batch_size=default_reranker_batch_size(model),
                )
            )
        for model_id, model in app.state.bge_m3_models.items():
            items.append(
                ModelsResponse.Model(
                    type="bgeM3",
                    id=model_id,
                    device=model_device(model),
                    dimensions=None,
                    dense_dimensions=bge_dense_dimensions(model),
                    sparse_dimensions=bge_sparse_dimensions(model),
                    colbert_dimensions=bge_colbert_dimensions(model),
                    batch_size=default_bge_batch_size(model),
                )
            )
        return ModelsResponse(models=items)

    @app.post("/embed")
    def embed(
        embed_request: Annotated[
            EmbedRequest,
            Body(
                example={
                    "texts": ["Привет, мир!"],
                    "dense_model_id": "sergeyzh/BERTA",
                    "dense_prompt": None,
                    "dense_task": "query",
                    "dense_truncate_dim": None,
                    "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
                    "sparse_task": "query",
                    "bge_model_id": "BAAI/bge-m3",
                }
            ),
        ],
    ) -> EmbedResponse:
        logger.info("Embed request: %s", repr(embed_request))
        if (
            embed_request.dense_model_id is None
            and embed_request.sparse_model_id is None
            and embed_request.bge_model_id is None
        ):
            raise HTTPException(
                status_code=400,
                detail="Provide at least one of dense_model_id, sparse_model_id, or bge_model_id.",
            )

        payload: EmbedResponse = EmbedResponse(texts_count=len(embed_request.texts))

        if embed_request.dense_model_id is not None:
            dense_model: SentenceTransformer | None = app.state.dense_models.get(
                embed_request.dense_model_id
            )
            if dense_model is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dense model not loaded: {embed_request.dense_model_id}",
                )
            timing_context.get().route_dense_embed_start = time.monotonic()
            if embed_request.dense_task == "query":
                dense_vectors = dense_model.encode_query(
                    embed_request.texts,
                    truncate_dim=embed_request.dense_truncate_dim,
                    prompt=embed_request.dense_prompt,
                    convert_to_numpy=True,
                )
            elif embed_request.dense_task == "document":
                dense_vectors = dense_model.encode_document(
                    embed_request.texts,
                    truncate_dim=embed_request.dense_truncate_dim,
                    prompt=embed_request.dense_prompt,
                    convert_to_numpy=True,
                )
            else:
                dense_vectors = dense_model.encode(
                    embed_request.texts,
                    truncate_dim=embed_request.dense_truncate_dim,
                    prompt=embed_request.dense_prompt,
                    convert_to_numpy=True,
                )
            timing_context.get().route_dense_embed_end = time.monotonic()
            dense_vectors = np.asarray(dense_vectors, dtype=np.float32)
            if dense_vectors.ndim == 1:
                dense_vectors = np.expand_dims(dense_vectors, axis=0)
            timing_context.get().route_dense_payload_start = time.monotonic()
            payload.dense = EmbedResponse.Dense(
                model_id=embed_request.dense_model_id,
                shape=(int(dense_vectors.shape[0]), int(dense_vectors.shape[1])),
                dtype="float32",
                encoding="base64",
                data=encode_bytes(dense_vectors.tobytes()),
            )
            timing_context.get().route_dense_payload_end = time.monotonic()

        if embed_request.sparse_model_id is not None:
            sparse_model: SparseEncoder | None = app.state.sparse_models.get(
                embed_request.sparse_model_id
            )
            if sparse_model is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sparse model not loaded: {embed_request.sparse_model_id}",
                )
            timing_context.get().route_sparse_embed_start = time.monotonic()
            if embed_request.sparse_task == "query":
                sparse_vectors = sparse_model.encode_query(
                    embed_request.texts,
                    max_active_dims=embed_request.sparse_max_active_dims,
                    convert_to_tensor=True,
                )
            elif embed_request.sparse_task == "document":
                sparse_vectors = sparse_model.encode_document(
                    embed_request.texts,
                    max_active_dims=embed_request.sparse_max_active_dims,
                    convert_to_tensor=True,
                )
            else:
                sparse_vectors = sparse_model.encode(
                    embed_request.texts,
                    max_active_dims=embed_request.sparse_max_active_dims,
                    convert_to_tensor=True,
                )
            timing_context.get().route_sparse_embed_end = time.monotonic()
            if embed_request.sparse_pruning_ratio:
                timing_context.get().route_sparse_prune_start = time.monotonic()
                sparse_vectors = prune_sparse_embedding(
                    sparse_vectors,
                    pruning_ratio=embed_request.sparse_pruning_ratio,
                )
                timing_context.get().route_sparse_prune_end = time.monotonic()
            timing_context.get().route_sparse_payload_start = time.monotonic()
            payload.sparse = EmbedResponse.Sparse(
                model_id=embed_request.sparse_model_id,
                items=serialize_sparse_embeddings(sparse_vectors),
                encoding="base64",
            )
            timing_context.get().route_sparse_payload_end = time.monotonic()

        if embed_request.bge_model_id is not None:
            bge_model: BGEM3FlagModel | None = app.state.bge_m3_models.get(
                embed_request.bge_model_id
            )
            if bge_model is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"BGE model not loaded: {embed_request.bge_model_id}",
                )
            timing_context.get().route_bge_embed_start = time.monotonic()
            bge_vectors = bge_model.encode(
                embed_request.texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
            timing_context.get().route_bge_embed_end = time.monotonic()

            dense_vectors = np.asarray(bge_vectors["dense_vecs"], dtype=np.float32)
            if dense_vectors.ndim == 1:
                dense_vectors = np.expand_dims(dense_vectors, axis=0)
            sparse_items = serialize_bge_lexical_weights(
                bge_vectors["lexical_weights"],
                dim=int(bge_sparse_dimensions(bge_model) or 0),
            )
            colbert_items = serialize_bge_colbert_embeddings(
                bge_vectors["colbert_vecs"]
            )

            timing_context.get().route_bge_payload_start = time.monotonic()
            payload.bgeM3 = EmbedResponse.BGEM3(
                model_id=embed_request.bge_model_id,
                dense=EmbedResponse.Dense(
                    model_id=embed_request.bge_model_id,
                    shape=(int(dense_vectors.shape[0]), int(dense_vectors.shape[1])),
                    dtype="float32",
                    encoding="base64",
                    data=encode_bytes(dense_vectors.tobytes()),
                ),
                sparse=EmbedResponse.Sparse(
                    model_id=embed_request.bge_model_id,
                    items=sparse_items,
                    encoding="base64",
                ),
                colbert=colbert_items,
            )
            timing_context.get().route_bge_payload_end = time.monotonic()
        return payload

    @app.post("/rerank")
    def rerank(
        rerank_request: Annotated[
            RerankRequest,
            Body(
                example={
                    "reranker_model_id": "BAAI/bge-reranker-v2-m3",
                    "query": "what is panda?",
                    "docs": [
                        "hi",
                        "The giant panda is a bear species endemic to China.",
                    ],
                }
            ),
        ],
    ) -> RerankResponse:
        if rerank_request.query is None and rerank_request.queries is None:
            raise HTTPException(
                status_code=400, detail="Provide either query or queries."
            )
        if rerank_request.query is not None and rerank_request.queries is not None:
            raise HTTPException(
                status_code=400, detail="Provide either query or queries, not both."
            )
        if len(rerank_request.docs) == 0:
            raise HTTPException(status_code=400, detail="Provide non-empty docs.")

        model_id = rerank_request.reranker_model_id
        reranker_model: FlagReranker | CrossEncoder | None = (
            app.state.reranker_models.get(model_id)
            or app.state.cross_encoder_models.get(model_id)
        )
        if reranker_model is None:
            raise HTTPException(
                status_code=400, detail=f"Reranker model not loaded: {model_id}"
            )

        queries = (
            [rerank_request.query]
            if rerank_request.query is not None
            else rerank_request.queries
        )
        if queries is None or len(queries) == 0:
            raise HTTPException(
                status_code=400, detail="Provide non-empty query or queries."
            )

        pairs = [[query, doc] for query in queries for doc in rerank_request.docs]
        raw_scores = rerank_scores(reranker_model, pairs)
        scores_array = np.asarray(raw_scores, dtype=np.float32).reshape(
            len(queries), len(rerank_request.docs)
        )
        scores = [[float(value) for value in row] for row in scores_array.tolist()]
        return RerankResponse(
            model_id=model_id,
            shape=(len(queries), len(rerank_request.docs)),
            scores=scores,
        )

    return app

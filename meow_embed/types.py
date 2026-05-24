from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, Required, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
from typing_extensions import ReadOnly

type Float32Array = npt.NDArray[np.float32]
type Float16Array = npt.NDArray[np.float16]
type UInt32Array = npt.NDArray[np.uint32]
type RerankActivationFn = Literal["default", "identity", "sigmoid"]


class EmbedRequestCommonDict(TypedDict):
    texts: ReadOnly[Sequence[str]]
    dense_truncate_dim: ReadOnly[NotRequired[int | None]]
    dense_prompt: ReadOnly[NotRequired[str]]
    dense_task: ReadOnly[NotRequired[Literal["query", "document"] | None]]
    sparse_max_active_dims: ReadOnly[NotRequired[int | None]]
    sparse_pruning_ratio: ReadOnly[NotRequired[float | None]]
    sparse_task: ReadOnly[NotRequired[Literal["query", "document"] | None]]


class DenseEmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[NotRequired[None]]


class SparseEmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[NotRequired[None]]


class BGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[Required[str]]


class DenseSparseEmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[NotRequired[None]]


class DenseBGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[Required[str]]


class SparseBGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[Required[str]]


class DenseSparseBGEM3EmbedRequestDict(EmbedRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[Required[str]]


EmbedRequestPayload = (
    DenseEmbedRequestDict
    | SparseEmbedRequestDict
    | BGEM3EmbedRequestDict
    | DenseSparseEmbedRequestDict
    | DenseBGEM3EmbedRequestDict
    | SparseBGEM3EmbedRequestDict
    | DenseSparseBGEM3EmbedRequestDict
)


class EmbedOneRequestCommonDict(TypedDict):
    text: ReadOnly[Required[str]]

    dense_truncate_dim: ReadOnly[NotRequired[int | None]]
    dense_prompt: ReadOnly[NotRequired[str]]
    dense_task: ReadOnly[NotRequired[Literal["query", "document"] | None]]

    sparse_max_active_dims: ReadOnly[NotRequired[int | None]]
    sparse_pruning_ratio: ReadOnly[NotRequired[float | None]]
    sparse_task: ReadOnly[NotRequired[Literal["query", "document"] | None]]


class DenseEmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[NotRequired[None]]


class SparseEmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[NotRequired[None]]


class BGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[Required[str]]


class DenseSparseEmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[NotRequired[None]]


class DenseBGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[NotRequired[None]]
    bge_model_id: ReadOnly[Required[str]]


class SparseBGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[NotRequired[None]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[Required[str]]


class DenseSparseBGEM3EmbedOneRequestDict(EmbedOneRequestCommonDict):
    dense_model_id: ReadOnly[Required[str]]
    sparse_model_id: ReadOnly[Required[str]]
    bge_model_id: ReadOnly[Required[str]]


EmbedOneRequestPayload = (
    DenseEmbedOneRequestDict
    | SparseEmbedOneRequestDict
    | BGEM3EmbedOneRequestDict
    | DenseSparseEmbedOneRequestDict
    | DenseBGEM3EmbedOneRequestDict
    | SparseBGEM3EmbedOneRequestDict
    | DenseSparseBGEM3EmbedOneRequestDict
)


class RerankRequestDict(TypedDict):
    reranker_model_id: ReadOnly[str]
    docs: ReadOnly[Sequence[str]]
    query: ReadOnly[NotRequired[str | None]]
    queries: ReadOnly[NotRequired[Sequence[str] | None]]
    prompt: ReadOnly[NotRequired[str | None]]
    prompt_name: ReadOnly[NotRequired[str | None]]
    activation_fn: ReadOnly[NotRequired[RerankActivationFn | None]]


class DenseResponseDict(TypedDict):
    model_id: str
    shape: tuple[int, int]
    dtype: Literal["float32"]
    encoding: Literal["base64"]
    data: str


class SparseItemResponseDict(TypedDict):
    dim: int
    nnz: int
    indices_dtype: Literal["uint32"]
    values_dtype: Literal["float32"]
    indices: str
    values: str


class SparseResponseDict(TypedDict):
    model_id: str
    items: list[SparseItemResponseDict]
    encoding: Literal["base64"]


class ColbertItemResponseDict(TypedDict):
    shape: tuple[int, int]
    dtype: Literal["float32"]
    encoding: Literal["base64"]
    data: str


class BGEM3ResponseDict(TypedDict):
    model_id: str
    dense: DenseResponseDict
    sparse: SparseResponseDict
    colbert: list[ColbertItemResponseDict]


class EmbedResponseDict(TypedDict):
    texts_count: int
    dense: DenseResponseDict | None
    sparse: SparseResponseDict | None
    bgeM3: BGEM3ResponseDict | None


class ModelInfoDict(TypedDict):
    type: Literal["dense", "sparse", "reranker", "bgeM3"]
    id: str
    device: str
    dimensions: int | None
    dense_dimensions: NotRequired[int | None]
    sparse_dimensions: NotRequired[int | None]
    colbert_dimensions: NotRequired[int | None]
    batch_size: int | None


class ModelsResponseDict(TypedDict):
    models: list[ModelInfoDict]


class RerankResponseDict(TypedDict):
    model_id: str
    shape: tuple[int, int]
    scores: list[list[float]]


@dataclass(slots=True)
class DenseEmbeddings:
    model_id: str
    vectors: Float32Array


@dataclass(slots=True)
class SparseEmbedding:
    dim: int
    indices: UInt32Array
    values: Float32Array


@dataclass(slots=True)
class SparseEmbeddings:
    model_id: str
    items: list[SparseEmbedding]


@dataclass(slots=True)
class BGEM3Embeddings:
    model_id: str
    dense: DenseEmbeddings
    sparse: SparseEmbeddings
    colbert: list[Float32Array]


class _PrettyTimingsMixin:
    server_timings: dict[str, float] | None
    client_timings: dict[str, float]

    @staticmethod
    def _format_timing_map(
        title: str,
        timing_map: dict[str, float] | None,
        *,
        max_key_len: int,
        indent: str = "  ",
        skip_total: bool = True,
    ) -> list[str]:
        lines = [f"{title}:"]
        if not timing_map:
            lines.append("  (none)")
            return lines
        for key, value in timing_map.items():
            if skip_total and key.endswith("_total_ms"):
                continue
            label = key.removesuffix("_ms")
            marker = "+" if value > 0.5 else "~"
            lines.append(f"{indent}{label:<{max_key_len}} : {marker}{value:.0f}ms")
        return lines

    @staticmethod
    def _server_total_ms(server_timings: dict[str, float] | None) -> float | None:
        if not server_timings:
            return None
        values = list(server_timings.values())
        if not values:
            return None
        return max(0.0, values[-1] - values[0])

    @staticmethod
    def _format_server_timings(
        server_timings: dict[str, float] | None, *, max_key_len: int, indent: str
    ) -> list[str]:
        if not server_timings:
            return [f"{indent}(none)"]
        lines: list[str] = []
        items = list(server_timings.items())
        previous: float | None = None
        depth = 0
        i = 0
        while i < len(items):
            key, ts = items[i]
            label = key.removesuffix("_ms")
            line_indent = indent + (" " * depth)
            label_width = max(1, max_key_len - depth)
            total_suffix = ""
            if label.endswith("_start"):
                end_label = f"{label[:-6]}_end"
                end_ts = next(
                    (
                        next_ts
                        for next_key, next_ts in items[i + 1 :]
                        if next_key.removesuffix("_ms") == end_label
                    ),
                    None,
                )
                if end_ts is not None:
                    total_suffix = f" (total {max(0.0, end_ts - ts):.0f} ms)"
            if (
                label.endswith("_start")
                and i + 1 < len(items)
                and items[i + 1][0].removesuffix("_ms") == f"{label[:-6]}_end"
            ):
                end_ts = items[i + 1][1]
                delta = max(0.0, end_ts - ts)
                marker = "+" if delta > 0.5 else "~"
                lines.append(
                    f"{line_indent}{label[:-6]:<{label_width}} : {marker}{delta:.0f}ms"
                )
                previous = end_ts
                i += 2
                continue

            if label.endswith("_end") and depth > 0:
                depth -= 1
                line_indent = indent + (" " * depth)
                label_width = max(1, max_key_len - depth)
            delta = 0.0 if previous is None else max(0.0, ts - previous)
            marker = "+" if delta > 0.5 else "~"
            lines.append(
                f"{line_indent}{label:<{label_width}} : {marker}{delta:.0f}ms{total_suffix}"
            )
            if label.endswith("_start"):
                depth += 1
            previous = ts
            i += 1
        return lines

    def pretty_timings(self) -> str:
        max_key_len = 0
        if self.server_timings:
            max_key_len = max(
                max_key_len, *(len(k.removesuffix("_ms")) for k in self.server_timings)
            )
        if self.client_timings:
            max_key_len = max(
                max_key_len, *(len(k.removesuffix("_ms")) for k in self.client_timings)
            )
        max_key_len = max(max_key_len, 8)
        lines: list[str] = []
        total_ms = next(
            (
                value
                for key, value in self.client_timings.items()
                if key.endswith("_total_ms")
            ),
            None,
        )
        first_client_ts = next(iter(self.client_timings.values()), 0.0)
        client_title = "timings"
        if total_ms is not None:
            client_title = f"timings (total {max(0.0, (total_ms - first_client_ts) * 1000.0):.0f} ms)"
        lines.append(f"{client_title}:")
        client_items = {
            key: value
            for key, value in self.client_timings.items()
            if not key.endswith("_total_ms")
        }
        remote_request_ms = self.client_timings.get("remote_request_ms")
        server_total_ms = self._server_total_ms(self.server_timings)
        previous_client_ts: float | None = None
        for key, value in client_items.items():
            stage_ms = (
                0.0
                if previous_client_ts is None
                else max(0.0, (value - previous_client_ts) * 1000.0)
            )
            if key == "remote_request_ms":
                remote_total_ms = stage_ms
                if server_total_ms is not None:
                    overhead_ms = max(0.0, remote_total_ms - server_total_ms)
                    remote_title = f"  remote_request (total {server_total_ms:.0f} + {overhead_ms:.0f} overhead ms)"
                else:
                    remote_title = f"  remote_request (total {remote_total_ms:.0f} ms)"
                lines.append(f"{remote_title}:")
                lines.extend(
                    self._format_server_timings(
                        self.server_timings, max_key_len=max_key_len, indent="   "
                    )
                )
                previous_client_ts = value
                continue
            label = key.removesuffix("_ms")
            marker = "+" if stage_ms > 0.5 else "~"
            lines.append(f"  {label:<{max_key_len}} : {marker}{stage_ms:.0f}ms")
            previous_client_ts = value
        if not client_items:
            lines.append("  (none)")
        elif remote_request_ms is None and self.server_timings:
            lines.append("  server_timings:")
            lines.extend(
                self._format_server_timings(
                    self.server_timings, max_key_len=max_key_len, indent="   "
                )
            )
        return "\n".join(lines)


@dataclass(slots=True)
class ParsedEmbedResponseCommon(_PrettyTimingsMixin):
    texts_count: int
    server_timings: dict[str, float] | None
    client_timings: dict[str, float]


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseDense(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseSparse(ParsedEmbedResponseCommon):
    sparse: SparseEmbeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseBGEM3(ParsedEmbedResponseCommon):
    bgeM3: BGEM3Embeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseDenseSparse(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    sparse: SparseEmbeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseDenseBGEM3(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    bgeM3: BGEM3Embeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseSparseBGEM3(ParsedEmbedResponseCommon):
    sparse: SparseEmbeddings
    bgeM3: BGEM3Embeddings


@dataclass(slots=True, kw_only=True)
class ParsedEmbedResponseDenseSparseBGEM3(ParsedEmbedResponseCommon):
    dense: DenseEmbeddings
    sparse: SparseEmbeddings
    bgeM3: BGEM3Embeddings


ParsedEmbedResponseVariant = (
    ParsedEmbedResponseDense
    | ParsedEmbedResponseSparse
    | ParsedEmbedResponseBGEM3
    | ParsedEmbedResponseDenseSparse
    | ParsedEmbedResponseDenseBGEM3
    | ParsedEmbedResponseSparseBGEM3
    | ParsedEmbedResponseDenseSparseBGEM3
)


@dataclass(slots=True)
class DenseEmbeddingVector:
    model_id: str
    vector: Float32Array


@dataclass(slots=True)
class SparseEmbeddingsOne:
    model_id: str
    item: SparseEmbedding


@dataclass(slots=True)
class BGEM3EmbeddingOne:
    model_id: str
    dense: DenseEmbeddingVector
    sparse: SparseEmbeddingsOne
    colbert: Float32Array


@dataclass(slots=True)
class ParsedEmbedOneCommon(_PrettyTimingsMixin):
    server_timings: dict[str, float] | None
    client_timings: dict[str, float]


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneDense(ParsedEmbedOneCommon):
    dense: DenseEmbeddingVector


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneSparse(ParsedEmbedOneCommon):
    sparse: SparseEmbeddingsOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneBGEM3(ParsedEmbedOneCommon):
    bgeM3: BGEM3EmbeddingOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneDenseSparse(ParsedEmbedOneCommon):
    dense: DenseEmbeddingVector
    sparse: SparseEmbeddingsOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneDenseBGEM3(ParsedEmbedOneCommon):
    dense: DenseEmbeddingVector
    bgeM3: BGEM3EmbeddingOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneSparseBGEM3(ParsedEmbedOneCommon):
    sparse: SparseEmbeddingsOne
    bgeM3: BGEM3EmbeddingOne


@dataclass(slots=True, kw_only=True)
class ParsedEmbedOneDenseSparseBGEM3(ParsedEmbedOneCommon):
    dense: DenseEmbeddingVector
    sparse: SparseEmbeddingsOne
    bgeM3: BGEM3EmbeddingOne


ParsedEmbedOneVariant = (
    ParsedEmbedOneDense
    | ParsedEmbedOneSparse
    | ParsedEmbedOneBGEM3
    | ParsedEmbedOneDenseSparse
    | ParsedEmbedOneDenseBGEM3
    | ParsedEmbedOneSparseBGEM3
    | ParsedEmbedOneDenseSparseBGEM3
)


@dataclass(slots=True)
class ParsedRerankResponse:
    model_id: str
    shape: tuple[int, int]
    scores: list[list[float]]

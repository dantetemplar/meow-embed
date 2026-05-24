# meow-embed
> FastAPI embedding server and Python client for dense and sparse text embeddings

## What is it?

`meow-embed` is an HTTP API for text embeddings.

It can return **dense + sparse + BGE-M3 (dense, sparse, colbert) in one request**, with efficient transport:
- vectors are encoded as `base64`
- request body can be `gzip`-compressed
- response body can be `gzip`-compressed

## Why?

There is already [Infinity](https://github.com/michaelfeil/infinity), but this project exists because:
- sparse embeddings are needed
- BGE-M3 colbert and sparse embeddings are needed
- gzip request handling is needed end-to-end
- one round-trip for dense + sparse is needed
- patching upstream behavior was too complex
- newer `sentence-transformers` versions are required
- caching on client side is needed

## How to use

### 1) Start server

Install with server extras from git:

```bash
uv tool install "git+https://github.com/dantetemplar/meow-embed.git[server]"

# if you don't have git installed you can use the zip file
uv tool install "https://github.com/dantetemplar/meow-embed/archive/refs/heads/main.zip#egg=meow-embed[server]"
```

Run server and load models:

```bash
meow-embed \
  --SentenceTransformer "sergeyzh/BERTA" \
  --SparseEncoder "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1" \
  --BGEM3FlagModel "BAAI/bge-m3" '{"use_fp16": true}' \
  --FlagReranker "BAAI/bge-reranker-v2-m3" '{"use_fp16": true}' \
  --CrossEncoder "jinaai/jina-reranker-v2-base-multilingual" '{"model_kwargs": {"torch_dtype": "auto"}, "trust_remote_code": true}' \
  --host 0.0.0.0
```

...wait until server is ready (Uvicorn running on http://0.0.0.0:8067)...

`meow-embed` handles only model flags (`--SentenceTransformer`, `--SparseEncoder`, `--FlagReranker`, `--CrossEncoder`, `--BGEM3FlagModel`).
All other flags are passed directly to Uvicorn CLI parsing.

Defaults (if not provided): `--host 0.0.0.0 --port 8067`.

Example with extra Uvicorn options:

```bash
meow-embed \
  --SentenceTransformer "sergeyzh/BERTA" \
  --reload \
  --log-level debug \
  --proxy-headers \
  --forwarded-allow-ips="*"
```

### 2) Use Python client

Install client extras from git:

```bash
uv add "git+https://github.com/dantetemplar/meow-embed.git[client]"

# if you don't have git installed you can use the zip file
uv add "https://github.com/dantetemplar/meow-embed/archive/refs/heads/main.zip#egg=meow-embed[client]"
```

Example:

```python
import httpx
import numpy as np

from meow_embed import MeowEmbedClient


def numpy_info(array: np.ndarray) -> str:
    return f"[ndarray] shape={array.shape}, dtype={array.dtype}"


meow = MeowEmbedClient(
    client=httpx.Client(base_url="http://127.0.0.1:8067"),
    aclient=httpx.AsyncClient(base_url="http://127.0.0.1:8067"), # NOTE: async version
)
models = meow.models() # NOTE: async version is await meow.amodels()
print(models)
# { 'models': [
#   {
#     'batch_size': 32,
#     'dimensions': 768,
#     'id': 'sergeyzh/BERTA',
#     'type': 'dense'},
#   { 'batch_size': 32,
#     'dimensions': 105879,
#     'id': 'opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1',
#     'type': 'sparse'},
#   { 'batch_size': 128,
#     'dimensions': None,
#     'id': 'BAAI/bge-reranker-v2-m3',
#     'type': 'reranker'},
#   { 'batch_size': 256,
#     'dimensions': None,
#     'id': 'BAAI/bge-m3',
#     'type': 'bgeM3'}]}

result = meow.embed(
    {
        "texts": ["hello world", "one request for both outputs"],
        "dense_model_id": "sergeyzh/BERTA",
        "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        "bge_model_id": "BAAI/bge-m3",
    }
) # NOTE: async version is await meow.aembed(...)
print("======== embed ========")
print(f"texts_count: {result.texts_count}")
print(result.pretty_timings())

print("dense")
print(f"    .model_id: {result.dense.model_id}")
print(f"    .vectors: {numpy_info(result.dense.vectors)}")
# dense
# .model_id: sergeyzh/BERTA
# .vectors: [ndarray] shape=(2, 768), dtype=float32

print("sparse")
print(f"    .model_id: {result.sparse.model_id}")
print(f"    .items: total {len(result.sparse.items)} items")
print(f"        .[0].indices: {numpy_info(result.sparse.items[0].indices)}")
print(f"        .[0].values: {numpy_info(result.sparse.items[0].values)}")
# sparse
# .model_id: opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1
# .items: total 2 items
#     .[0].indices: [ndarray] shape=(226,), dtype=uint32
#     .[0].values: [ndarray] shape=(226,), dtype=float32

print("bgeM3")
print(f"    .model_id: {result.bgeM3.model_id}")
print(f"    .dense.vectors: {numpy_info(result.bgeM3.dense.vectors)}")
print(f"    .sparse.items: total {len(result.bgeM3.sparse.items)} items")
print(f"        .[0].indices: {numpy_info(result.bgeM3.sparse.items[0].indices)}")
print(f"        .[0].values: {numpy_info(result.bgeM3.sparse.items[0].values)}")
print(f"    .colbert: total {len(result.bgeM3.colbert)} items")
print(f"        .[0]: {numpy_info(result.bgeM3.colbert[0])}")
# bgeM3
# .model_id: BAAI/bge-m3
# .dense.vectors: [ndarray] shape=(2, 1024), dtype=float32
# .sparse.items: total 2 items
#     .[0].indices: [ndarray] shape=(3,), dtype=uint32
#     .[0].values: [ndarray] shape=(3,), dtype=float32
# .colbert: total 2 items
#     .[0]: [ndarray] shape=(4, 1024), dtype=float32


# Single text: use embed_one / aembed_one for a 1D dense vector (no .vectors[0]).
one = meow.embed_one(
    {
        "text": "hello world",
        "dense_model_id": "sergeyzh/BERTA",
        "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        "bge_model_id": "BAAI/bge-m3",
    }
) # NOTE: async version is await meow.aembed_one(...)
assert one.dense.vector.shape == (768,)
print("======== embed one ========")
print(one.pretty_timings())
print("dense")
print(f"    .model_id: {one.dense.model_id}")
print(f"    .vectors: {numpy_info(one.dense.vector)}")
# dense
#     .model_id: sergeyzh/BERTA
#     .vectors: [ndarray] shape=(768,), dtype=float32

print("sparse")
print(f"    .model_id: {one.sparse.model_id}")
print(f"    .item.indices: {numpy_info(one.sparse.item.indices)}")
print(f"    .item.values: {numpy_info(one.sparse.item.values)}")
# sparse
#     .model_id: opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1
#     .item.indices: [ndarray] shape=(226,), dtype=uint32
#     .item.values: [ndarray] shape=(226,), dtype=float32

print("bgeM3")
print(f"    .model_id: {one.bgeM3.model_id}")
print(f"    .dense.vector: {numpy_info(one.bgeM3.dense.vector)}")
print(f"    .sparse.item.indices: {numpy_info(one.bgeM3.sparse.item.indices)}")
print(f"    .sparse.item.values: {numpy_info(one.bgeM3.sparse.item.values)}")
print(f"    .colbert: {numpy_info(one.bgeM3.colbert)}")
# bgeM3
#     .model_id: BAAI/bge-m3
#     .dense.vector: [ndarray] shape=(1024,), dtype=float32
#     .sparse.item.indices: [ndarray] shape=(3,), dtype=uint32
#     .sparse.item.values: [ndarray] shape=(3,), dtype=float32
#     .colbert: [ndarray] shape=(4, 1024), dtype=float32
```

Rerank with FlagReranker (`BAAI/bge-reranker-v2-m3`) or CrossEncoder (`jinaai/jina-reranker-v2-base-multilingual`):

```python
# one query vs many documents -> scores shape (1, len(docs))
reranked = meow.rerank(
    {
        "reranker_model_id": "BAAI/bge-reranker-v2-m3",
        "query": "what is panda?",
        "docs": [
            "hi",
            "The giant panda is a bear species endemic to China.",
        ],
        "activation_fn": "identity",  # raw logits; "sigmoid" = 0..1; omit = model default
    }
)  # NOTE: async version is await meow.arerank(...)
print(reranked.model_id)
print(reranked.shape)   # (1, 2)
print(reranked.scores)  # [[-8.18, 5.21]]

# bulk: many queries vs the same doc list -> scores shape (len(queries), len(docs))
bulk = meow.rerank(
    {
        "reranker_model_id": "jinaai/jina-reranker-v2-base-multilingual",
        "queries": ["organic skincare", "makeup trends"],
        "docs": [
            "Organic skincare for sensitive skin with aloe vera.",
            "New makeup trends focus on bold colors.",
        ],
    }
)
print(bulk.shape)   # (2, 2)
print(bulk.scores)
```

### Client-side LMDB cache

Pass an `EmbedCache` instance to enable caching. Embeddings are keyed per text and model options. `EmbedCache.open(path)` opens an LMDB directory (default `~/.cache/meow-embed/client-cache.lmdb`); the optional `map_size=` kwarg sets the map size in bytes (default 2 GiB). For full control over the LMDB environment, construct `EmbedCache(env=lmdb.open(...))` directly.

Quick default-path convenience:

```python
import httpx

from meow_embed import MeowEmbedClient

meow = MeowEmbedClient(
    client=httpx.Client(base_url="http://127.0.0.1:8067"),
    use_cache=True,  # auto-creates EmbedCache at ~/.cache/meow-embed/client-cache.lmdb
)
```

```python
import tempfile
from pathlib import Path

import httpx

from meow_embed import EmbedCache, MeowEmbedClient
from meow_embed.types import DenseSparseEmbedRequestDict


with tempfile.TemporaryDirectory() as tmp:
    cache_path = Path(tmp) / "embed-cache.lmdb"
    cache = EmbedCache.open(cache_path)
    try:
        meow = MeowEmbedClient(
            httpx.Client(base_url="http://127.0.0.1:8067"),
            cache=cache,
        )
        payload: DenseSparseEmbedRequestDict = {
            "texts": ["hello world", "cache demo"],
            "dense_model_id": "sergeyzh/BERTA",
            "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        }
        meow.embed(payload)  # fills LMDB on miss
        meow.embed(payload)  # reads from LMDB
    finally:
        cache.close()
```

### curl examples

#### Check loaded models

```bash
curl -sS "http://127.0.0.1:8067/models"
```

#### JSON request (dense + sparse in one call)

```bash
curl -sS -X POST "http://127.0.0.1:8067/embed" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "dense_model_id": "sergeyzh/BERTA",
    "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    "texts": ["hello", "world"],
    "dense_truncate_dim": null,
    "sparse_max_active_dims": null,
    "sparse_pruning_ratio": null
  }'
```

#### JSON request (rerank: query vs docs)

```bash
curl -sS -X POST "http://127.0.0.1:8067/rerank" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "reranker_model_id": "BAAI/bge-reranker-v2-m3",
    "query": "what is panda?",
    "docs": ["hi", "The giant panda is a bear species endemic to China."]
  }'
```

#### JSON request (rerank: queries vs docs bulk)

```bash
curl -sS -X POST "http://127.0.0.1:8067/rerank" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "reranker_model_id": "BAAI/bge-reranker-v2-m3",
    "queries": ["what is panda?", "capital of france"],
    "docs": ["The giant panda is a bear species endemic to China.", "Paris is the capital city of France."]
  }'
```

#### Gzipped request + gzipped response

```bash
cat <<'JSON' | gzip -c | curl -sS -X POST "http://127.0.0.1:8067/embed" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -H "Content-Encoding: gzip" \
  -H "Accept-Encoding: gzip" \
  --compressed \
  --data-binary @-
{
  "dense_model_id": "sergeyzh/BERTA",
  "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
  "texts": ["hello", "world"]
}
JSON
```

#### JSON request (BGE-M3 dense+sparse+colbert)

```bash
curl -sS -X POST "http://127.0.0.1:8067/embed" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "bge_model_id": "BAAI/bge-m3",
    "texts": ["What is BGE M3?", "Definition of BM25"]
  }'
```


## Development

Install as editable package with the `all` extra and the `dev` dependency group (pytest, mypy, pyright, …):

```bash
uv pip install -e ".[all]" --group dev --extra-index-url https://download.pytorch.org/whl/cu126
```

Run server:

```bash
meow-embed \
  --SentenceTransformer "sergeyzh/BERTA" \
  --SparseEncoder "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1" \
  --BGEM3FlagModel "BAAI/bge-m3" '{"use_fp16": true}' \
  --FlagReranker "BAAI/bge-reranker-v2-m3" '{"use_fp16": true}' \
  --CrossEncoder "jinaai/jina-reranker-v2-base-multilingual" '{"model_kwargs": {"torch_dtype": "auto"}, "trust_remote_code": true}' \
  --host 0.0.0.0
```

Run pytest:

```bash
python -m pytest
```

Run mypy tests:

```bash
python -m mypy_test tests/
```

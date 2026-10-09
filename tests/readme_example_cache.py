import tempfile
from pathlib import Path

import httpx

from meow_embed import EmbedCache, MeowEmbedClient
from meow_embed.types import EmbedRequestPayload

with tempfile.TemporaryDirectory() as tmp:
    cache_path = Path(tmp) / "embed-cache.lmdb"
    cache = EmbedCache.open(cache_path)
    try:
        meow = MeowEmbedClient(
            httpx.Client(base_url="http://127.0.0.1:8067"),
            cache=cache,
        )
        payload: EmbedRequestPayload = {
            "texts": ["hello world", "cache demo"],
            "dense_model_id": "sergeyzh/BERTA",
            "sparse_model_id": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        }
        first = meow.embed(payload)  # fills LMDB on miss
        print("======== cache miss ========")
        print(first.pretty_timings())
        second = meow.embed(payload)  # reads from LMDB
        print("======== cache hit ========")
        print(second.pretty_timings())
    finally:
        cache.close()

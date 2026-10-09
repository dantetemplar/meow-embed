from __future__ import annotations

import base64
import gzip
import json
from io import BytesIO
from pathlib import Path
from typing import Any, assert_type, cast

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from meow_embed import EmbedCache, MeowEmbedClient, server
from meow_embed.media import normalize_embed_payload
from meow_embed.types import (
    EmbedOneRequestPayload,
    EmbedRequestPayload,
    MediaDataDict,
    ParsedEmbedOne,
    ParsedEmbedResponse,
)


def image_bytes(color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(output, format="PNG")
    return output.getvalue()


def media(raw: bytes, filename: str = "") -> MediaDataDict:
    return {"data": base64.b64encode(raw).decode("ascii"), "filename": filename}


class DenseModelStub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[tuple[str, list[Any], dict[str, Any]]] = []
        self.paths: list[Path] = []

    def _encode(self, task: str, inputs: list[Any], **kwargs: Any) -> np.ndarray:
        self.calls.append((task, inputs, kwargs))
        rows = []
        for item in inputs:
            if isinstance(item, str):
                rows.append([float(item), 0.0])
            else:
                text = item.get("text", "0")
                rows.append([float(text.split()[0]), 1.0])
                for video in item.get("video", []):
                    assert video["array"].shape == (2, 8, 8, 3)
                    assert video["video_metadata"] == {
                        "fps": 1.0,
                        "total_num_frames": 2,
                    }
        return np.asarray(rows, dtype=np.float32)

    def encode(self, inputs: list[Any], **kwargs: Any) -> np.ndarray:
        return self._encode("encode", inputs, **kwargs)

    def encode_query(self, inputs: list[Any], **kwargs: Any) -> np.ndarray:
        return self._encode("query", inputs, **kwargs)

    def encode_document(self, inputs: list[Any], **kwargs: Any) -> np.ndarray:
        return self._encode("document", inputs, **kwargs)


def test_normalize_files_bytes_and_base64(tmp_path: Path) -> None:
    path = tmp_path / "shoe.png"
    path.write_bytes(image_bytes())
    payload: EmbedRequestPayload = {
        "dense_model_id": "d",
        "texts": [
            {
                "text": "<|image|><|image|><|image|>",
                "image": [path, path.read_bytes(), media(path.read_bytes())],
            }
        ],
    }
    normalized = normalize_embed_payload(payload)
    images = cast(dict[str, Any], normalized["texts"][0])["image"]
    assert all(base64.b64decode(item["data"]) == path.read_bytes() for item in images)
    assert images[0]["filename"] == "shoe.png"
    assert cast(dict[str, Any], payload["texts"][0])["image"][0] == path
    json.dumps(normalized)


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"image": []},
        {"text": 1},
        {"unknown": "x"},
        {"text": "<|image|><|image|>", "image": b"one"},
        {"text": "<|video|>", "image": b"one"},
        {"audio": {"data": "invalid!"}},
    ],
)
def test_invalid_inputs(item: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        normalize_embed_payload(
            cast(EmbedRequestPayload, {"dense_model_id": "d", "texts": [item]})
        )


@pytest.mark.parametrize(
    "models", [{"sparse_model_id": "s"}, {"dense_model_id": "d", "bge_model_id": "b"}]
)
def test_multimodal_requires_dense_only(models: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="dense model only"):
        normalize_embed_payload(
            cast(EmbedRequestPayload, {**models, "texts": [{"image": b"image"}]})
        )


@pytest.mark.parametrize("task", [None, "query", "document"])
def test_server_interleaving_and_batch_order(
    task: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "SentenceTransformer", DenseModelStub)
    audio_paths = []
    video_paths = []

    def load_video(
        path: str, backend: str
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        assert backend == "pyav"
        assert Path(path).read_bytes() == b"video data"
        video_paths.append(Path(path))
        return np.zeros((2, 8, 8, 3), dtype=np.uint8), {
            "fps": 1.0,
            "total_num_frames": 2,
        }

    monkeypatch.setattr(server, "load_video", load_video)

    def load_audio(path: str, sampling_rate: int, backend: str) -> np.ndarray:
        assert sampling_rate == 16000
        assert backend == "librosa"
        assert Path(path).read_bytes() == b"audio data"
        audio_paths.append(Path(path))
        return np.zeros(160, dtype=np.float32)

    monkeypatch.setattr(server, "load_audio", load_audio)
    app = server.build_app(
        server.ModelConfig(
            [
                server.ModelInstanceConfig("dense", "d", {}),
            ]
        )
    )
    with TestClient(app) as client:
        model = app.state.dense_models["d"]
        response = client.post(
            "/embed",
            json={
                "dense_model_id": "d",
                "dense_task": task,
                "texts": [
                    "1",
                    {
                        "text": "2 <|image|><|image|><|video|><|audio|>",
                        "image": [media(image_bytes()), media(image_bytes("blue"))],
                        "video": media(b"video data", "../../demo.mp4"),
                        "audio": media(b"audio data", "sound.wav"),
                    },
                    {"image": media(image_bytes())},
                    "3",
                ],
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        vectors = np.frombuffer(
            base64.b64decode(result["dense"]["data"]), dtype=np.float32
        ).reshape(4, 2)
        np.testing.assert_array_equal(vectors, [[1, 0], [2, 1], [0, 1], [3, 0]])
        assert result["texts_count"] == 4
        assert all(call[0] == (task or "encode") for call in model.calls)
        interleaved = model.calls[1][1][0]
        assert interleaved["text"] == "2 <|image|><|image|><|video|><|audio|>"
        assert [image.getpixel((0, 0)) for image in interleaved["image"]] == [
            (255, 0, 0),
            (0, 0, 255),
        ]
        assert interleaved["audio"][0]["sampling_rate"] == 16000
        assert video_paths and audio_paths
        assert all(not path.exists() for path in video_paths + audio_paths)


def test_dense_prompt_applies_only_to_text() -> None:
    class PromptModel(DenseModelStub):
        def _encode(self, task: str, inputs: list[Any], **kwargs: Any) -> np.ndarray:
            self.calls.append((task, inputs, kwargs))
            return np.zeros((len(inputs), 2), dtype=np.float32)

    model = PromptModel()
    request = server.EmbedRequest.model_validate(
        {
            "dense_model_id": "d",
            "dense_prompt": "prefix: ",
            "texts": [
                "hello",
                {"text": "hello <|image|>", "image": media(image_bytes())},
                {"image": media(image_bytes())},
            ],
        }
    )
    server.encode_dense_inputs(cast(Any, model), request)
    assert model.calls[0][2]["prompt"] == "prefix: "
    assert model.calls[1][1][0]["text"] == "prefix: hello <|image|>"
    assert model.calls[1][2]["prompt"] == ""
    assert "text" not in model.calls[2][1][0]


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},
        {"texts": [{"image": "/etc/passwd"}]},
        {"texts": [{"image": {"data": "invalid!"}}]},
        {"texts": [{"text": "<|image|><|image|>", "image": media(image_bytes())}]},
        {"texts": [{"image": media(image_bytes())}], "sparse_model_id": "s"},
    ],
)
def test_server_rejects_invalid_payloads(payload: dict[str, Any]) -> None:
    with TestClient(server.build_app(server.ModelConfig([]))) as client:
        response = client.post("/embed", json={"dense_model_id": "d", **payload})
        assert response.status_code == 422


def test_bad_image_returns_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "SentenceTransformer", DenseModelStub)
    app = server.build_app(
        server.ModelConfig([server.ModelInstanceConfig("dense", "d", {})])
    )
    with TestClient(app) as client:
        response = client.post(
            "/embed",
            json={"dense_model_id": "d", "texts": [{"image": media(b"not an image")}]},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid image media."


def transport_handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(gzip.decompress(request.content))
    assert all(isinstance(item, (str, dict)) for item in payload["texts"])
    vectors = np.zeros((len(payload["texts"]), 2), dtype=np.float32)
    return httpx.Response(
        200,
        json={
            "texts_count": len(payload["texts"]),
            "dense": {
                "model_id": "d",
                "shape": list(vectors.shape),
                "dtype": "float32",
                "encoding": "base64",
                "data": base64.b64encode(vectors.tobytes()).decode(),
            },
            "sparse": None,
            "bgeM3": None,
        },
    )


def test_sync_client_typing_transport_and_cache(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(gzip.decompress(request.content)))
        return transport_handler(request)

    cache = EmbedCache.open(tmp_path / "cache")
    path = tmp_path / "shoe.png"
    path.write_bytes(image_bytes())
    try:
        with httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as http:
            meow = MeowEmbedClient(client=http, cache=cache)
            payload: EmbedRequestPayload = {
                "dense_model_id": "d",
                "texts": ["hello", {"image": path}],
            }
            result = meow.embed(payload)
            assert_type(result, ParsedEmbedResponse)
            assert result.dense is not None
            assert result.sparse is None
            assert result.bgeM3 is None
            assert result.dense.vectors.shape == (2, 2)
            meow.embed(payload)
            assert len(requests) == 1
            path.write_bytes(image_bytes("blue"))
            meow.embed(payload)
            assert len(requests) == 2
            assert len(requests[1]["texts"]) == 1
            assert (
                base64.b64decode(requests[1]["texts"][0]["image"][0]["data"])
                == path.read_bytes()
            )
            one: EmbedOneRequestPayload = {
                "dense_model_id": "d",
                "text": {"image": path},
            }
            single = meow.embed_one(one)
            assert_type(single, ParsedEmbedOne)
            assert single.dense is not None
            assert single.sparse is None
            assert single.bgeM3 is None
            assert single.dense.vector.shape == (2,)
            assert len(requests) == 2
    finally:
        cache.close()


@pytest.mark.anyio
async def test_async_client_multimodal_typing_and_cache(tmp_path: Path) -> None:
    cache = EmbedCache.open(tmp_path / "cache")
    try:
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(transport_handler)
        ) as http:
            meow = MeowEmbedClient(aclient=http, cache=cache)
            payload: EmbedOneRequestPayload = {
                "dense_model_id": "d",
                "text": {"image": image_bytes()},
            }
            result = await meow.aembed_one(payload)
            assert_type(result, ParsedEmbedOne)
            assert result.dense is not None
            assert result.sparse is None
            assert result.bgeM3 is None
            assert result.dense.vector.shape == (2,)
            cached = await meow.aembed_one(payload)
            assert cached.server_timings is None
            batch: EmbedRequestPayload = {
                "dense_model_id": "d",
                "texts": [payload["text"], "hello"],
            }
            many = await meow.aembed(batch)
            assert_type(many, ParsedEmbedResponse)
            assert many.dense is not None
            assert many.sparse is None
            assert many.bgeM3 is None
            assert many.dense.vectors.shape == (2, 2)
            assert many.texts_count == 2
    finally:
        cache.close()


def test_cache_distinguishes_order_and_modalities(tmp_path: Path) -> None:
    cache = EmbedCache.open(tmp_path / "cache")
    try:

        def key(item: Any) -> bytes:
            progress = cache.prepare(
                cast(EmbedRequestPayload, {"dense_model_id": "d", "texts": [item]})
            )
            return progress.streams[0].keys[0]

        assert key({"image": [b"a", b"b"]}) != key({"image": [b"b", b"a"]})
        assert key({"audio": b"a"}) != key({"video": b"a"})
        assert key({"image": b"a"}) != key(
            "multimodal:"
            + json.dumps(
                {"image": [media(b"a")]}, sort_keys=True, separators=(",", ":")
            )
        )
    finally:
        cache.close()

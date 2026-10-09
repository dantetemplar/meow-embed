"""Opt-in real-model coverage (MEOW_EMBED_TEST_EMBEDDINGGEMMA=1)."""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Any

import av
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sentence_transformers import SentenceTransformer

from meow_embed import EmbedCache, MeowEmbedClient, server
from meow_embed.types import EmbedRequestPayload, EmbedInput

pytestmark = pytest.mark.skipif(
    os.getenv("MEOW_EMBED_TEST_EMBEDDINGGEMMA") != "1",
    reason="Set MEOW_EMBED_TEST_EMBEDDINGGEMMA=1 to download and test the real model.",
)
MODEL_ID = "google/embeddinggemma-2"


@pytest.fixture(scope="module")
def real_model() -> SentenceTransformer:
    return SentenceTransformer(
        MODEL_ID, device="cpu", model_kwargs={"torch_dtype": "float32"}
    )


@pytest.fixture
def media_files(tmp_path: Path) -> dict[str, Path]:
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "red").save(image)
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as audio_stream:
        audio_stream.setnchannels(2)
        audio_stream.setsampwidth(2)
        audio_stream.setframerate(32000)
        audio_stream.writeframes(np.zeros((32000, 2), dtype=np.int16).tobytes())
    video = tmp_path / "video.mp4"
    with av.open(str(video), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for _ in range(4):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), 128, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return {"image": image, "audio": audio, "video": video}


def test_real_model_all_modalities_and_cache(
    real_model: SentenceTransformer,
    media_files: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server, "SentenceTransformer", lambda *args, **kwargs: real_model
    )
    app = server.build_app(
        server.ModelConfig([server.ModelInstanceConfig("dense", MODEL_ID, {})])
    )
    requests: list[httpx.Request] = []
    with TestClient(app) as api:

        def send(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            response = api.request(
                request.method,
                request.url.path,
                content=request.content,
                headers=dict(request.headers),
            )
            return httpx.Response(
                response.status_code, content=response.content, headers=response.headers
            )

        cache = EmbedCache.open(tmp_path / "cache")
        try:
            with httpx.Client(
                base_url="http://test", transport=httpx.MockTransport(send)
            ) as http:
                client = MeowEmbedClient(client=http, cache=cache)
                inputs: list[EmbedInput] = [
                    "Waterproof trail running shoes",
                    {"image": media_files["image"]},
                    {"audio": media_files["audio"]},
                    {"video": media_files["video"]},
                    {
                        "text": (
                            "Shoes <|image|> Mesh <|image|> "
                            "Demo <|video|> Sound <|audio|>"
                        ),
                        "image": [media_files["image"], media_files["image"]],
                        "video": media_files["video"],
                        "audio": media_files["audio"],
                    },
                    {"text": "A red image", "image": media_files["image"]},
                ]
                payload: EmbedRequestPayload = {
                    "dense_model_id": MODEL_ID,
                    "texts": inputs,
                }
                result = client.embed(payload)
                assert result.dense is not None
                assert result.sparse is None
                assert result.bgeM3 is None
                assert result.dense.vectors.shape == (6, 768)
                assert np.isfinite(result.dense.vectors).all()
                assert (np.linalg.norm(result.dense.vectors, axis=1) > 0).all()
                cached = client.embed(payload)
                assert cached.dense is not None
                np.testing.assert_array_equal(
                    result.dense.vectors, cached.dense.vectors
                )
                assert len(requests) == 1
                assert cached.server_timings is None
                for index, item in enumerate(inputs):
                    one = client.embed_one({"dense_model_id": MODEL_ID, "text": item})
                    assert one.dense is not None
                    np.testing.assert_array_equal(
                        one.dense.vector, result.dense.vectors[index]
                    )
                assert len(requests) == 1
        finally:
            cache.close()


@pytest.mark.parametrize("task", [None, "query", "document"])
def test_real_model_tasks_prompts_and_truncation(
    real_model: SentenceTransformer,
    media_files: dict[str, Path],
    task: Any,
) -> None:
    from meow_embed.media import normalize_embed_payload

    payload: EmbedRequestPayload = {
        "dense_model_id": MODEL_ID,
        "dense_task": task,
        "dense_prompt": "title: none | text: ",
        "dense_truncate_dim": 256,
        "texts": [
            "Waterproof trail running shoes",
            {"text": "Shoes <|image|>", "image": media_files["image"]},
        ],
    }
    request = server.EmbedRequest.model_validate(normalize_embed_payload(payload))
    actual = server.encode_dense_inputs(real_model, request)
    with Image.open(media_files["image"]) as image:
        inputs: dict[str, Any] = {
            "text": "title: none | text: Shoes <|image|>",
            "image": [image.convert("RGB")],
        }
        if task == "query":
            expected = real_model.encode_query(
                inputs, prompt="", truncate_dim=256, convert_to_numpy=True
            )
        elif task == "document":
            expected = real_model.encode_document(
                inputs, prompt="", truncate_dim=256, convert_to_numpy=True
            )
        else:
            expected = real_model.encode(
                inputs, prompt="", truncate_dim=256, convert_to_numpy=True
            )
    assert actual.shape == (2, 256)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[1], expected, rtol=1e-5, atol=1e-6)

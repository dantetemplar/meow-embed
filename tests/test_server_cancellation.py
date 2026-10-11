from __future__ import annotations

import asyncio
import base64
import gzip
import json
import threading
from typing import Any, Literal

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from meow_embed import cancellation, server


class ModelStub:
    batch_size = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[list[Any]] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.cancel_event: threading.Event | None = None

    def vectors(self, inputs: list[Any]) -> np.ndarray:
        self.calls.append(inputs)
        self.cancel_event = cancellation._cancel_event.get()
        self.started.set()
        assert self.release.wait(timeout=5), "Inference was not released"
        return np.array([[float(item), 0] for item in inputs], dtype=np.float32)

    def encode(
        self, inputs: list[str], batch_size: int = 2, **kwargs: Any
    ) -> Any:
        return self.vectors(inputs)

    encode_query = encode
    encode_document = encode

    def predict(
        self, pairs: list[list[str]], batch_size: int = 2, **kwargs: Any
    ) -> np.ndarray:
        return self.vectors([pair[1] for pair in pairs])[:, 0]

    compute_score = predict


class CrossEncoderStub(ModelStub):
    pass


class SparseStub(ModelStub):
    def encode(
        self, inputs: list[str], batch_size: int = 2, **kwargs: Any
    ) -> torch.Tensor:
        return torch.from_numpy(self.vectors(inputs)).to_sparse()

    encode_query = encode
    encode_document = encode


class BGEStub(ModelStub):
    def encode(
        self, inputs: list[str], batch_size: int = 2, **kwargs: Any
    ) -> dict[str, Any]:
        vectors = self.vectors(inputs)
        return {
            "dense_vecs": vectors,
            "lexical_weights": [{"0": float(row[0])} for row in vectors],
            "colbert_vecs": [row[None, :] for row in vectors],
        }


ModelKind = Literal["dense", "sparse", "bgeM3", "reranker", "crossEncoder"]
KINDS: list[ModelKind] = ["dense", "sparse", "bgeM3", "reranker", "crossEncoder"]


def make_app(kind: ModelKind, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(server, "SentenceTransformer", ModelStub)
    monkeypatch.setattr(server, "SparseEncoder", SparseStub)
    monkeypatch.setattr(server, "BGEM3FlagModel", BGEStub)
    monkeypatch.setattr(server, "FlagReranker", ModelStub)
    monkeypatch.setattr(server, "CrossEncoder", CrossEncoderStub)
    return server.build_app(
        server.ModelConfig([server.ModelInstanceConfig(kind, "m", {})])
    )


def model_for(app: Any, kind: ModelKind) -> ModelStub:
    attr = {
        "dense": "dense_models",
        "sparse": "sparse_models",
        "bgeM3": "bge_m3_models",
        "reranker": "reranker_models",
        "crossEncoder": "cross_encoder_models",
    }[kind]
    return getattr(app.state, attr)["m"]


def payload_for(kind: ModelKind) -> tuple[str, dict[str, Any]]:
    if kind in ("reranker", "crossEncoder"):
        return "/rerank", {
            "reranker_model_id": "m",
            "queries": ["a", "b"],
            "docs": ["1", "2", "3"],
        }
    key = {
        "dense": "dense_model_id",
        "sparse": "sparse_model_id",
        "bgeM3": "bge_model_id",
    }[kind]
    return "/embed", {key: "m", "texts": ["1", "2", "3", "4", "5"]}


class RequestHarness:
    def __init__(
        self, app: Any, path: str, payload: dict[str, Any], compressed: bool = False
    ) -> None:
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        body = json.dumps(payload).encode()
        headers = [(b"content-type", b"application/json")]
        if compressed:
            body = gzip.compress(body)
            headers.append((b"content-encoding", b"gzip"))
        # Exercise preservation of all body chunks, not just a single message.
        middle = len(body) // 2
        self.incoming.put_nowait(
            {"type": "http.request", "body": body[:middle], "more_body": True}
        )
        self.incoming.put_nowait(
            {"type": "http.request", "body": body[middle:], "more_body": False}
        )
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
        }

        async def send(message: dict[str, Any]) -> None:
            self.sent.append(message)

        self.task = asyncio.create_task(app(scope, self.incoming.get, send))

    def disconnect(self) -> None:
        self.incoming.put_nowait({"type": "http.disconnect"})

    async def finish(self) -> None:
        await asyncio.wait_for(self.task, timeout=5)


@pytest.mark.anyio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("compressed", [False, True])
async def test_disconnect_stops_after_current_batch(
    kind: ModelKind, compressed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(kind, monkeypatch)
    async with app.router.lifespan_context(app):
        model = model_for(app, kind)
        model.release.clear()
        path, payload = payload_for(kind)
        request = RequestHarness(app, path, payload, compressed)
        try:
            assert await asyncio.to_thread(model.started.wait, 5)
            request.disconnect()
            assert model.cancel_event is not None
            assert await asyncio.to_thread(model.cancel_event.wait, 5)
            assert app.state.inference_lock.locked(), "Do not release the GPU mid-batch"
        finally:
            model.release.set()
            await request.finish()
        assert len(model.calls) == 1
        assert len(model.calls[0]) == 2
        assert request.sent == []
        assert not app.state.inference_lock.locked()
        # Cancellation belongs to one request, not to the model or application.
        next_request = RequestHarness(app, path, payload)
        await next_request.finish()
        assert next_request.sent[0]["status"] == 200
        assert len(model.calls) == 4


@pytest.mark.anyio
@pytest.mark.parametrize("kind", KINDS)
async def test_disconnect_while_waiting_for_lock_skips_inference(
    kind: ModelKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(kind, monkeypatch)

    class ObservedLock:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.lock.acquire()
            self.waiting = threading.Event()
            self.cancel_event: threading.Event | None = None

        def acquire(self, timeout: float) -> bool:
            self.cancel_event = cancellation._cancel_event.get()
            self.waiting.set()
            return self.lock.acquire(timeout=timeout)

        def release(self) -> None:
            self.lock.release()

    async with app.router.lifespan_context(app):
        lock = ObservedLock()
        app.state.inference_lock = lock
        path, payload = payload_for(kind)
        request = RequestHarness(app, path, payload)
        try:
            assert await asyncio.to_thread(lock.waiting.wait, 5)
            request.disconnect()
            assert lock.cancel_event is not None
            assert await asyncio.to_thread(lock.cancel_event.wait, 5)
            # Must finish even though another request still owns the lock.
            await request.finish()
            assert model_for(app, kind).calls == []
            assert request.sent == []
        finally:
            lock.release()
            await request.finish()


@pytest.mark.parametrize("kind", KINDS)
def test_batched_results_keep_order(
    kind: ModelKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(kind, monkeypatch)
    path, payload = payload_for(kind)
    with TestClient(app) as client:
        response = client.post(path, json=payload)
        assert response.status_code == 200, response.text
        assert len(model_for(app, kind).calls) == 3
        result = response.json()
        if kind in ("reranker", "crossEncoder"):
            assert result["scores"] == [[1, 2, 3], [1, 2, 3]]
        else:
            result = result["bgeM3"] if kind == "bgeM3" else result
            if kind in ("dense", "bgeM3"):
                dense = result["dense"]
                vectors = np.frombuffer(
                    base64.b64decode(dense["data"]), dtype=np.float32
                )
                np.testing.assert_array_equal(
                    vectors.reshape(5, 2)[:, 0], [1, 2, 3, 4, 5]
                )
            if kind in ("sparse", "bgeM3"):
                values = [
                    np.frombuffer(
                        base64.b64decode(item["values"]), dtype=np.float32
                    ).item()
                    for item in result["sparse"]["items"]
                ]
                assert values == [1, 2, 3, 4, 5]
            if kind == "bgeM3":
                assert len(result["colbert"]) == 5


@pytest.mark.anyio
async def test_server_cancellation_signals_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = make_app("dense", monkeypatch)
    async with app.router.lifespan_context(app):
        model = model_for(app, "dense")
        model.release.clear()
        request = RequestHarness(app, *payload_for("dense"))
        assert await asyncio.to_thread(model.started.wait, 5)
        request.task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await request.finish()
            assert model.cancel_event is not None
            assert await asyncio.to_thread(model.cancel_event.wait, 5)
        finally:
            model.release.set()
        assert await asyncio.to_thread(app.state.inference_lock.acquire, True, 5)
        app.state.inference_lock.release()
        assert len(model.calls) == 1

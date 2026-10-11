"""Disconnect detection and cooperative cancellation for inference threads."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

import anyio
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class InferenceCancelled(Exception):
    """The client left before inference completed."""


_cancel_event: ContextVar[threading.Event | None] = ContextVar(
    "inference_cancel_event", default=None
)
_T = TypeVar("_T")


def check_cancelled() -> None:
    event = _cancel_event.get()
    if event is not None and event.is_set():
        raise InferenceCancelled


@contextmanager
def inference_lock(lock: threading.Lock) -> Iterator[None]:
    """Wait for exclusive GPU access without retaining disconnected requests."""
    check_cancelled()
    while not lock.acquire(timeout=0.05):
        check_cancelled()
    try:
        check_cancelled()
        yield
    finally:
        lock.release()


def inference_batches(items: list[_T], batch_size: int | None) -> Iterator[list[_T]]:
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise RuntimeError(f"Invalid inference batch size: {batch_size!r}")
    for start in range(0, len(items), batch_size):
        check_cancelled()
        yield items[start : start + batch_size]
        check_cancelled()


class CancelOnDisconnectMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        event = threading.Event()
        token = _cancel_event.set(event)
        # Only the watcher reads the transport; the handler gets every body chunk
        # through a bounded channel. It never races the watcher for receive().
        sender, receiver = anyio.create_memory_object_stream[Message](1)
        try:
            async with sender, receiver, anyio.create_task_group() as tasks:
                async def watch_disconnect() -> None:
                    try:
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                event.set()
                                tasks.cancel_scope.cancel()
                                return
                            await sender.send(message)
                    finally:
                        # Also stop workers when the server cancels the request.
                        event.set()

                async def send_connected(message: Message) -> None:
                    if not event.is_set():
                        await send(message)

                tasks.start_soon(watch_disconnect)
                try:
                    await self.app(scope, receiver.receive, send_connected)
                except InferenceCancelled:
                    # Sync threadpool handlers finish cooperatively; no 500 or
                    # response is needed for a client that has already left.
                    pass
                finally:
                    event.set()
                    tasks.cancel_scope.cancel()
        finally:
            event.set()
            _cancel_event.reset(token)

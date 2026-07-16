"""One in-flight audit factory per evidence key."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from threading import Lock
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

_LOCK = Lock()
_INFLIGHT: dict[str, Future] = {}


def _drop_inflight(key: str, fut: Future) -> None:
    with _LOCK:
        if _INFLIGHT.get(key) is fut:
            del _INFLIGHT[key]


def _fail_shared(fut: Future, exc: BaseException) -> None:
    if fut.done():
        return
    # concurrent.futures rejects CancelledError-like states for waiters; use TimeoutError.
    if isinstance(exc, asyncio.CancelledError):
        fut.set_exception(TimeoutError("leader cancelled"))
    else:
        fut.set_exception(exc)


async def _await_shared(fut: Future, timeout: float) -> T:
    """Wait without cancelling the shared Future on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not fut.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError()
        await asyncio.sleep(min(0.05, remaining))
    return fut.result()


def run_once(key: str, factory: Callable[[], T], *, timeout: float = 300.0) -> T:
    leader = False
    with _LOCK:
        existing = _INFLIGHT.get(key)
        if existing is not None:
            fut: Future = existing
        else:
            fut = Future()
            _INFLIGHT[key] = fut
            leader = True

    if not leader:
        return fut.result(timeout=timeout)

    try:
        value = factory()
    except BaseException as exc:
        _fail_shared(fut, exc)
        raise
    else:
        if not fut.done():
            fut.set_result(value)
        return value
    finally:
        _drop_inflight(key, fut)


async def run_once_async(
    key: str,
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = 300.0,
) -> T:
    """Waiters await; they do not hold a threadpool worker."""
    leader = False
    with _LOCK:
        existing = _INFLIGHT.get(key)
        if existing is not None:
            fut: Future = existing
        else:
            fut = Future()
            _INFLIGHT[key] = fut
            leader = True

    if not leader:
        return await _await_shared(fut, timeout)

    try:
        # Timeout covers executor queue wait + work (followers already timed).
        value = await asyncio.wait_for(factory(), timeout=timeout)
    except BaseException as exc:
        _fail_shared(fut, exc)
        raise
    else:
        if not fut.done():
            fut.set_result(value)
        return value
    finally:
        _drop_inflight(key, fut)

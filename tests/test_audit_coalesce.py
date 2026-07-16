import asyncio
import threading
import time

from server.audit_coalesce import run_once, run_once_async


def test_run_once_shares_result_across_waiters():
    barrier = threading.Barrier(3)
    calls = {"n": 0}
    results: list[int] = []
    errors: list[BaseException] = []

    def factory():
        calls["n"] += 1
        time.sleep(0.08)
        return 42

    def worker():
        try:
            barrier.wait(timeout=2)
            results.append(run_once("k1", factory))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert errors == []
    assert results == [42, 42, 42]
    assert calls["n"] == 1


def test_run_once_allows_second_call_after_finish():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return calls["n"]

    assert run_once("k2", factory) == 1
    assert run_once("k2", factory) == 2
    assert calls["n"] == 2


def test_run_once_clears_inflight_after_factory_error():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("leader failed")

    try:
        run_once("k3", boom)
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")

    assert run_once("k3", lambda: 7) == 7
    assert calls["n"] == 1


def test_run_once_async_shares_result():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return 99

    async def main():
        a, b = await asyncio.gather(
            run_once_async("ak1", factory),
            run_once_async("ak1", factory),
        )
        return a, b

    assert asyncio.run(main()) == (99, 99)
    assert calls["n"] == 1


def test_run_once_async_waiter_timeout_does_not_poison():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        started.set()
        await release.wait()
        return 11

    async def factory_next():
        calls["n"] += 1
        return 22

    async def main():
        leader = asyncio.create_task(run_once_async("poison", factory, timeout=5.0))
        await started.wait()
        try:
            await run_once_async("poison", factory, timeout=0.08)
            raise AssertionError("expected TimeoutError")
        except TimeoutError:
            pass
        release.set()
        assert await leader == 11
        assert await run_once_async("poison", factory_next, timeout=2.0) == 22
        return calls["n"]

    assert asyncio.run(main()) == 2


def test_run_once_async_leader_timeout_notifies_waiters():
    started = asyncio.Event()
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        started.set()
        await asyncio.sleep(10)
        return 1

    async def main():
        leader = asyncio.create_task(run_once_async("lead-to", factory, timeout=0.12))
        await started.wait()
        waiter = asyncio.create_task(run_once_async("lead-to", factory, timeout=2.0))
        try:
            await leader
            raise AssertionError("expected leader TimeoutError")
        except TimeoutError:
            pass
        try:
            await waiter
            raise AssertionError("expected waiter error")
        except TimeoutError:
            pass
        async def ok():
            return 9
        assert await run_once_async("lead-to", ok, timeout=2.0) == 9
        return calls["n"]

    # leader only; waiter shared the same in-flight factory
    assert asyncio.run(main()) == 1


def test_run_once_async_leader_cancel_notifies_waiters():
    started = asyncio.Event()

    async def factory():
        started.set()
        await asyncio.sleep(10)
        return 1

    async def main():
        leader = asyncio.create_task(run_once_async("lead-c", factory, timeout=5.0))
        await started.wait()
        waiter = asyncio.create_task(run_once_async("lead-c", factory, timeout=5.0))
        await asyncio.sleep(0.05)
        leader.cancel()
        try:
            await leader
        except asyncio.CancelledError:
            pass
        try:
            await waiter
            raise AssertionError("expected waiter error")
        except TimeoutError:
            pass
        async def ok():
            return 5
        assert await run_once_async("lead-c", ok, timeout=2.0) == 5

    asyncio.run(main())

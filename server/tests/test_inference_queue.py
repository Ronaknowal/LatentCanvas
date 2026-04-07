import asyncio
import pytest
import pytest_asyncio

from inference_queue import InferenceQueue


@pytest.mark.asyncio
class TestInferenceQueue:
    async def test_submit_and_get_latest(self):
        q = InferenceQueue()
        await q.submit(b"frame1")
        result = await q.get_latest()
        assert result == b"frame1"

    async def test_get_latest_returns_none_when_empty(self):
        q = InferenceQueue()
        result = await q.get_latest()
        assert result is None

    async def test_latest_wins_overwrites_pending(self):
        q = InferenceQueue()
        await q.submit(b"frame1")
        await q.submit(b"frame2")
        await q.submit(b"frame3")
        result = await q.get_latest()
        assert result == b"frame3"

    async def test_get_latest_clears_after_read(self):
        q = InferenceQueue()
        await q.submit(b"frame1")
        await q.get_latest()
        result = await q.get_latest()
        assert result is None

    async def test_submit_after_get_works(self):
        q = InferenceQueue()
        await q.submit(b"frame1")
        await q.get_latest()
        await q.submit(b"frame2")
        result = await q.get_latest()
        assert result == b"frame2"

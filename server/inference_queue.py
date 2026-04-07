import asyncio


class InferenceQueue:
    def __init__(self):
        self._latest: bytes | None = None
        self._lock = asyncio.Lock()

    async def submit(self, data: bytes) -> None:
        async with self._lock:
            self._latest = data

    async def get_latest(self) -> bytes | None:
        async with self._lock:
            data = self._latest
            self._latest = None
            return data

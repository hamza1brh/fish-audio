"""Request batcher for GPU efficiency."""

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class BatchRequest:
    """Single request in a batch."""

    text: str
    prompt_tokens: list[np.ndarray] | None = None
    prompt_texts: list[str] | None = None
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1024
    chunk_length: int = 200


@dataclass
class BatchResult:
    """Result for a single request."""

    audio: np.ndarray | None = None
    error: Exception | None = None


class RequestBatcher:
    """Batches incoming requests for efficient GPU processing.

    Collects requests within a time window, then processes them together.
    """

    def __init__(
        self,
        engine: Any,
        max_batch_size: int = 4,
        max_wait_ms: int = 100,
    ) -> None:
        self._engine = engine
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._pending: list[tuple[BatchRequest, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._process_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the batcher background task."""
        if self._running:
            return
        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"Batcher started (batch_size={self._max_batch_size}, wait_ms={self._max_wait_ms})")

    async def stop(self) -> None:
        """Stop the batcher."""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        logger.info("Batcher stopped")

    async def submit(self, request: BatchRequest) -> np.ndarray | None:
        """Submit a request and wait for result."""
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        async with self._lock:
            self._pending.append((request, future))

        result: BatchResult = await future

        if result.error:
            raise result.error
        return result.audio

    async def _process_loop(self) -> None:
        """Background loop that processes batches."""
        while self._running:
            await asyncio.sleep(self._max_wait_ms / 1000)

            async with self._lock:
                if not self._pending:
                    continue

                batch = self._pending[: self._max_batch_size]
                self._pending = self._pending[self._max_batch_size :]

            if batch:
                await self._process_batch(batch)

    async def _process_batch(
        self, batch: list[tuple[BatchRequest, asyncio.Future]]
    ) -> None:
        """Process a batch of requests."""
        logger.debug(f"Processing batch of {len(batch)} requests")

        loop = asyncio.get_event_loop()

        for request, future in batch:
            try:
                audio = await loop.run_in_executor(
                    None,
                    lambda r=request: self._generate_single(r),
                )
                future.set_result(BatchResult(audio=audio))
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                future.set_result(BatchResult(error=e))

    def _generate_single(self, request: BatchRequest) -> np.ndarray:
        """Generate audio for a single request (runs in executor)."""
        segments = list(
            self._engine.generate(
                text=request.text,
                prompt_tokens=request.prompt_tokens,
                prompt_texts=request.prompt_texts,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                max_new_tokens=request.max_new_tokens,
                chunk_length=request.chunk_length,
                streaming=False,
            )
        )
        if not segments:
            raise RuntimeError("No audio generated")
        return segments[0]

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        return len(self._pending)

    @property
    def stats(self) -> dict:
        """Batcher statistics."""
        return {
            "pending": self.pending_count,
            "max_batch_size": self._max_batch_size,
            "max_wait_ms": self._max_wait_ms,
            "running": self._running,
        }


"""
S1-Mini Batch Queue
====================

This module provides a batch queue for collecting and grouping TTS requests
for batched GPU inference.

Features:
---------
1. Collects requests for a configurable time window
2. Groups compatible requests by reference audio hash and text length
3. Returns futures that resolve when audio is ready
4. Thread-safe for concurrent request submission

Architecture:
-------------
    BatchQueue
    ├── Request Collection (asyncio.Queue)
    ├── Batch Formation (background task)
    │   ├── Group by reference audio hash
    │   └── Sort by text length for efficient padding
    └── Result Distribution (per-request futures)

Usage:
------
    queue = BatchQueue(max_batch_size=4, batch_timeout_ms=200)
    await queue.start()

    # Submit request and wait for result
    future = await queue.submit(request)
    audio = await future

    await queue.stop()
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

from loguru import logger


class BatchRequestStatus(Enum):
    """Status of a batch request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchRequest:
    """
    A single request submitted to the batch queue.

    Attributes:
        request_id: Unique identifier for the request
        text: Text to synthesize
        reference_audio: Optional reference audio bytes
        reference_text: Text spoken in reference audio
        reference_id: Pre-registered reference ID
        parameters: Generation parameters (temperature, top_p, etc.)
        future: Future that will be resolved with the audio result
        submit_time: Time when request was submitted
        reference_hash: Hash of reference audio for grouping
    """
    request_id: str
    text: str
    reference_audio: Optional[bytes] = None
    reference_text: Optional[str] = None
    reference_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    submit_time: float = field(default_factory=time.time)
    reference_hash: Optional[str] = None

    def __post_init__(self):
        # Compute reference hash for grouping
        if self.reference_audio is not None:
            self.reference_hash = hashlib.sha256(self.reference_audio).hexdigest()[:16]
        elif self.reference_id is not None:
            self.reference_hash = f"id:{self.reference_id}"
        else:
            self.reference_hash = "no_ref"


@dataclass
class Batch:
    """
    A batch of requests to be processed together.

    Attributes:
        batch_id: Unique identifier for the batch
        requests: List of requests in the batch
        reference_hash: Common reference hash (or None for mixed batch)
        created_time: Time when batch was formed
    """
    batch_id: str
    requests: List[BatchRequest]
    reference_hash: Optional[str] = None
    created_time: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def texts(self) -> List[str]:
        return [r.text for r in self.requests]

    @property
    def text_lengths(self) -> List[int]:
        return [len(r.text) for r in self.requests]


class BatchQueue:
    """
    Async batch queue for collecting and grouping TTS requests.

    This class collects requests over a configurable time window and
    groups them into batches for efficient GPU processing.

    Attributes:
        max_batch_size: Maximum number of requests per batch
        batch_timeout_ms: Maximum time to wait for batch to fill
        enable_grouping: Whether to group requests by reference audio
        process_callback: Callback function to process batches

    Example:
        >>> async def process_batch(batch):
        ...     # Process batch and return results
        ...     return results
        >>>
        >>> queue = BatchQueue(
        ...     max_batch_size=4,
        ...     batch_timeout_ms=200,
        ...     process_callback=process_batch,
        ... )
        >>> await queue.start()
        >>>
        >>> # Submit request
        >>> result = await queue.submit(request)
    """

    def __init__(
        self,
        max_batch_size: int = 4,
        batch_timeout_ms: int = 200,
        enable_grouping: bool = True,
        process_callback: Optional[Callable] = None,
    ):
        """
        Initialize the batch queue.

        Args:
            max_batch_size: Maximum requests per batch
            batch_timeout_ms: Time to wait for batch to fill (milliseconds)
            enable_grouping: Group requests by reference audio hash
            process_callback: Async function to process batches
        """
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.enable_grouping = enable_grouping
        self.process_callback = process_callback

        # Internal state
        self._pending_requests: Dict[str, BatchRequest] = {}
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._collector_task: Optional[asyncio.Task] = None
        self._running = False
        self._batch_counter = 0
        self._request_counter = 0
        self._lock = asyncio.Lock()

        # Stats
        self._total_requests = 0
        self._total_batches = 0
        self._total_wait_time = 0.0

    @property
    def is_running(self) -> bool:
        """Check if the batch queue is running."""
        return self._running

    @property
    def pending_count(self) -> int:
        """Number of requests waiting to be batched."""
        return len(self._pending_requests)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "pending_requests": self.pending_count,
            "avg_wait_time_ms": (
                self._total_wait_time / self._total_requests * 1000
                if self._total_requests > 0 else 0
            ),
            "avg_batch_size": (
                self._total_requests / self._total_batches
                if self._total_batches > 0 else 0
            ),
        }

    async def start(self) -> None:
        """Start the batch collector background task."""
        if self._running:
            logger.warning("BatchQueue already running")
            return

        self._running = True
        self._collector_task = asyncio.create_task(self._batch_collector())
        logger.info(
            f"BatchQueue started: max_batch_size={self.max_batch_size}, "
            f"timeout_ms={self.batch_timeout_ms}"
        )

    async def stop(self) -> None:
        """Stop the batch collector and process remaining requests."""
        if not self._running:
            return

        self._running = False

        # Cancel collector task
        if self._collector_task:
            self._collector_task.cancel()
            try:
                await self._collector_task
            except asyncio.CancelledError:
                pass

        # Fail any pending requests
        async with self._lock:
            for request in self._pending_requests.values():
                if not request.future.done():
                    request.future.set_exception(
                        RuntimeError("BatchQueue stopped before request was processed")
                    )
            self._pending_requests.clear()

        logger.info("BatchQueue stopped")

    async def submit(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
        reference_id: Optional[str] = None,
        **parameters,
    ) -> asyncio.Future:
        """
        Submit a request to the batch queue.

        Args:
            text: Text to synthesize
            reference_audio: Optional reference audio bytes
            reference_text: Text spoken in reference audio
            reference_id: Pre-registered reference ID
            **parameters: Additional generation parameters

        Returns:
            Future that will resolve with the audio result

        Raises:
            RuntimeError: If the queue is not running
        """
        if not self._running:
            raise RuntimeError("BatchQueue is not running. Call start() first.")

        # Create request
        self._request_counter += 1
        request_id = f"req_{self._request_counter}_{int(time.time() * 1000)}"

        # Get or create event loop for future
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        request = BatchRequest(
            request_id=request_id,
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            reference_id=reference_id,
            parameters=parameters,
            future=loop.create_future(),
        )

        # Add to pending requests
        async with self._lock:
            self._pending_requests[request_id] = request

        # Notify collector
        await self._request_queue.put(request_id)

        self._total_requests += 1
        logger.debug(f"Request {request_id} submitted (text_len={len(text)})")

        return request.future

    async def _batch_collector(self) -> None:
        """
        Background task that collects requests and forms batches.

        This task:
        1. Waits for the batch timeout or until batch is full
        2. Groups compatible requests together
        3. Sends batches for processing
        """
        logger.info("Batch collector started")

        while self._running:
            try:
                # Wait for at least one request
                try:
                    first_request_id = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=1.0,  # Check running status periodically
                    )
                except asyncio.TimeoutError:
                    continue

                # Record batch start time
                batch_start = time.time()
                timeout_seconds = self.batch_timeout_ms / 1000.0

                # Collect more requests until timeout or batch full
                collected_ids = [first_request_id]

                while len(collected_ids) < self.max_batch_size:
                    remaining = timeout_seconds - (time.time() - batch_start)
                    if remaining <= 0:
                        break

                    try:
                        request_id = await asyncio.wait_for(
                            self._request_queue.get(),
                            timeout=remaining,
                        )
                        collected_ids.append(request_id)
                    except asyncio.TimeoutError:
                        break

                # Get the actual requests
                async with self._lock:
                    requests = [
                        self._pending_requests.pop(rid)
                        for rid in collected_ids
                        if rid in self._pending_requests
                    ]

                if not requests:
                    continue

                # Form batches (potentially multiple if grouping by reference)
                batches = self._form_batches(requests)

                # Process batches
                for batch in batches:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch collector error: {e}")
                await asyncio.sleep(0.1)  # Prevent tight loop on error

        logger.info("Batch collector stopped")

    def _form_batches(self, requests: List[BatchRequest]) -> List[Batch]:
        """
        Form batches from a list of requests.

        If enable_grouping is True, requests are grouped by reference audio hash.
        Within each group, requests are sorted by text length for efficient padding.

        Args:
            requests: List of requests to batch

        Returns:
            List of formed batches
        """
        if not self.enable_grouping:
            # Single batch with all requests
            self._batch_counter += 1
            batch = Batch(
                batch_id=f"batch_{self._batch_counter}",
                requests=sorted(requests, key=lambda r: len(r.text)),
                reference_hash=None,
            )
            return [batch]

        # Group by reference hash
        groups: Dict[str, List[BatchRequest]] = {}
        for request in requests:
            ref_hash = request.reference_hash or "no_ref"
            if ref_hash not in groups:
                groups[ref_hash] = []
            groups[ref_hash].append(request)

        # Form batches from groups
        batches = []
        for ref_hash, group_requests in groups.items():
            # Sort by text length within group
            group_requests.sort(key=lambda r: len(r.text))

            # Split into max_batch_size chunks
            for i in range(0, len(group_requests), self.max_batch_size):
                chunk = group_requests[i:i + self.max_batch_size]
                self._batch_counter += 1
                batch = Batch(
                    batch_id=f"batch_{self._batch_counter}",
                    requests=chunk,
                    reference_hash=ref_hash,
                )
                batches.append(batch)

        return batches

    async def _process_batch(self, batch: Batch) -> None:
        """
        Process a single batch.

        Args:
            batch: Batch to process
        """
        self._total_batches += 1

        # Record wait times
        current_time = time.time()
        for request in batch.requests:
            wait_time = current_time - request.submit_time
            self._total_wait_time += wait_time

        logger.info(
            f"Processing {batch.batch_id}: size={batch.size}, "
            f"ref_hash={batch.reference_hash}, "
            f"text_lengths={batch.text_lengths}"
        )

        if self.process_callback is None:
            # No callback, fail all requests
            for request in batch.requests:
                if not request.future.done():
                    request.future.set_exception(
                        RuntimeError("No process_callback configured")
                    )
            return

        try:
            # Call the process callback
            results = await self.process_callback(batch)

            # Distribute results to individual requests
            if isinstance(results, list) and len(results) == len(batch.requests):
                for request, result in zip(batch.requests, results):
                    if not request.future.done():
                        request.future.set_result(result)
            else:
                # Single result for all (shouldn't happen normally)
                for request in batch.requests:
                    if not request.future.done():
                        request.future.set_result(results)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fail all requests in the batch
            for request in batch.requests:
                if not request.future.done():
                    request.future.set_exception(e)


class SyncBatchQueue:
    """
    Synchronous wrapper for BatchQueue for use in threaded contexts.

    This class provides a sync interface to the async BatchQueue,
    useful when integrating with sync code or running in threads.
    """

    def __init__(
        self,
        max_batch_size: int = 4,
        batch_timeout_ms: int = 200,
        enable_grouping: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.enable_grouping = enable_grouping

        self._async_queue: Optional[BatchQueue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[asyncio.AbstractEventLoop] = None

    def start(self, process_callback: Callable) -> None:
        """Start the batch queue in a new event loop."""
        import threading

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            self._async_queue = BatchQueue(
                max_batch_size=self.max_batch_size,
                batch_timeout_ms=self.batch_timeout_ms,
                enable_grouping=self.enable_grouping,
                process_callback=process_callback,
            )

            self._loop.run_until_complete(self._async_queue.start())
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Wait for loop to be ready
        import time
        while self._loop is None or self._async_queue is None:
            time.sleep(0.01)

    def submit(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
        reference_id: Optional[str] = None,
        timeout: float = 60.0,
        **parameters,
    ) -> Any:
        """
        Submit a request synchronously and wait for result.

        Args:
            text: Text to synthesize
            reference_audio: Optional reference audio
            reference_text: Text in reference audio
            reference_id: Pre-registered reference ID
            timeout: Maximum time to wait for result
            **parameters: Generation parameters

        Returns:
            Audio result
        """
        if self._loop is None or self._async_queue is None:
            raise RuntimeError("SyncBatchQueue not started")

        import concurrent.futures

        async def submit_and_wait():
            future = await self._async_queue.submit(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                reference_id=reference_id,
                **parameters,
            )
            return await asyncio.wait_for(future, timeout=timeout)

        # Submit to the event loop
        future = asyncio.run_coroutine_threadsafe(submit_and_wait(), self._loop)
        return future.result(timeout=timeout + 1)

    def stop(self) -> None:
        """Stop the batch queue."""
        if self._loop is not None and self._async_queue is not None:
            asyncio.run_coroutine_threadsafe(
                self._async_queue.stop(),
                self._loop,
            ).result(timeout=5.0)
            self._loop.call_soon_threadsafe(self._loop.stop)

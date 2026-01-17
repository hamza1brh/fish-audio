"""
S1-Mini Batch Worker
====================

This module provides a worker thread for processing batched TTS requests.
It receives batches of requests and processes them using the batched inference
functions.

Architecture:
-------------
    BatchedModelWorker
    ├── Input Queue (batch requests)
    ├── Model (DualARTransformer with batched KV cache)
    ├── Batched Inference
    └── Output Distribution (per-request response queues)

Usage:
------
    worker = BatchedModelWorker(
        model=model,
        decode_one_token=decode_fn,
        device="cuda",
        max_batch_size=4,
    )
    worker.start()

    # Submit batch
    response = worker.process_batch(batch_request)

    worker.stop()
"""

import gc
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from loguru import logger

from s1_mini.batch_queue import Batch, BatchRequest


@dataclass
class BatchGenerateRequest:
    """
    Request for batched generation.

    Attributes:
        batch: Batch of requests to process
        response_queues: Map of request_id -> response queue
    """
    batch: Batch
    response_queues: Dict[str, queue.Queue] = field(default_factory=dict)


@dataclass
class BatchGenerateResult:
    """
    Result from batched generation.

    Attributes:
        request_id: ID of the original request
        status: "success" or "error"
        codes: Generated VQ codes (if success)
        error: Exception (if error)
    """
    request_id: str
    status: str
    codes: Optional[torch.Tensor] = None
    error: Optional[Exception] = None


class BatchedModelWorker:
    """
    Worker thread for processing batched TTS requests.

    This class manages a background thread that:
    1. Receives batched generation requests
    2. Runs batched inference
    3. Distributes results to individual response queues

    Attributes:
        model: DualARTransformer model
        decode_one_token: Decode function
        device: Target device
        max_batch_size: Maximum batch size
        input_queue: Queue for receiving batch requests
    """

    def __init__(
        self,
        model: Any,
        decode_one_token: Callable,
        device: str,
        max_batch_size: int = 4,
    ):
        """
        Initialize the batch worker.

        Args:
            model: Pre-loaded DualARTransformer model
            decode_one_token: Decode function (possibly compiled)
            device: Target device
            max_batch_size: Maximum supported batch size
        """
        self.model = model
        self.decode_one_token = decode_one_token
        self.device = device
        self.max_batch_size = max_batch_size

        self.input_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._init_event = threading.Event()

        # Statistics
        self._total_batches = 0
        self._total_requests = 0
        self._total_errors = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the worker thread."""
        if self._running:
            logger.warning("BatchedModelWorker already running")
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="BatchedModelWorker",
        )
        self._worker_thread.start()

        # Wait for initialization
        if not self._init_event.wait(timeout=60):
            raise RuntimeError("BatchedModelWorker initialization timed out")

        logger.info("BatchedModelWorker started")

    def stop(self) -> None:
        """Stop the worker thread."""
        if not self._running:
            return

        self._running = False
        self.input_queue.put(None)  # Signal shutdown

        if self._worker_thread:
            self._worker_thread.join(timeout=10)

        logger.info("BatchedModelWorker stopped")

    def process_batch(
        self,
        batch: Batch,
        response_queues: Dict[str, queue.Queue],
    ) -> None:
        """
        Submit a batch for processing.

        Args:
            batch: Batch of requests to process
            response_queues: Map of request_id -> response queue
        """
        if not self._running:
            raise RuntimeError("BatchedModelWorker is not running")

        request = BatchGenerateRequest(
            batch=batch,
            response_queues=response_queues,
        )
        self.input_queue.put(request)

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "running": self._running,
        }

    def _worker_loop(self) -> None:
        """Main worker loop that processes batches."""
        from fish_speech.models.text2semantic.inference_batched import (
            generate_long_batched,
        )

        # Set up KV cache for batched inference
        try:
            with torch.device(self.device):
                self.model.setup_caches(
                    max_batch_size=self.max_batch_size,
                    max_seq_len=self.model.config.max_seq_len,
                    dtype=next(self.model.parameters()).dtype,
                )
            self.model._cache_setup_done = True
            logger.info(f"KV cache initialized for batch_size={self.max_batch_size}")
        except Exception as e:
            logger.error(f"Failed to initialize KV cache: {e}")
            self._running = False
            self._init_event.set()
            return

        self._init_event.set()

        while self._running:
            try:
                # Get next batch request
                item = self.input_queue.get(timeout=1.0)

                if item is None:
                    # Shutdown signal
                    break

                if not isinstance(item, BatchGenerateRequest):
                    logger.warning(f"Unexpected item type: {type(item)}")
                    continue

                self._process_batch_request(item, generate_long_batched)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                traceback.print_exc()

        logger.info("Worker loop exited")

    def _process_batch_request(
        self,
        request: BatchGenerateRequest,
        generate_fn: Callable,
    ) -> None:
        """
        Process a single batch request.

        Args:
            request: Batch request to process
            generate_fn: Batched generation function
        """
        batch = request.batch
        response_queues = request.response_queues

        self._total_batches += 1
        self._total_requests += batch.size

        logger.info(
            f"Processing batch {batch.batch_id}: size={batch.size}, "
            f"texts={[len(t) for t in batch.texts]}"
        )

        try:
            # Prepare inputs for batched generation
            texts = []
            prompt_texts_list = []
            prompt_tokens_list = []
            parameters_list = []

            for req in batch.requests:
                texts.append(req.text)

                # Get prompt tokens and texts from parameters
                params = req.parameters
                prompt_tokens_list.append(params.get("prompt_tokens"))
                prompt_texts_list.append(params.get("prompt_text"))
                parameters_list.append(params)

            # Use first request's parameters as defaults
            first_params = parameters_list[0] if parameters_list else {}

            # Run batched generation
            results = list(generate_fn(
                model=self.model,
                device=self.device,
                decode_one_token=self.decode_one_token,
                texts=texts,
                max_new_tokens=first_params.get("max_new_tokens", 2048),
                top_p=first_params.get("top_p", 0.8),
                repetition_penalty=first_params.get("repetition_penalty", 1.1),
                temperature=first_params.get("temperature", 0.7),
                compile=first_params.get("compile", False),
                chunk_length=first_params.get("chunk_length", 512),
                prompt_texts=prompt_texts_list,
                prompt_tokens=prompt_tokens_list,
            ))

            # Distribute results to response queues
            # Filter out "next" action results
            sample_results = [r for r in results if r.action == "sample"]

            for result in sample_results:
                batch_idx = result.batch_idx
                req = batch.requests[batch_idx]
                req_id = req.request_id

                if req_id in response_queues:
                    response_queues[req_id].put(
                        BatchGenerateResult(
                            request_id=req_id,
                            status="success",
                            codes=result.codes,
                        )
                    )

            # Signal completion for all requests
            for req in batch.requests:
                if req.request_id in response_queues:
                    response_queues[req.request_id].put(
                        BatchGenerateResult(
                            request_id=req.request_id,
                            status="done",
                        )
                    )

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM during batch processing: {e}")
            self._total_errors += batch.size

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Fail all requests in the batch
            for req in batch.requests:
                if req.request_id in response_queues:
                    response_queues[req.request_id].put(
                        BatchGenerateResult(
                            request_id=req.request_id,
                            status="error",
                            error=e,
                        )
                    )

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            traceback.print_exc()
            self._total_errors += batch.size

            # Fail all requests in the batch
            for req in batch.requests:
                if req.request_id in response_queues:
                    response_queues[req.request_id].put(
                        BatchGenerateResult(
                            request_id=req.request_id,
                            status="error",
                            error=e,
                        )
                    )


def create_batched_model_worker(
    model: Any,
    decode_one_token: Callable,
    device: str,
    max_batch_size: int = 4,
) -> BatchedModelWorker:
    """
    Create and start a batched model worker.

    Args:
        model: Pre-loaded DualARTransformer model
        decode_one_token: Decode function
        device: Target device
        max_batch_size: Maximum batch size

    Returns:
        Started BatchedModelWorker instance
    """
    worker = BatchedModelWorker(
        model=model,
        decode_one_token=decode_one_token,
        device=device,
        max_batch_size=max_batch_size,
    )
    worker.start()
    return worker

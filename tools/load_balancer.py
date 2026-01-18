"""
Load Balancer for Fish Speech Multi-Worker Setup

Distributes incoming requests across multiple worker processes using
round-robin or least-connections algorithm.

Usage:
    python tools/load_balancer.py --port 8000 --workers 8001,8002,8003
    python tools/load_balancer.py --port 8000 --workers 8001,8002 --strategy least-connections

Endpoints:
    POST /v1/tts      - Text-to-speech (proxied to workers)
    GET  /v1/health   - Health check for all workers
    GET  /v1/workers  - List worker status
"""

import argparse
import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger


class LoadBalancer:
    """Simple load balancer with round-robin or least-connections strategy."""

    def __init__(
        self,
        worker_ports: list[int],
        strategy: Literal["round-robin", "least-connections"] = "round-robin",
        host: str = "localhost",
    ):
        self.workers = [f"http://{host}:{port}" for port in worker_ports]
        self.strategy = strategy
        self.current_index = 0
        self.active_connections = defaultdict(int)
        self.worker_health = {w: True for w in self.workers}
        self.request_count = defaultdict(int)
        self.lock = asyncio.Lock()

    async def get_next_worker(self) -> str:
        """Get the next worker to use based on strategy."""
        async with self.lock:
            healthy_workers = [w for w in self.workers if self.worker_health[w]]

            if not healthy_workers:
                raise RuntimeError("No healthy workers available")

            if self.strategy == "round-robin":
                # Simple round-robin
                worker = healthy_workers[self.current_index % len(healthy_workers)]
                self.current_index += 1
            else:
                # Least connections
                worker = min(healthy_workers, key=lambda w: self.active_connections[w])

            self.active_connections[worker] += 1
            self.request_count[worker] += 1
            return worker

    async def release_worker(self, worker: str):
        """Release a worker connection."""
        async with self.lock:
            self.active_connections[worker] = max(0, self.active_connections[worker] - 1)

    async def check_health(self) -> dict:
        """Check health of all workers."""
        results = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for worker in self.workers:
                try:
                    response = await client.get(f"{worker}/v1/health")
                    healthy = response.status_code == 200
                    self.worker_health[worker] = healthy
                    results[worker] = {
                        "healthy": healthy,
                        "status_code": response.status_code,
                        "active_connections": self.active_connections[worker],
                        "total_requests": self.request_count[worker],
                    }
                except Exception as e:
                    self.worker_health[worker] = False
                    results[worker] = {
                        "healthy": False,
                        "error": str(e),
                        "active_connections": self.active_connections[worker],
                        "total_requests": self.request_count[worker],
                    }
        return results

    def get_stats(self) -> dict:
        """Get current load balancer statistics."""
        return {
            "strategy": self.strategy,
            "workers": [
                {
                    "url": w,
                    "healthy": self.worker_health[w],
                    "active_connections": self.active_connections[w],
                    "total_requests": self.request_count[w],
                }
                for w in self.workers
            ],
            "total_requests": sum(self.request_count.values()),
        }


# Global load balancer instance
lb: LoadBalancer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(f"Load balancer starting with {len(lb.workers)} workers")
    logger.info(f"Strategy: {lb.strategy}")
    logger.info(f"Workers: {lb.workers}")

    # Initial health check
    health = await lb.check_health()
    healthy_count = sum(1 for w in health.values() if w.get("healthy", False))
    logger.info(f"Initial health check: {healthy_count}/{len(lb.workers)} workers healthy")

    yield

    logger.info("Load balancer shutting down")


app = FastAPI(
    title="Fish Speech Load Balancer",
    description="Distributes TTS requests across multiple workers",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "name": "Fish Speech Load Balancer",
        "endpoints": {
            "/v1/tts": "POST - Text-to-speech (proxied)",
            "/v1/health": "GET - Health check all workers",
            "/v1/workers": "GET - Worker statistics",
        },
    }


@app.get("/v1/health")
async def health_check():
    """Check health of all workers."""
    health = await lb.check_health()
    healthy_count = sum(1 for w in health.values() if w.get("healthy", False))

    return {
        "status": "healthy" if healthy_count > 0 else "unhealthy",
        "healthy_workers": healthy_count,
        "total_workers": len(lb.workers),
        "workers": health,
    }


@app.get("/v1/workers")
async def worker_stats():
    """Get worker statistics."""
    return lb.get_stats()


@app.api_route("/v1/tts", methods=["POST"])
async def proxy_tts(request: Request):
    """Proxy TTS requests to workers."""
    worker = None
    try:
        worker = await lb.get_next_worker()
        logger.debug(f"Routing request to {worker}")

        # Read request body
        body = await request.body()

        # Forward headers
        headers = dict(request.headers)
        headers.pop("host", None)  # Remove host header

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Forward the request
            response = await client.post(
                f"{worker}/v1/tts",
                content=body,
                headers=headers,
            )

            # Check if streaming response
            content_type = response.headers.get("content-type", "")

            if "audio" in content_type or "octet-stream" in content_type:
                # Return audio directly
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    media_type=content_type,
                )
            else:
                # Return JSON response
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    media_type=content_type,
                )

    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={"error": "No healthy workers available", "detail": str(e)},
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": "Worker timeout", "worker": worker},
        )
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": "Bad gateway", "detail": str(e)},
        )
    finally:
        if worker:
            await lb.release_worker(worker)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_other(path: str, request: Request):
    """Proxy any other requests to a worker."""
    worker = None
    try:
        worker = await lb.get_next_worker()

        body = await request.body() if request.method in ["POST", "PUT"] else None

        headers = dict(request.headers)
        headers.pop("host", None)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.request(
                method=request.method,
                url=f"{worker}/{path}",
                content=body,
                headers=headers,
                params=dict(request.query_params),
            )

            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
            )

    except Exception as e:
        logger.error(f"Proxy error for /{path}: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": "Bad gateway", "detail": str(e)},
        )
    finally:
        if worker:
            await lb.release_worker(worker)


def parse_args():
    parser = argparse.ArgumentParser(description="Fish Speech Load Balancer")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=str,
        required=True,
        help="Comma-separated list of worker ports (e.g., 8001,8002,8003)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Worker host (default: localhost)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["round-robin", "least-connections"],
        default="least-connections",
        help="Load balancing strategy (default: least-connections)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    worker_ports = [int(p.strip()) for p in args.workers.split(",")]

    lb = LoadBalancer(
        worker_ports=worker_ports,
        strategy=args.strategy,
        host=args.host,
    )

    print("=" * 60)
    print("Fish Speech Load Balancer")
    print("=" * 60)
    print(f"Listening on: http://0.0.0.0:{args.port}")
    print(f"Strategy: {args.strategy}")
    print(f"Workers: {lb.workers}")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )

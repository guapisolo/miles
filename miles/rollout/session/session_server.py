"""Standalone Session Server that proxies through the inference router.

This decouples session/TITO logic from the Miles Router, allowing sessions
to work with the SGLang Rust Router or any other backend.  Inference
requests are proxied through the router (sglang or miles), which handles
load balancing and forwarding to worker engines.
"""

import asyncio
import json
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.sessions import setup_session_routes

logger = logging.getLogger(__name__)


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, args, backend_url: str):
        self.backend_url = backend_url
        self.app = FastAPI()

        timeout = getattr(args, "miles_router_timeout", None)
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024),
            timeout=httpx.Timeout(timeout),
        )

        # Close the httpx connection pool when uvicorn shuts down to avoid FD leaks.
        self.app.add_event_handler("shutdown", self.client.aclose)

        # Abort/resume gate for weight-update pauses.
        # When set (default), requests flow through; when cleared, handlers block.
        self._resume_event: asyncio.Event | None = None  # lazy-init (needs event loop)

        setup_session_routes(self.app, self, args)

    @property
    def resume_event(self) -> asyncio.Event:
        if self._resume_event is None:
            self._resume_event = asyncio.Event()
            self._resume_event.set()
        return self._resume_event

    def is_paused(self) -> bool:
        return self._resume_event is not None and not self._resume_event.is_set()

    async def pause_sessions(self) -> None:
        """Pause all sessions: block new proxy calls, abort in-flight SGLang requests."""
        self.resume_event.clear()
        try:
            await self.client.post(
                f"{self.backend_url}/abort_request",
                json={"abort_all": True},
            )
        except Exception:
            logger.warning("Failed to send abort_request to backend", exc_info=True)

    async def resume_sessions(self) -> None:
        """Resume all paused sessions: unblock waiting handlers.

        Polls the backend /health endpoint first to ensure SGLang is ready
        to serve requests after a weight update.
        """
        while True:
            try:
                resp = await self.client.get(f"{self.backend_url}/health")
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        logger.info("Backend health check passed, resuming sessions")
        self.resume_event.set()

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        url = f"{self.backend_url}/{path}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}

        response = await self.client.request(request.method, url, content=body, headers=headers)
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

    def build_proxy_response(self, result: dict) -> Response:
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        content_type = headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=status_code, headers=headers)
        except Exception:
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)


def run_session_server(args, backend_url: str):
    """Entry point to start the standalone session server as a subprocess."""
    server = SessionServer(args, backend_url)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        args.session_server_ip,
        args.session_server_port,
        backend_url,
    )
    uvicorn.run(server.app, host=args.session_server_ip, port=args.session_server_port, log_level="info")

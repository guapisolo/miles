import asyncio
import logging
import time
from typing import Any

import httpx
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
_HTTP_CLIENT: httpx.AsyncClient | None = None
_HTTP_BASE_URL: str | None = None
_HTTP_TIMEOUT_SECONDS = 60.0
_HTTP_MAX_RETRIES = 60
_HTTP_RETRY_INTERVAL_SECONDS = 1.0
_HTTP_CLIENT_LOCK = asyncio.Lock()
_HTTP_CLIENT_CONCURRENCY = 2048


async def _get_http_client(base_url: str) -> httpx.AsyncClient:
    global _HTTP_CLIENT, _HTTP_BASE_URL
    async with _HTTP_CLIENT_LOCK:
        if _HTTP_CLIENT is None or _HTTP_BASE_URL != base_url:
            if _HTTP_CLIENT is not None:
                await _HTTP_CLIENT.aclose()
            _HTTP_BASE_URL = base_url
            _HTTP_CLIENT = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=_HTTP_CLIENT_CONCURRENCY),
                timeout=httpx.Timeout(None),
            )
        return _HTTP_CLIENT


async def post(url: str, payload: dict[str, Any]) -> JSONResponse:
    client = await _get_http_client(url)
    response = None
    for attempt in range(1, _HTTP_MAX_RETRIES + 1):
        try:
            http_response = await client.post(url, json=payload)
            http_response.raise_for_status()
            response = http_response.json()
            break
        except httpx.RequestError:
            if attempt == _HTTP_MAX_RETRIES:
                raise
            logger.warning("OpenAI API error, retrying... (attempt %s/%s)", attempt, _HTTP_MAX_RETRIES)
            time.sleep(_HTTP_RETRY_INTERVAL_SECONDS)
    return response

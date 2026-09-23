"""Optional service-token authentication shared with the Tarebar dashboard.

The dashboard signs a short-lived compact JWS (HS256) with ``SERVICE_AUTH_SECRET``
and sends it as ``Authorization: Bearer <token>`` or, for ``EventSource``,
``<img>``, ``<video>`` and download links that cannot set headers, as the
``access_token`` query parameter. When the secret is unset nothing is enforced.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from typing import Any, Mapping
from urllib.parse import parse_qs

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


SERVICE_AUTH_SECRET_VARIABLE = "SERVICE_AUTH_SECRET"
TOKEN_ISSUER = "tarebar"
TOKEN_ALGORITHM = "HS256"
EXPIRY_LEEWAY_SECONDS = 30
UNAUTHORIZED_DETAIL = "invalid or missing service token"

PROTECTED_PREFIX = "/api/v1/"
# These routers keep their own ``X-Analytics-Key`` service-to-service check.
KEY_AUTHENTICATED_PREFIXES = ("/api/v1/management/", "/api/v1/ingest/")


def _decode_segment(segment: str) -> bytes:
    """Decode unpadded base64url, rejecting characters outside the alphabet."""

    padded = segment + "=" * (-len(segment) % 4)
    return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)


def verify_service_token(
    token: str | None, secret: str | None, now: float | None = None
) -> dict[str, Any] | None:
    """Return the token claims when the token is valid, otherwise ``None``."""

    if not token or not secret or not isinstance(token, str):
        return None
    parts = token.split(".")
    if len(parts) != 3 or not all(parts):
        return None
    header_segment, payload_segment, signature_segment = parts
    try:
        header = json.loads(_decode_segment(header_segment))
        signature = _decode_segment(signature_segment)
    except (ValueError, UnicodeError, binascii.Error):
        return None
    if not isinstance(header, dict) or header.get("alg") != TOKEN_ALGORITHM:
        return None
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{header_segment}.{payload_segment}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        claims = json.loads(_decode_segment(payload_segment))
    except (ValueError, UnicodeError, binascii.Error):
        return None
    if not isinstance(claims, dict) or claims.get("iss") != TOKEN_ISSUER:
        return None
    expires_at = claims.get("exp")
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return None
    current = time.time() if now is None else now
    if current > float(expires_at) + EXPIRY_LEEWAY_SECONDS:
        return None
    return claims


def requires_service_token(method: str, path: str) -> bool:
    """Whether a request must carry a service token once a secret is configured."""

    if method.upper() == "OPTIONS":
        return False
    if not path.startswith(PROTECTED_PREFIX):
        return False
    return not path.startswith(KEY_AUTHENTICATED_PREFIXES)


def token_from_request(headers: Mapping[str, str], query_string: str) -> str | None:
    """Read the bearer header first, then the ``access_token`` query parameter."""

    authorization = headers.get("authorization", "")
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() == "bearer" and credentials.strip():
        return credentials.strip()
    values = parse_qs(query_string).get("access_token")
    return values[0] if values and values[0] else None


class ServiceTokenMiddleware:
    """Pure ASGI middleware so SSE and MJPEG responses keep streaming untouched."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Read per request: deployments rotate the secret with a restart, tests toggle it.
        secret = (os.environ.get(SERVICE_AUTH_SECRET_VARIABLE) or "").strip()
        if not secret or not requires_service_token(scope["method"], scope["path"]):
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        query_string = scope.get("query_string", b"").decode("latin-1")
        token = token_from_request(headers, query_string)
        if verify_service_token(token, secret) is None:
            response = JSONResponse(
                {"detail": UNAUTHORIZED_DETAIL},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

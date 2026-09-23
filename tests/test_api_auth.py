import asyncio
import base64
import hashlib
import hmac
import json

import pytest
from fastapi.middleware.cors import CORSMiddleware

from app.api import main
from app.api.auth import (
    ServiceTokenMiddleware,
    requires_service_token,
    token_from_request,
    verify_service_token,
)


SECRET = "shared-service-secret"
NOW = 1_800_000_000


def _segment(value: object) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(value, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _token(
    claims: dict[str, object] | None = None,
    *,
    secret: str = SECRET,
    header: dict[str, object] | None = None,
) -> str:
    """Compact JWS exactly as the dashboard signs it (unpadded base64url, HS256)."""

    payload = {"iss": "tarebar", "sub": "user-1", "role": "ADMIN", "iat": NOW, "exp": NOW + 900}
    payload.update(claims or {})
    signing_input = f"{_segment(header or {'alg': 'HS256', 'typ': 'JWT'})}.{_segment(payload)}"
    signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_segment(signature)}"


def test_valid_token_returns_its_claims() -> None:
    claims = verify_service_token(_token(), SECRET, now=NOW)

    assert claims is not None
    assert claims["sub"] == "user-1" and claims["role"] == "ADMIN"


def test_expired_token_is_rejected_after_the_leeway() -> None:
    token = _token({"exp": NOW})

    assert verify_service_token(token, SECRET, now=NOW + 30) is not None
    assert verify_service_token(token, SECRET, now=NOW + 31) is None


def test_token_without_a_numeric_expiry_is_rejected() -> None:
    assert verify_service_token(_token({"exp": None}), SECRET, now=NOW) is None
    assert verify_service_token(_token({"exp": "tomorrow"}), SECRET, now=NOW) is None
    assert verify_service_token(_token({"exp": True}), SECRET, now=NOW) is None


def test_bad_signature_is_rejected() -> None:
    assert verify_service_token(_token(secret="another-secret"), SECRET, now=NOW) is None
    header, payload, signature = _token().split(".")
    forged_payload = _segment({"iss": "tarebar", "sub": "attacker", "role": "ADMIN", "exp": NOW + 900})
    assert verify_service_token(f"{header}.{forged_payload}.{signature}", SECRET, now=NOW) is None
    assert verify_service_token(f"{header}.{payload}.{_segment(b'short')}", SECRET, now=NOW) is None


@pytest.mark.parametrize("algorithm", ["none", "HS512", "RS256", "hs256", None])
def test_wrong_algorithm_is_rejected_even_with_a_matching_signature(algorithm) -> None:
    token = _token(header={"alg": algorithm, "typ": "JWT"})

    assert verify_service_token(token, SECRET, now=NOW) is None


def test_wrong_issuer_is_rejected() -> None:
    assert verify_service_token(_token({"iss": "someone-else"}), SECRET, now=NOW) is None
    assert verify_service_token(_token({"iss": None}), SECRET, now=NOW) is None


@pytest.mark.parametrize(
    "garbage",
    [
        None, "", "garbage", "a.b", "a.b.c", "a.b.c.d", "..", "é.é.é", "e30.e30.e30",
        "!!!.###.$$$", _segment(["HS256"]) + "." + _segment({}) + "." + _segment(b"x"),
    ],
)
def test_garbage_is_rejected_without_raising(garbage) -> None:
    assert verify_service_token(garbage, SECRET, now=NOW) is None


def test_missing_secret_never_validates_a_token() -> None:
    assert verify_service_token(_token(secret=""), "", now=NOW) is None
    assert verify_service_token(_token(), None, now=NOW) is None


@pytest.mark.parametrize(
    ("method", "path", "protected"),
    [
        ("GET", "/api/v1/jobs", True),
        ("POST", "/api/v1/jobs", True),
        ("GET", "/api/v1/jobs/abc/events", True),
        ("GET", "/api/v1/jobs/abc/export.zip", True),
        ("POST", "/api/v1/fruit-quality", True),
        ("GET", "/api/v1/fleet/status", True),
        ("GET", "/api/v1/management-extra", True),
        ("GET", "/api/v1/management/overview", False),
        ("POST", "/api/v1/ingest/minutes", False),
        ("OPTIONS", "/api/v1/jobs", False),
        ("options", "/api/v1/jobs", False),
        ("GET", "/health", False),
        ("GET", "/docs", False),
        ("GET", "/openapi.json", False),
        ("GET", "/redoc", False),
        ("GET", "/", False),
    ],
)
def test_path_exemptions(method: str, path: str, protected: bool) -> None:
    assert requires_service_token(method, path) is protected


def test_token_is_read_from_the_bearer_header_or_the_query_string() -> None:
    assert token_from_request({"authorization": "Bearer abc"}, "") == "abc"
    assert token_from_request({"authorization": "bearer  abc "}, "access_token=query") == "abc"
    assert token_from_request({}, "after=3&access_token=a.b.c") == "a.b.c"
    assert token_from_request({"authorization": "Basic abc"}, "") is None
    assert token_from_request({}, "access_token=") is None


# --- the real application, driven over raw ASGI (httpx is not installed) ------


def _request(
    method: str, path: str, *, headers: dict[str, str] | None = None, query: str = ""
) -> tuple[int, dict[str, str], bytes]:
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "",
        "query_string": query.encode(), "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
        "headers": [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()],
    }
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    asyncio.run(main.app(scope, receive, send))
    start = next(item for item in messages if item["type"] == "http.response.start")
    body = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")  # type: ignore[misc]
    response_headers = {key.decode(): value.decode() for key, value in start["headers"]}  # type: ignore[union-attr]
    return int(start["status"]), response_headers, body  # type: ignore[arg-type]


ORIGIN = main.origins[0]


def test_cors_is_the_outermost_user_middleware() -> None:
    # Starlette keeps user middleware outermost-first.
    assert [item.cls for item in main.app.user_middleware] == [CORSMiddleware, ServiceTokenMiddleware]


def test_nothing_is_enforced_without_a_secret(monkeypatch) -> None:
    monkeypatch.delenv("SERVICE_AUTH_SECRET", raising=False)
    assert _request("GET", "/api/v1/applications")[0] == 200
    monkeypatch.setenv("SERVICE_AUTH_SECRET", "")
    assert _request("GET", "/api/v1/applications")[0] == 200


def test_missing_token_is_a_401_that_still_carries_cors_headers(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_AUTH_SECRET", SECRET)

    status, headers, body = _request("GET", "/api/v1/applications", headers={"Origin": ORIGIN})

    assert status == 401
    assert json.loads(body) == {"detail": "invalid or missing service token"}
    assert headers["access-control-allow-origin"] == ORIGIN


def test_invalid_token_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_AUTH_SECRET", SECRET)
    stale = _token({"exp": 1_000_000_000})

    assert _request("GET", "/api/v1/trackers", headers={"Authorization": f"Bearer {stale}"})[0] == 401
    assert _request("GET", "/api/v1/trackers", query="access_token=" + _token(secret="x"))[0] == 401


def test_valid_token_is_accepted_from_header_and_query(monkeypatch) -> None:
    import time

    monkeypatch.setenv("SERVICE_AUTH_SECRET", SECRET)
    token = _token({"iat": int(time.time()), "exp": int(time.time()) + 900})

    status, headers, body = _request(
        "GET", "/api/v1/trackers", headers={"Authorization": f"Bearer {token}", "Origin": ORIGIN}
    )
    assert status == 200 and "data" in json.loads(body)
    assert headers["access-control-allow-origin"] == ORIGIN
    assert _request("GET", "/api/v1/trackers", query=f"access_token={token}")[0] == 200


def test_preflight_health_and_key_authenticated_routes_stay_exempt(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_AUTH_SECRET", SECRET)
    monkeypatch.setenv("ANALYTICS_READ_KEY", "read-key")

    status, headers, _body = _request("OPTIONS", "/api/v1/jobs", headers={
        "Origin": ORIGIN, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    })
    assert status == 200
    assert headers["access-control-allow-origin"] == ORIGIN
    assert "authorization" in headers["access-control-allow-headers"].lower()

    status, _headers, body = _request("GET", "/health")
    assert status == 200 and json.loads(body)["status"] in {"ok", "degraded"}

    # The management router keeps its own X-Analytics-Key check.
    status, _headers, body = _request(
        "GET", "/api/v1/management/overview", query="from=2026-09-01&to=2026-09-02"
    )
    assert status == 401 and json.loads(body) == {"detail": "invalid analytics key"}

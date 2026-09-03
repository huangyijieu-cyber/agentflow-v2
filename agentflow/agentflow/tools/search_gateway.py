"""Client for the AgentFlow search egress gateway.

All public web I/O used by the enabled search tools must go through this
client.  In particular, the session deliberately ignores HTTP(S)_PROXY from
the training host so that the request to EC2 cannot be redirected back to the
old, unstable egress proxy.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

import requests


DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 60.0
# Retry belongs in one layer only.  The Windows Gateway already owns upstream
# retry policy, so training-side retries are opt-in to avoid multiplying the
# several sequential requests made by Wikipedia RAG.
DEFAULT_MAX_RETRIES = 0
DEFAULT_RETRY_BACKOFF = 0.5
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


class SearchGatewayError(RuntimeError):
    """A sanitized error returned by, or raised while reaching, the gateway."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "search_gateway_error",
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        retryable: bool = False,
        upstream_status: Optional[int] = None,
        retryable_was_explicit: bool = False,
    ) -> None:
        self.message = _sanitize_message(message)
        self.code = str(code or "search_gateway_error")
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable
        self.upstream_status = upstream_status
        self.retryable_was_explicit = retryable_was_explicit
        super().__init__(self.__str__())

    def __str__(self) -> str:
        details = [self.code]
        if self.status_code is not None:
            details.append(f"HTTP {self.status_code}")
        if self.request_id:
            details.append(f"request_id={self.request_id}")
        return f"{self.message} ({', '.join(details)})"


class SearchGatewayConfigurationError(SearchGatewayError):
    """The training process is missing a safe gateway configuration."""


def _sanitize_message(value: Any) -> str:
    """Keep gateway errors useful without copying arbitrary response bodies."""
    message = " ".join(str(value or "Search gateway request failed").split())
    return message[:500]


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SearchGatewayConfigurationError(
            f"{name} must be a number",
            code="invalid_gateway_configuration",
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise SearchGatewayConfigurationError(
            f"{name} must be greater than zero",
            code="invalid_gateway_configuration",
        )
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SearchGatewayConfigurationError(
            f"{name} must be an integer",
            code="invalid_gateway_configuration",
        ) from exc
    if parsed < 0:
        raise SearchGatewayConfigurationError(
            f"{name} must be zero or greater",
            code="invalid_gateway_configuration",
        )
    return parsed


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _bounded_request_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SearchGatewayError(
            f"{name} must be an integer",
            code="invalid_request",
        ) from exc
    if parsed < minimum or parsed > maximum:
        raise SearchGatewayError(
            f"{name} must be between {minimum} and {maximum}",
            code="invalid_request",
        )
    return parsed


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


class SearchGatewayClient:
    """Small, fail-closed HTTP client for the Windows Search Gateway."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        ca_bundle: Optional[str] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = self._validate_base_url(base_url)
        self.token = self._validate_token(token)
        self.connect_timeout = _positive_float(
            connect_timeout, "SEARCH_GATEWAY_CONNECT_TIMEOUT"
        )
        self.read_timeout = _positive_float(
            read_timeout, "SEARCH_GATEWAY_READ_TIMEOUT"
        )
        self.max_retries = _non_negative_int(
            max_retries, "SEARCH_GATEWAY_MAX_RETRIES"
        )
        self.retry_backoff = _positive_float(
            retry_backoff, "SEARCH_GATEWAY_RETRY_BACKOFF"
        )

        if ca_bundle:
            ca_bundle = os.path.abspath(os.path.expanduser(ca_bundle))
            if not os.path.isfile(ca_bundle) or not os.access(ca_bundle, os.R_OK):
                raise SearchGatewayConfigurationError(
                    "SEARCH_GATEWAY_CA_BUNDLE does not point to a readable file",
                    code="invalid_gateway_configuration",
                )
            self.verify: Any = ca_bundle
        else:
            self.verify = True

        self.session = session if session is not None else requests.Session()
        # This is the key isolation control: do not inherit the training
        # server's broken HTTP_PROXY / HTTPS_PROXY settings.
        self.session.trust_env = False

    @classmethod
    def from_env(
        cls,
        *,
        session: Optional[requests.Session] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "SearchGatewayClient":
        env = os.environ if environ is None else environ
        base_url = env.get("SEARCH_GATEWAY_BASE_URL", "")
        token = env.get("SEARCH_GATEWAY_TOKEN", "")

        if not base_url:
            raise SearchGatewayConfigurationError(
                "SEARCH_GATEWAY_BASE_URL is required",
                code="missing_gateway_configuration",
            )
        if not token:
            raise SearchGatewayConfigurationError(
                "SEARCH_GATEWAY_TOKEN is required",
                code="missing_gateway_configuration",
            )

        return cls(
            base_url=base_url,
            token=token,
            ca_bundle=env.get("SEARCH_GATEWAY_CA_BUNDLE") or None,
            connect_timeout=env.get(
                "SEARCH_GATEWAY_CONNECT_TIMEOUT", str(DEFAULT_CONNECT_TIMEOUT)
            ),
            read_timeout=env.get(
                "SEARCH_GATEWAY_READ_TIMEOUT", str(DEFAULT_READ_TIMEOUT)
            ),
            max_retries=env.get(
                "SEARCH_GATEWAY_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)
            ),
            retry_backoff=env.get(
                "SEARCH_GATEWAY_RETRY_BACKOFF", str(DEFAULT_RETRY_BACKOFF)
            ),
            session=session,
        )

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        value = str(base_url or "").strip().rstrip("/")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise SearchGatewayConfigurationError(
                "SEARCH_GATEWAY_BASE_URL must be an HTTP(S) URL without "
                "credentials, query parameters, or a fragment",
                code="invalid_gateway_configuration",
            )
        return value

    @staticmethod
    def _validate_token(token: str) -> str:
        raw_value = str(token or "")
        value = raw_value.strip()
        if (
            not value
            or raw_value != value
            or any(character.isspace() for character in value)
        ):
            raise SearchGatewayConfigurationError(
                "SEARCH_GATEWAY_TOKEN must be non-empty and contain no whitespace",
                code="invalid_gateway_configuration",
            )
        return value

    def _url(self, path: str) -> str:
        # Do not use urljoin here.  A leading slash would otherwise remove the
        # required /agentflow-search prefix from the EC2 base URL.
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")

    @staticmethod
    def _decode_error(response: requests.Response) -> SearchGatewayError:
        payload: Any = None
        try:
            payload = response.json()
        except (TypeError, ValueError):
            payload = None

        error_data: Dict[str, Any] = {}
        if isinstance(payload, dict):
            candidate = payload.get("error")
            if isinstance(candidate, dict):
                error_data = candidate
            else:
                detail = payload.get("detail")
                if isinstance(detail, dict):
                    nested = detail.get("error")
                    error_data = nested if isinstance(nested, dict) else detail

        retryable_is_explicit = "retryable" in error_data
        request_id = error_data.get("request_id")
        if not request_id:
            request_id = response.headers.get("X-Request-ID")

        return SearchGatewayError(
            error_data.get("message") or "Search gateway rejected the request",
            code=error_data.get("code") or f"gateway_http_{response.status_code}",
            status_code=response.status_code,
            request_id=str(request_id) if request_id else None,
            retryable=_as_bool(error_data.get("retryable", False)),
            upstream_status=_optional_int(error_data.get("upstream_status")),
            retryable_was_explicit=retryable_is_explicit,
        )

    def _retry_delay(
        self, attempt: int, response: Optional[requests.Response] = None
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    parsed_retry_after = float(retry_after)
                    if math.isfinite(parsed_retry_after):
                        return min(max(parsed_retry_after, 0.0), 5.0)
                except (TypeError, ValueError):
                    pass
        return min(self.retry_backoff * (2**attempt), 5.0)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = self._url(path)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "AgentFlow-SearchGatewayClient/1.0",
        }

        for attempt in range(self.max_retries + 1):
            request_kwargs: Dict[str, Any] = {
                "headers": headers,
                "timeout": (self.connect_timeout, self.read_timeout),
                "verify": self.verify,
            }
            if payload is not None:
                request_kwargs["json"] = payload

            try:
                response = self.session.request(
                    method.upper(), url, **request_kwargs
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise SearchGatewayError(
                    "Search gateway is unreachable or timed out",
                    code="gateway_unreachable",
                    retryable=True,
                ) from exc
            except requests.RequestException as exc:
                raise SearchGatewayError(
                    "Search gateway request could not be sent",
                    code="gateway_request_error",
                ) from exc

            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except (TypeError, ValueError) as exc:
                    raise SearchGatewayError(
                        "Search gateway returned a non-JSON success response",
                        code="invalid_gateway_response",
                        status_code=response.status_code,
                        request_id=response.headers.get("X-Request-ID"),
                    ) from exc

            error = self._decode_error(response)
            status_is_retryable = response.status_code in RETRYABLE_STATUS_CODES
            retry_allowed = status_is_retryable and (
                error.retryable or not error.retryable_was_explicit
            )
            if retry_allowed and attempt < self.max_retries:
                time.sleep(self._retry_delay(attempt, response))
                continue
            raise error

        raise SearchGatewayError(
            "Search gateway request failed",
            code="search_gateway_error",
        )

    def health(self) -> Dict[str, Any]:
        result = self._request("GET", "healthz")
        if not isinstance(result, dict):
            raise SearchGatewayError(
                "Search gateway health response must be a JSON object",
                code="invalid_gateway_response",
            )
        return result

    def fetch(self, url: str) -> Dict[str, Any]:
        target = str(url or "").strip()
        parsed = urlsplit(target)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SearchGatewayError(
                "fetch URL must be an absolute HTTP(S) URL",
                code="invalid_request",
            )

        result = self._request("POST", "v1/fetch", payload={"url": target})
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise SearchGatewayError(
                "Search gateway fetch response is missing text",
                code="invalid_gateway_response",
            )
        return result

    def wikipedia_search(
        self,
        query: str,
        *,
        max_pages: int = 10,
        max_length: int = 256,
        language: str = "en",
    ) -> Dict[str, Any]:
        payload = {
            "query": str(query or "").strip(),
            "max_pages": _bounded_request_int(
                max_pages, "max_pages", minimum=1, maximum=10
            ),
            "max_length": _bounded_request_int(
                max_length, "max_length", minimum=1, maximum=20000
            ),
            "language": str(language or "en"),
        }
        if not payload["query"]:
            raise SearchGatewayError(
                "Wikipedia query must not be empty", code="invalid_request"
            )

        result = self._request("POST", "v1/search/wikipedia", payload=payload)
        # Accept a bare list for compatibility with early Gateway builds, but
        # expose one stable client contract to the tool.
        if isinstance(result, list):
            result = {"results": result}
        elif isinstance(result, dict) and "results" not in result:
            # Some early builds named this collection "pages".
            pages = result.get("pages")
            if isinstance(pages, list):
                result = {**result, "results": pages}
        if not isinstance(result, dict) or not isinstance(
            result.get("results"), list
        ):
            raise SearchGatewayError(
                "Wikipedia gateway response is missing results",
                code="invalid_gateway_response",
            )
        return result

    def brave_search(
        self,
        query: str,
        *,
        count: int = 10,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query": str(query or "").strip(),
            "count": _bounded_request_int(
                count, "count", minimum=1, maximum=20
            ),
        }
        if not payload["query"]:
            raise SearchGatewayError(
                "Brave query must not be empty", code="invalid_request"
            )
        for key, value in (
            ("country", country),
            ("search_lang", search_lang),
            ("ui_lang", ui_lang),
            ("freshness", freshness),
        ):
            if value is not None:
                payload[key] = str(value)

        result = self._request("POST", "v1/search/brave", payload=payload)
        if not isinstance(result, dict):
            raise SearchGatewayError(
                "Brave gateway response must be a JSON object",
                code="invalid_gateway_response",
            )
        return result

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "SearchGatewayClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _health_check() -> int:
    try:
        with SearchGatewayClient.from_env() as client:
            response = client.health()
        if response.get("status") != "ok":
            raise SearchGatewayError(
                "Search gateway health status is not ok",
                code="unhealthy_gateway",
            )
    except SearchGatewayError as exc:
        print(f"Search gateway health check failed: {exc}", file=sys.stderr)
        return 1

    print("Search gateway health check passed.")
    return 0


def _readiness_check(environ: Optional[Mapping[str, str]] = None) -> int:
    """Verify both the tunnel and the Windows public egress before training."""
    env = os.environ if environ is None else environ
    smoke_url = env.get("SEARCH_GATEWAY_SMOKE_URL", "https://www.baidu.com/")
    wikipedia_query = env.get(
        "SEARCH_GATEWAY_SMOKE_WIKIPEDIA_QUERY", "Moon"
    )
    check_brave = _as_bool(env.get("SEARCH_GATEWAY_CHECK_BRAVE", "false"))
    brave_query = env.get("SEARCH_GATEWAY_SMOKE_BRAVE_QUERY", "Moon")

    try:
        with SearchGatewayClient.from_env(environ=env) as client:
            health = client.health()
            if health.get("status") != "ok":
                raise SearchGatewayError(
                    "Search gateway health status is not ok",
                    code="unhealthy_gateway",
                )

            fetched = client.fetch(smoke_url)
            if not fetched["text"].strip():
                raise SearchGatewayError(
                    "Search gateway readiness fetch returned empty text",
                    code="egress_not_ready",
                )

            wikipedia = client.wikipedia_search(
                wikipedia_query,
                max_pages=1,
                max_length=64,
                language="en",
            )
            if not wikipedia["results"]:
                raise SearchGatewayError(
                    "Search gateway readiness Wikipedia query returned no results",
                    code="egress_not_ready",
                )

            if check_brave:
                client.brave_search(brave_query, count=1)
    except SearchGatewayError as exc:
        print(f"Search gateway readiness check failed: {exc}", file=sys.stderr)
        return 1

    checked = "health, fetch, Wikipedia"
    if check_brave:
        checked += ", Brave"
    print(f"Search gateway readiness check passed ({checked}).")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AgentFlow Search Gateway checks")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--health-check",
        action="store_true",
        help="validate configuration and call GET /healthz",
    )
    checks.add_argument(
        "--readiness-check",
        action="store_true",
        help="also verify fetch and Wikipedia public egress",
    )
    args = parser.parse_args(argv)
    if args.health_check:
        return _health_check()
    if args.readiness_check:
        return _readiness_check()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Long-running network-path monitor for AgentFlow web tools.

The monitor reproduces the two outbound request patterns used by AgentFlow:

* Yibu/Brave search: GET, TLS verification disabled, 20 second timeout,
  and up to three attempts.
* Wikipedia search API: GET, TLS verification disabled, 10 second timeout,
  and one attempt so the monitor does not create its own 429 storm.
* Web fetch: browser-like headers, TLS verification disabled, and a 10
  second timeout.

Every actual HTTP attempt is written to JSONL and CSV.  Proxy credentials,
API keys, authorization headers, and URL query strings are never written to
the output.  The generated Markdown report keeps transport/proxy failures
separate from origin HTTP responses such as 403, 412, and 429.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import socket
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
import urllib3


SCHEMA_VERSION = "1.0"
DEFAULT_SEARCH_URL = "https://yibuapi.com/brave/v1/web/search"
DEFAULT_WIKIPEDIA_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_CONTROL_URL = "https://example.com/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)

NETWORK_FAILURE_CATEGORIES = {
    "PROXY_AUTH_407",
    "PROXY_TUNNEL_502",
    "PROXY_TUNNEL_503",
    "PROXY_TUNNEL_504",
    "PROXY_CONNECT_ERROR",
    "TLS_WRONG_VERSION",
    "TLS_CERTIFICATE_ERROR",
    "TLS_HANDSHAKE_TIMEOUT",
    "TLS_ERROR",
    "DNS_RESOLUTION_ERROR",
    "TCP_CONNECT_TIMEOUT",
    "READ_TIMEOUT",
    "TIMEOUT",
    "CONNECTION_REFUSED",
    "CONNECTION_RESET",
    "CONNECT_ERROR",
    "TRANSPORT_ERROR",
}

CONFIRMED_PROXY_CATEGORIES = {
    "PROXY_AUTH_407",
    "PROXY_TUNNEL_502",
    "PROXY_TUNNEL_503",
    "PROXY_TUNNEL_504",
    "PROXY_CONNECT_ERROR",
    "PROXY_GATEWAY_502",
    "PROXY_GATEWAY_503",
    "PROXY_GATEWAY_504",
}

NETWORK_FAILURE_CATEGORIES.update(
    {"PROXY_GATEWAY_502", "PROXY_GATEWAY_503", "PROXY_GATEWAY_504"}
)

CSV_FIELDS = [
    "schema_version",
    "run_id",
    "seq",
    "cycle_id",
    "operation_id",
    "attempt_no",
    "max_attempts",
    "started_at_local",
    "started_at_utc",
    "ended_at_utc",
    "scheduled_at_utc",
    "scheduler_lag_ms",
    "duration_ms",
    "ttfb_ms",
    "target_name",
    "probe_kind",
    "method",
    "url",
    "target_host",
    "effective_proxy",
    "proxy_auth_present",
    "tls_verify",
    "connect_timeout_s",
    "read_timeout_s",
    "outcome",
    "layer",
    "category",
    "attribution",
    "confidence",
    "route_ok",
    "app_ok",
    "network_failure_strict",
    "confirmed_proxy_failure",
    "http_status",
    "http_reason",
    "final_url",
    "redirect_count",
    "bytes_read",
    "content_type",
    "response_headers",
    "body_read_error_category",
    "body_read_error_message",
    "body_read_error_fingerprint",
    "exception_class",
    "root_cause_classes",
    "errno",
    "message",
    "error_fingerprint",
]


class _BodyReadHandled(Exception):
    """Internal control-flow marker after preserving response-header evidence."""


def parse_duration(value: str) -> float:
    """Parse a duration such as 30s, 5m, 24h, or 1d."""
    text = str(value).strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([smhd]?)", text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid duration {value!r}; use forms such as 30s, 5m, 24h, 1d"
        )
    number = float(match.group(1))
    unit = match.group(2) or "s"
    factor = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    seconds = number * factor
    if seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be greater than zero")
    return seconds


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def iso_local(value: datetime, local_tz: ZoneInfo) -> str:
    return value.astimezone(local_tz).isoformat(timespec="milliseconds")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SecretScrubber:
    """Remove known and structurally recognizable credentials from output."""

    _userinfo = re.compile(
        r"(?i)\b(?P<scheme>https?|socks4a?|socks5h?)://(?P<userinfo>[^/@\s]+)@"
    )
    _bearer = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
    _query_secret = re.compile(
        r"(?i)([?&](?:api[_-]?key|key|token|access[_-]?token|signature|password|passwd)="
        r")[^&#\s]*"
    )
    _key_value_secret = re.compile(
        r"(?i)(\b(?:api[_-]?key|authorization|proxy-authorization|password|passwd)"
        r"\s*[:=]\s*)[^,;\s]+"
    )
    _full_url_query = re.compile(r"(?i)(https?://[^\s?'\"()]+)\?[^\s'\"()]+")
    _exception_url_query = re.compile(
        r"(?i)(\b(?:with\s+)?url:\s*[^\s?'\"()]+)\?[^\s'\"()]+"
    )

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        expanded: set[str] = set()
        for raw in secrets:
            if not raw:
                continue
            expanded.add(str(raw))
            expanded.add(unquote(str(raw)))
            expanded.add(quote(unquote(str(raw)), safe=""))
        self._secrets = sorted((s for s in expanded if len(s) >= 4), key=len, reverse=True)

    def sanitize_text(self, value: Any, limit: int = 1200) -> str:
        text = "" if value is None else str(value)
        for secret in self._secrets:
            text = text.replace(secret, "<redacted>")
        text = self._userinfo.sub(lambda m: f"{m.group('scheme')}://***:***@", text)
        text = self._bearer.sub(r"\1<redacted>", text)
        text = self._full_url_query.sub(r"\1?<redacted-query>", text)
        text = self._exception_url_query.sub(r"\1?<redacted-query>", text)
        text = self._query_secret.sub(r"\1<redacted>", text)
        text = self._key_value_secret.sub(r"\1<redacted>", text)
        text = text.replace("\r", " ").replace("\n", " ")
        return text[:limit]

    def sanitize_url(self, raw_url: str) -> str:
        try:
            parsed = urlsplit(raw_url)
            host = parsed.hostname or ""
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            port = f":{parsed.port}" if parsed.port else ""
            path = parsed.path or "/"
            return self.sanitize_text(
                urlunsplit((parsed.scheme, f"{host}{port}", path, "", ""))
            )
        except Exception:
            return self.sanitize_text(str(raw_url).split("?", 1)[0])


def proxy_endpoint(raw_proxy: Optional[str], scrubber: SecretScrubber) -> tuple[str, bool]:
    if not raw_proxy:
        return "DIRECT", False
    try:
        parsed = urlsplit(raw_proxy)
        host = parsed.hostname or "unknown"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port else ""
        endpoint = f"{parsed.scheme or 'proxy'}://{host}{port}"
        return scrubber.sanitize_text(endpoint), parsed.username is not None
    except Exception:
        return scrubber.sanitize_text(raw_proxy), "@" in raw_proxy


def collect_secret_values(api_key: Optional[str]) -> list[str]:
    values: list[str] = []
    if api_key:
        values.append(api_key)
    for name in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        raw = os.environ.get(name)
        if not raw:
            continue
        values.append(raw)
        try:
            parsed = urlsplit(raw)
            if parsed.username:
                values.append(parsed.username)
            if parsed.password:
                values.append(parsed.password)
        except Exception:
            pass
    return values


def load_selected_env_file(path: Path) -> list[str]:
    """Load only the search API settings needed by this monitor."""
    allowed = {"BRAVE_API_KEY", "YIBU_BRAVE_API_KEY", "BRAVE_YIBU_BASE_URL"}
    loaded: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            name, raw_value = stripped.split("=", 1)
            name = name.strip()
            if name not in allowed or name in os.environ:
                continue
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[name] = value
            loaded.append(name)
    return loaded


def effective_proxy_for(url: str, scrubber: SecretScrubber) -> tuple[str, bool]:
    try:
        proxies = requests.utils.get_environ_proxies(url)
        scheme = urlsplit(url).scheme.lower()
        raw = proxies.get(scheme) or proxies.get("all")
        return proxy_endpoint(raw, scrubber)
    except Exception:
        return "UNKNOWN", False


@dataclass(frozen=True)
class ProbeTarget:
    name: str
    kind: str
    url: str
    headers: Mapping[str, str]
    params: Mapping[str, Any]
    timeout_s: float
    max_attempts: int
    retry_delay_s: float
    expect_json: bool
    verify_tls: bool


def parse_named_url(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("custom URL must use NAME=https://host/path")
    name, url = value.split("=", 1)
    name = name.strip()
    url = url.strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError(
            "target NAME may contain only letters, digits, dot, underscore, and dash"
        )
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("target URL must be an absolute http(s) URL")
    return name, url


def exception_chain(exc: BaseException) -> list[BaseException]:
    found: list[BaseException] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if not isinstance(item, BaseException) or id(item) in seen:
            return
        seen.add(id(item))
        found.append(item)
        visit(getattr(item, "__cause__", None))
        visit(getattr(item, "__context__", None))
        visit(getattr(item, "reason", None))
        for arg in getattr(item, "args", ()):
            if isinstance(arg, BaseException):
                visit(arg)

    visit(exc)
    return found


def classify_exception(
    exc: BaseException, using_proxy: bool, scrubber: SecretScrubber
) -> dict[str, Any]:
    chain = exception_chain(exc)
    classes = [f"{type(item).__module__}.{type(item).__name__}" for item in chain]
    combined = " | ".join(f"{type(item).__name__}: {item}" for item in chain)
    lowered = combined.lower()

    def has_type(*types: type[BaseException]) -> bool:
        return any(isinstance(item, types) for item in chain)

    category = "TRANSPORT_ERROR"
    layer = "unknown"
    attribution = "unknown"
    confidence = "low"

    if "wrong_version_number" in lowered or "wrong version number" in lowered:
        category = "TLS_WRONG_VERSION"
        layer = "tls"
        attribution = "shared_proxy_suspected" if using_proxy else "endpoint_or_middlebox"
        confidence = "probable" if using_proxy else "possible"
    elif re.search(r"tunnel connection failed\D*407", lowered) or (
        has_type(requests.exceptions.ProxyError) and "407" in lowered
    ):
        category = "PROXY_AUTH_407"
        layer = "proxy"
        attribution = "proxy"
        confidence = "certain"
    elif re.search(r"tunnel connection failed\D*504", lowered):
        category = "PROXY_TUNNEL_504"
        layer = "proxy"
        attribution = "proxy"
        confidence = "certain"
    elif re.search(r"tunnel connection failed\D*503", lowered):
        category = "PROXY_TUNNEL_503"
        layer = "proxy"
        attribution = "proxy"
        confidence = "certain"
    elif re.search(r"tunnel connection failed\D*502", lowered):
        category = "PROXY_TUNNEL_502"
        layer = "proxy"
        attribution = "proxy"
        confidence = "certain"
    elif "certificate verify failed" in lowered or "sslcertverificationerror" in lowered:
        category = "TLS_CERTIFICATE_ERROR"
        layer = "tls"
        attribution = "endpoint_or_tls_interceptor"
        confidence = "certain"
    elif "handshake" in lowered and "timed out" in lowered:
        category = "TLS_HANDSHAKE_TIMEOUT"
        layer = "tls"
        attribution = "shared_proxy_suspected" if using_proxy else "unknown"
        confidence = "possible"
    elif has_type(requests.exceptions.SSLError) or "sslerror" in lowered:
        category = "TLS_ERROR"
        layer = "tls"
        attribution = "endpoint_or_tls_interceptor"
        confidence = "certain"
    elif any(
        marker in lowered
        for marker in (
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "getaddrinfo failed",
            "nameresolutionerror",
        )
    ) or has_type(socket.gaierror):
        category = "DNS_RESOLUTION_ERROR"
        layer = "dns"
        attribution = "proxy_host" if using_proxy else "target_host"
        confidence = "probable"
    elif has_type(requests.exceptions.ConnectTimeout):
        category = "TCP_CONNECT_TIMEOUT"
        layer = "tcp"
        attribution = "proxy_connect_suspected" if using_proxy else "target_connect"
        confidence = "probable"
    elif has_type(requests.exceptions.ReadTimeout) or "read timed out" in lowered:
        category = "READ_TIMEOUT"
        layer = "tcp"
        attribution = "unknown"
        confidence = "certain"
    elif has_type(requests.exceptions.Timeout) or "timed out" in lowered:
        category = "TIMEOUT"
        layer = "tcp"
        attribution = "unknown"
        confidence = "certain"
    elif any(marker in lowered for marker in ("connection reset", "remote disconnected", "broken pipe")):
        category = "CONNECTION_RESET"
        layer = "tcp"
        attribution = "unknown"
        confidence = "certain"
    elif "connection refused" in lowered:
        category = "CONNECTION_REFUSED"
        layer = "tcp"
        attribution = "proxy_host" if using_proxy else "target_host"
        confidence = "certain"
    elif has_type(requests.exceptions.ProxyError):
        category = "PROXY_CONNECT_ERROR"
        layer = "proxy"
        attribution = "proxy"
        confidence = "probable"
    elif has_type(requests.exceptions.ConnectionError):
        category = "CONNECT_ERROR"
        layer = "tcp"
        attribution = "unknown"
        confidence = "probable"

    errno_value: Optional[int] = None
    for item in chain:
        candidate = getattr(item, "errno", None)
        if isinstance(candidate, int):
            errno_value = candidate
            break

    safe_message = scrubber.sanitize_text(combined)
    fingerprint_source = re.sub(r"\b\d+(?:\.\d+)?\b", "#", safe_message.lower())
    fingerprint = hashlib.sha256(
        f"{category}|{fingerprint_source}".encode("utf-8", "replace")
    ).hexdigest()[:16]

    proxy_error_present = has_type(requests.exceptions.ProxyError)
    proxy_error_is_confirmed = proxy_error_present and category not in {
        "TLS_WRONG_VERSION",
        "TLS_CERTIFICATE_ERROR",
        "TLS_HANDSHAKE_TIMEOUT",
        "TLS_ERROR",
    }
    if proxy_error_is_confirmed and attribution == "unknown":
        attribution = "proxy"

    return {
        "outcome": "route_failure",
        "layer": layer,
        "category": category,
        "attribution": attribution,
        "confidence": confidence,
        "route_ok": False,
        "app_ok": False,
        "network_failure_strict": category in NETWORK_FAILURE_CATEGORIES,
        "confirmed_proxy_failure": (
            category in CONFIRMED_PROXY_CATEGORIES or proxy_error_is_confirmed
        ),
        "exception_class": type(exc).__name__,
        "root_cause_classes": classes,
        "errno": errno_value,
        "message": safe_message,
        "error_fingerprint": fingerprint,
    }


def selected_response_headers(headers: Mapping[str, str], scrubber: SecretScrubber) -> dict[str, str]:
    selected: dict[str, str] = {}
    for name in ("Via", "Server", "X-Cache", "X-Squid-Error", "Content-Type"):
        if name in headers:
            selected[name.lower()] = scrubber.sanitize_text(headers[name], limit=300)
    if "Proxy-Authenticate" in headers:
        scheme = headers["Proxy-Authenticate"].strip().split(None, 1)[0]
        selected["proxy-authenticate"] = f"present:{scrubber.sanitize_text(scheme, 60)}"
    return selected


def proxy_header_evidence(headers: Mapping[str, str]) -> tuple[bool, bool]:
    """Return (strong, weak) evidence that an HTTP response came from a proxy."""
    lowered = {str(k).lower(): str(v).lower() for k, v in headers.items()}
    strong = any(
        key in lowered
        for key in ("proxy-authenticate", "x-squid-error", "proxy-agent")
    )
    server = lowered.get("server", "")
    if any(word in server for word in ("squid", "netentsec", "forward-proxy")):
        strong = True
    weak = "via" in lowered or "envoy" in server
    return strong, weak


def classify_response(
    response: requests.Response,
    using_proxy: bool,
    expect_json: bool,
    scrubber: SecretScrubber,
) -> dict[str, Any]:
    status = response.status_code
    strong_proxy_evidence, weak_proxy_evidence = proxy_header_evidence(response.headers)
    base = {
        "exception_class": "",
        "root_cause_classes": [],
        "errno": None,
        "message": "",
        "error_fingerprint": "",
    }

    if 200 <= status < 400:
        if expect_json:
            try:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"expected JSON object, got {type(payload).__name__}")
            except Exception as exc:
                message = scrubber.sanitize_text(f"{type(exc).__name__}: {exc}")
                base.update(
                    {
                        "outcome": "application_error",
                        "layer": "http_origin",
                        "category": "API_INVALID_JSON",
                        "attribution": "target_application",
                        "confidence": "certain",
                        "route_ok": True,
                        "app_ok": False,
                        "network_failure_strict": False,
                        "confirmed_proxy_failure": False,
                        "exception_class": type(exc).__name__,
                        "root_cause_classes": [type(exc).__name__],
                        "message": message,
                        "error_fingerprint": hashlib.sha256(message.encode()).hexdigest()[:16],
                    }
                )
                return base
        base.update(
            {
                "outcome": "success",
                "layer": "http_origin",
                "category": "SUCCESS",
                "attribution": "target",
                "confidence": "certain",
                "route_ok": True,
                "app_ok": True,
                "network_failure_strict": False,
                "confirmed_proxy_failure": False,
            }
        )
        return base

    if status == 407:
        is_proxy = strong_proxy_evidence or using_proxy
        category = "PROXY_AUTH_407" if is_proxy else "HTTP_407_AMBIGUOUS"
        base.update(
            {
                "outcome": "route_failure" if is_proxy else "origin_rejection",
                "layer": "proxy" if is_proxy else "http_origin",
                "category": category,
                "attribution": "proxy" if is_proxy else "unknown",
                "confidence": "probable" if is_proxy else "low",
                "route_ok": False if is_proxy else True,
                "app_ok": False,
                "network_failure_strict": is_proxy,
                "confirmed_proxy_failure": is_proxy,
            }
        )
        return base

    if status in {502, 503, 504} and strong_proxy_evidence:
        category = f"PROXY_GATEWAY_{status}"
        base.update(
            {
                "outcome": "route_failure_suspected",
                "layer": "proxy",
                "category": category,
                "attribution": "proxy",
                "confidence": "probable",
                "route_ok": False,
                "app_ok": False,
                "network_failure_strict": True,
                "confirmed_proxy_failure": True,
            }
        )
        return base

    if status in {502, 503, 504} and using_proxy and weak_proxy_evidence:
        base.update(
            {
                "outcome": "route_failure_suspected",
                "layer": "proxy_or_http_origin",
                "category": f"PROXY_GATEWAY_{status}_SUSPECTED",
                "attribution": "proxy_or_target_upstream",
                "confidence": "possible",
                "route_ok": True,
                "app_ok": False,
                "network_failure_strict": False,
                "confirmed_proxy_failure": False,
            }
        )
        return base

    if status == 403 and using_proxy and strong_proxy_evidence:
        base.update(
            {
                "outcome": "proxy_policy_rejection",
                "layer": "proxy",
                "category": "PROXY_POLICY_403",
                "attribution": "proxy",
                "confidence": "probable",
                "route_ok": False,
                "app_ok": False,
                "network_failure_strict": False,
                "confirmed_proxy_failure": False,
            }
        )
        return base

    if 400 <= status < 500:
        category = f"ORIGIN_HTTP_{status}"
        attribution = "target_or_api_credentials" if status in {401, 403} else "target"
        base.update(
            {
                "outcome": "origin_rejection",
                "layer": "http_origin",
                "category": category,
                "attribution": attribution,
                "confidence": "probable",
                "route_ok": True,
                "app_ok": False,
                "network_failure_strict": False,
                "confirmed_proxy_failure": False,
            }
        )
        return base

    if 500 <= status < 600:
        category = f"ORIGIN_HTTP_{status}"
        if status == 504:
            category = "HTTP_504_GATEWAY_OR_ORIGIN_AMBIGUOUS"
        base.update(
            {
                "outcome": "origin_service_error",
                "layer": "http_origin",
                "category": category,
                "attribution": "target_or_upstream",
                "confidence": "possible" if status == 504 else "probable",
                "route_ok": True,
                "app_ok": False,
                "network_failure_strict": False,
                "confirmed_proxy_failure": False,
            }
        )
        return base

    base.update(
        {
            "outcome": "http_error",
            "layer": "http_origin",
            "category": f"HTTP_{status}",
            "attribution": "unknown",
            "confidence": "low",
            "route_ok": True,
            "app_ok": False,
            "network_failure_strict": False,
            "confirmed_proxy_failure": False,
        }
    )
    return base


class EventSink:
    def __init__(self, output_dir: Path, fsync_every: int) -> None:
        self.output_dir = output_dir
        self.events_path = output_dir / "events.jsonl"
        self.csv_path = output_dir / "events.csv"
        self._lock = threading.Lock()
        self._seq = 0
        self._writes_since_fsync = 0
        self._fsync_every = max(1, fsync_every)
        self._json_handle = self.events_path.open("a", encoding="utf-8", buffering=1)
        self._csv_handle = self.csv_path.open("a", encoding="utf-8", newline="", buffering=1)
        os.chmod(self.events_path, 0o600)
        os.chmod(self.csv_path, 0o600)
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=CSV_FIELDS)
        if self.csv_path.stat().st_size == 0:
            self._csv_writer.writeheader()
            self._csv_handle.flush()

    def write(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            self._json_handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            csv_row: dict[str, Any] = {}
            for field in CSV_FIELDS:
                value = event.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                csv_row[field] = value
            self._csv_writer.writerow(csv_row)
            self._json_handle.flush()
            self._csv_handle.flush()
            self._writes_since_fsync += 1
            if self._writes_since_fsync >= self._fsync_every:
                os.fsync(self._json_handle.fileno())
                os.fsync(self._csv_handle.fileno())
                self._writes_since_fsync = 0
            return dict(event)

    def flush(self, durable: bool = False) -> None:
        with self._lock:
            self._json_handle.flush()
            self._csv_handle.flush()
            if durable:
                os.fsync(self._json_handle.fileno())
                os.fsync(self._csv_handle.fileno())
                self._writes_since_fsync = 0

    def close(self) -> None:
        with self._lock:
            if self._json_handle.closed:
                return
            self._json_handle.flush()
            self._csv_handle.flush()
            os.fsync(self._json_handle.fileno())
            os.fsync(self._csv_handle.fileno())
            self._json_handle.close()
            self._csv_handle.close()


class ProbeEngine:
    def __init__(
        self,
        run_id: str,
        local_tz: ZoneInfo,
        scrubber: SecretScrubber,
        sink: EventSink,
        max_bytes: int,
        stop_event: threading.Event,
    ) -> None:
        self.run_id = run_id
        self.local_tz = local_tz
        self.scrubber = scrubber
        self.sink = sink
        self.max_bytes = max(1, max_bytes)
        self.stop_event = stop_event

    def probe_target(
        self,
        target: ProbeTarget,
        cycle_id: int,
        scheduled_at: datetime,
        scheduler_lag_ms: float,
    ) -> dict[str, Any]:
        operation_id = uuid.uuid4().hex
        last_event: dict[str, Any] = {}
        for attempt_no in range(1, target.max_attempts + 1):
            if self.stop_event.is_set():
                break
            last_event = self._one_attempt(
                target,
                cycle_id,
                operation_id,
                attempt_no,
                scheduled_at,
                scheduler_lag_ms,
            )
            self.sink.write(last_event)
            if last_event["app_ok"]:
                break
            if self.stop_event.is_set():
                break
            if attempt_no < target.max_attempts:
                if self.stop_event.wait(target.retry_delay_s):
                    break
        return last_event

    def _one_attempt(
        self,
        target: ProbeTarget,
        cycle_id: int,
        operation_id: str,
        attempt_no: int,
        scheduled_at: datetime,
        scheduler_lag_ms: float,
    ) -> dict[str, Any]:
        started = utc_now()
        started_mono = time.monotonic()
        effective_proxy, proxy_auth_present = effective_proxy_for(target.url, self.scrubber)
        using_proxy = effective_proxy not in {"DIRECT", "UNKNOWN"}
        parsed_target = urlsplit(target.url)
        response: Optional[requests.Response] = None

        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "seq": 0,
            "cycle_id": cycle_id,
            "operation_id": operation_id,
            "attempt_no": attempt_no,
            "max_attempts": target.max_attempts,
            "started_at_local": iso_local(started, self.local_tz),
            "started_at_utc": iso_utc(started),
            "ended_at_utc": "",
            "scheduled_at_utc": iso_utc(scheduled_at),
            "scheduler_lag_ms": round(scheduler_lag_ms, 3),
            "duration_ms": 0.0,
            "ttfb_ms": None,
            "target_name": target.name,
            "probe_kind": target.kind,
            "method": "GET",
            "url": self.scrubber.sanitize_url(target.url),
            "target_host": parsed_target.hostname or "",
            "effective_proxy": effective_proxy,
            "proxy_auth_present": proxy_auth_present,
            "tls_verify": target.verify_tls,
            "connect_timeout_s": target.timeout_s,
            "read_timeout_s": target.timeout_s,
            "http_status": None,
            "http_reason": "",
            "final_url": "",
            "redirect_count": 0,
            "bytes_read": 0,
            "content_type": "",
            "response_headers": {},
            "stop_reason": "",
        }

        try:
            with requests.Session() as session:
                session.trust_env = True
                response = session.get(
                    target.url,
                    params=dict(target.params),
                    headers=dict(target.headers),
                    timeout=(target.timeout_s, target.timeout_s),
                    verify=target.verify_tls,
                    allow_redirects=True,
                    stream=True,
                )
                event["ttfb_ms"] = round(response.elapsed.total_seconds() * 1000, 3)
                event.update(
                    {
                        "http_status": response.status_code,
                        "http_reason": self.scrubber.sanitize_text(response.reason, limit=120),
                        "final_url": self.scrubber.sanitize_url(response.url),
                        "redirect_count": len(response.history),
                        "content_type": self.scrubber.sanitize_text(
                            response.headers.get("Content-Type", ""), limit=200
                        ),
                        "response_headers": selected_response_headers(
                            response.headers, self.scrubber
                        ),
                    }
                )
                pre_body_classification: Optional[dict[str, Any]] = None
                if not 200 <= response.status_code < 400:
                    pre_body_classification = classify_response(
                        response, using_proxy, False, self.scrubber
                    )
                body = bytearray()
                try:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        remaining = self.max_bytes - len(body)
                        if remaining <= 0:
                            break
                        body.extend(chunk[:remaining])
                        if len(body) >= self.max_bytes:
                            break
                except Exception as body_exc:
                    body_error = classify_exception(body_exc, using_proxy, self.scrubber)
                    event["bytes_read"] = len(body)
                    event["body_read_error_category"] = body_error["category"]
                    event["body_read_error_message"] = body_error["message"]
                    event["body_read_error_fingerprint"] = body_error["error_fingerprint"]
                    if pre_body_classification is not None:
                        event.update(pre_body_classification)
                        event["exception_class"] = body_error["exception_class"]
                        event["root_cause_classes"] = body_error["root_cause_classes"]
                        event["errno"] = body_error["errno"]
                        event["message"] = self.scrubber.sanitize_text(
                            f"HTTP {response.status_code}; response body read failed: "
                            f"{body_error['message']}"
                        )
                        event["error_fingerprint"] = hashlib.sha256(
                            (
                                f"{event['category']}|{body_error['error_fingerprint']}"
                            ).encode()
                        ).hexdigest()[:16]
                    else:
                        event.update(body_error)
                    raise _BodyReadHandled from None
                response._content = bytes(body)
                response._content_consumed = True
                event["bytes_read"] = len(body)
                event.update(
                    classify_response(response, using_proxy, target.expect_json, self.scrubber)
                )
        except _BodyReadHandled:
            pass
        except Exception as exc:
            event.update(classify_exception(exc, using_proxy, self.scrubber))
        finally:
            if response is not None:
                response.close()
            ended = utc_now()
            event["ended_at_utc"] = iso_utc(ended)
            event["duration_ms"] = round((time.monotonic() - started_mono) * 1000, 3)
        return event


def load_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    bad_lines = 0
    if not path.exists():
        return events, bad_lines
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    events.append(item)
                else:
                    bad_lines += 1
            except json.JSONDecodeError:
                bad_lines += 1
    return events, bad_lines


def percentile(values: list[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def fmt_number(value: Optional[float], digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "N/A"
    return f"{100.0 * numerator / denominator:.3f}%"


def floor_bucket(value: datetime, minutes: int) -> datetime:
    value = value.replace(second=0, microsecond=0)
    minute = value.minute - value.minute % minutes
    return value.replace(minute=minute)


def build_hourly_rows(events: list[dict[str, Any]], local_tz: ZoneInfo) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        dt = parse_timestamp(event["started_at_utc"]).astimezone(local_tz)
        hour = dt.replace(minute=0, second=0, microsecond=0).isoformat()
        buckets[(hour, event.get("target_name", "unknown"))].append(event)
        buckets[(hour, "ALL")].append(event)

    rows: list[dict[str, Any]] = []
    for (hour, target), items in sorted(buckets.items()):
        attempts = len(items)
        network_failures = sum(bool(x.get("network_failure_strict")) for x in items)
        proxy_failures = sum(bool(x.get("confirmed_proxy_failure")) for x in items)
        proxy_suspected = sum(
            str(x.get("category", "")).startswith("PROXY_")
            and not bool(x.get("confirmed_proxy_failure"))
            for x in items
        )
        app_successes = sum(bool(x.get("app_ok")) for x in items)
        business_4xx = sum(
            str(x.get("category", "")).startswith("ORIGIN_HTTP_4") for x in items
        )
        durations = [float(x["duration_ms"]) for x in items if x.get("route_ok")]
        rows.append(
            {
                "local_hour": hour,
                "target": target,
                "attempts": attempts,
                "app_successes": app_successes,
                "strict_network_failures": network_failures,
                "confirmed_proxy_failures": proxy_failures,
                "proxy_suspected_or_policy": proxy_suspected,
                "business_4xx": business_4xx,
                "strict_transport_availability_pct": round(
                    100.0 * (attempts - network_failures) / attempts, 6
                ),
                "p50_route_latency_ms": (
                    round(percentile(durations, 50), 3) if durations else ""
                ),
                "p95_route_latency_ms": (
                    round(percentile(durations, 95), 3) if durations else ""
                ),
                "categories": json.dumps(
                    Counter(str(x.get("category", "UNKNOWN")) for x in items),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return rows


def build_five_minute_rows(
    events: list[dict[str, Any]], local_tz: ZoneInfo
) -> list[dict[str, Any]]:
    buckets: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        dt = parse_timestamp(event["started_at_utc"]).astimezone(local_tz)
        buckets[floor_bucket(dt, 5)].append(event)

    rows: list[dict[str, Any]] = []
    for bucket, items in sorted(buckets.items()):
        strict = [x for x in items if x.get("network_failure_strict")]
        proxy = [x for x in items if x.get("confirmed_proxy_failure")]
        proxy_suspected = [
            x
            for x in items
            if str(x.get("category", "")).startswith("PROXY_")
            and not bool(x.get("confirmed_proxy_failure"))
        ]
        affected_hosts = sorted({str(x.get("target_host", "")) for x in strict})
        categories = Counter(str(x.get("category", "UNKNOWN")) for x in items)
        correlated = len({host for host in affected_hosts if host}) >= 2
        rows.append(
            {
                "local_bucket_start": bucket.isoformat(),
                "attempts": len(items),
                "strict_network_failures": len(strict),
                "confirmed_proxy_failures": len(proxy),
                "proxy_suspected_or_policy": len(proxy_suspected),
                "affected_hosts": ",".join(affected_hosts),
                "cross_host_correlated": correlated,
                "categories": json.dumps(categories, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def build_episodes(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_target[str(event.get("target_name", "unknown"))].append(event)

    episodes: list[dict[str, Any]] = []
    for target, items in by_target.items():
        items.sort(key=lambda x: (x.get("started_at_utc", ""), int(x.get("seq", 0))))
        current: Optional[dict[str, Any]] = None
        successes_after_failure: list[dict[str, Any]] = []
        for item in items:
            failed = bool(item.get("network_failure_strict"))
            if failed:
                if current is None:
                    current = {
                        "target": target,
                        "first_failure": item["started_at_local"],
                        "last_failure": item["started_at_local"],
                        "failure_count": 0,
                        "categories": Counter(),
                        "first_failure_utc": item["started_at_utc"],
                        "last_failure_utc": item["started_at_utc"],
                    }
                current["last_failure"] = item["started_at_local"]
                current["last_failure_utc"] = item["started_at_utc"]
                current["failure_count"] += 1
                current["categories"][str(item.get("category", "UNKNOWN"))] += 1
                successes_after_failure = []
            elif current is not None and item.get("route_ok"):
                successes_after_failure.append(item)
                if len(successes_after_failure) >= 2:
                    current["stable_recovery"] = item["started_at_local"]
                    current["stable_recovery_utc"] = item["started_at_utc"]
                    episodes.append(current)
                    current = None
                    successes_after_failure = []
        if current is not None:
            current["stable_recovery"] = "not observed"
            current["stable_recovery_utc"] = ""
            episodes.append(current)

    for episode in episodes:
        start = parse_timestamp(episode["first_failure_utc"])
        endpoint = episode.get("stable_recovery_utc") or episode["last_failure_utc"]
        end = parse_timestamp(endpoint)
        episode["duration_s"] = max(0.0, (end - start).total_seconds())
        episode["categories"] = dict(episode["categories"])
    episodes.sort(key=lambda x: x["first_failure_utc"])
    return episodes


def longest_failure_streak(items: list[dict[str, Any]]) -> tuple[int, str, str]:
    best_count = 0
    best_start = ""
    best_end = ""
    count = 0
    start = ""
    end = ""
    for item in sorted(items, key=lambda x: (x.get("started_at_utc", ""), int(x.get("seq", 0)))):
        if item.get("network_failure_strict"):
            if count == 0:
                start = item.get("started_at_local", "")
            count += 1
            end = item.get("started_at_local", "")
            if count > best_count:
                best_count, best_start, best_end = count, start, end
        elif item.get("route_ok"):
            count = 0
            start = ""
    return best_count, best_start, best_end


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["no_data"]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def generate_report(
    output_dir: Path,
    meta: dict[str, Any],
    local_tz: ZoneInfo,
    completed: bool,
    bad_lines: int = 0,
    events_path: Optional[Path] = None,
) -> dict[str, Any]:
    events_path = events_path or (output_dir / "events.jsonl")
    events, parsed_bad_lines = load_events(events_path)
    events.sort(key=lambda x: (x.get("started_at_utc", ""), int(x.get("seq", 0))))
    bad_lines += parsed_bad_lines
    hourly_rows = build_hourly_rows(events, local_tz)
    five_rows = build_five_minute_rows(events, local_tz)
    episodes = build_episodes(events)
    write_csv_atomic(output_dir / "hourly_summary.csv", hourly_rows)
    write_csv_atomic(output_dir / "buckets_5m.csv", five_rows)

    categories = Counter(str(x.get("category", "UNKNOWN")) for x in events)
    attempts = len(events)
    strict_failures = sum(bool(x.get("network_failure_strict")) for x in events)
    confirmed_proxy = sum(bool(x.get("confirmed_proxy_failure")) for x in events)
    app_successes = sum(bool(x.get("app_ok")) for x in events)
    route_responses = sum(bool(x.get("route_ok")) for x in events)
    operations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        operations[str(event.get("operation_id", ""))].append(event)
    logical_calls = len(operations)
    logical_successes = sum(any(x.get("app_ok") for x in group) for group in operations.values())
    recovered_by_retry = sum(
        len(group) > 1 and any(x.get("app_ok") for x in group)
        for group in operations.values()
    )

    first_event = events[0]["started_at_local"] if events else "N/A"
    last_event = events[-1]["ended_at_utc"] if events else "N/A"
    strict_items = [x for x in events if x.get("network_failure_strict")]
    first_failure = strict_items[0]["started_at_local"] if strict_items else "none"
    last_failure = strict_items[-1]["started_at_local"] if strict_items else "none"

    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_target[str(event.get("target_name", "unknown"))].append(event)

    report: list[str] = []
    report.append("# AgentFlow 网络稳定性监测报告")
    report.append("")
    report.append(f"- 状态：**{'完整结束' if completed else '运行中或被中断'}**")
    report.append(f"- Run ID：`{meta.get('run_id', 'unknown')}`")
    report.append(f"- 本地时区：`{meta.get('timezone', 'unknown')}`")
    report.append(f"- 首条探测：{first_event}")
    report.append(f"- 末条探测（UTC）：{last_event}")
    report.append(f"- 退出原因：`{meta.get('stop_reason', 'running')}`")
    report.append("")
    report.append("## 结论摘要")
    report.append("")
    report.append(f"- 实际 HTTP attempts：**{attempts}**；逻辑调用：**{logical_calls}**。")
    report.append(
        f"- 严格网络/传输故障：**{strict_failures}**（{fmt_rate(strict_failures, attempts)}）；"
        f"确认/高可信代理故障：**{confirmed_proxy}**（{fmt_rate(confirmed_proxy, attempts)}）。"
    )
    report.append(
        f"- 收到目标或 HTTP 网关响应：**{route_responses}**；业务成功 attempts："
        f"**{app_successes}**（{fmt_rate(app_successes, attempts)}）。"
    )
    report.append(
        f"- 逻辑调用最终成功：**{logical_successes}/{logical_calls}**；"
        f"经重试恢复：**{recovered_by_retry}**。"
    )
    report.append(f"- 首次严格网络故障：{first_failure}；末次：{last_failure}。")
    report.append(
        "- **口径说明：403/412/429 等目标 HTTP 响应表示请求已穿过网络链路，"
        "它们单列为业务/目标站拒绝，不计入严格网络故障率。**"
    )
    report.append(
        "- 旧训练报告按日志行计数，一次请求可能产生多行；本报告按真实 HTTP attempt 计数，"
        "两者只能对照故障时间窗和错误签名，不能直接比较绝对错误率。"
    )
    report.append("")

    report.append("## 探测配置与数据完整性")
    report.append("")
    report.append(f"- 主机：`{meta.get('hostname', 'unknown')}`")
    report.append(
        f"- 计划时长：{meta.get('duration_s', 'unknown')} 秒；间隔："
        f"{meta.get('interval_s', 'unknown')} 秒；并发 worker：{meta.get('concurrency', 'unknown')}。"
    )
    report.append(
        f"- 已执行 cycle：{meta.get('cycles_executed', 0)}；因请求过慢跳过的 cycle："
        f"{meta.get('cycles_skipped', 0)}；JSONL 损坏行：{bad_lines}。"
    )
    report.append(f"- TLS 证书校验：`{meta.get('verify_tls', False)}`（默认与训练代码一致为 False）。")
    report.append("- 代理端点（已移除用户名和密码）：")
    for scheme, endpoint in sorted(meta.get("proxy_endpoints", {}).items()):
        report.append(f"  - `{scheme}` → `{endpoint}`")
    report.append("- 探测目标（查询参数和密钥未落盘）：")
    for target in meta.get("targets", []):
        report.append(
            f"  - `{target.get('name')}`：`{target.get('url')}`，kind={target.get('kind')}，"
            f"timeout={target.get('timeout_s')}s，max_attempts={target.get('max_attempts')}，"
            f"effective_proxy=`{target.get('effective_proxy', 'unknown')}`"
        )
    for warning in meta.get("warnings", []):
        report.append(f"- 警告：{warning}")
    report.append("")

    report.append("## 各目标汇总")
    report.append("")
    report.append(
        "| 目标 | attempts | 逻辑调用 | 业务成功 | 严格网络故障 | 确认代理故障 | 严格传输可用率 | 最长连续故障 |"
    )
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for target, items in sorted(by_target.items()):
        target_operations = {str(x.get("operation_id", "")) for x in items}
        target_strict = sum(bool(x.get("network_failure_strict")) for x in items)
        target_proxy = sum(bool(x.get("confirmed_proxy_failure")) for x in items)
        target_success = sum(bool(x.get("app_ok")) for x in items)
        streak, _, _ = longest_failure_streak(items)
        report.append(
            f"| {target} | {len(items)} | {len(target_operations)} | {target_success} | "
            f"{target_strict} | {target_proxy} | {fmt_rate(len(items) - target_strict, len(items))} | {streak} |"
        )
    if not by_target:
        report.append("| N/A | 0 | 0 | 0 | 0 | 0 | N/A | 0 |")
    report.append("")

    report.append("## 错误分类")
    report.append("")
    report.append("| 分类 | 次数 | 占全部 attempts | 网络故障口径 |")
    report.append("|---|---:|---:|---|")
    for category, count in categories.most_common():
        is_network = "是" if category in NETWORK_FAILURE_CATEGORIES else "否"
        report.append(f"| `{category}` | {count} | {fmt_rate(count, attempts)} | {is_network} |")
    if not categories:
        report.append("| N/A | 0 | N/A | 否 |")
    report.append("")

    report.append("## 逐小时趋势（本地时间）")
    report.append("")
    report.append(
        "| 小时 | attempts | 严格网络故障 | 确认代理故障 | 疑似代理/策略拒绝 | 业务4xx | 严格传输可用率 |"
    )
    report.append("|---|---:|---:|---:|---:|---:|---:|")
    all_hour_rows = [row for row in hourly_rows if row["target"] == "ALL"]
    for row in all_hour_rows:
        report.append(
            f"| {row['local_hour']} | {row['attempts']} | {row['strict_network_failures']} | "
            f"{row['confirmed_proxy_failures']} | {row['proxy_suspected_or_policy']} | "
            f"{row['business_4xx']} | "
            f"{row['strict_transport_availability_pct']:.3f}% |"
        )
    if not all_hour_rows:
        report.append("| N/A | 0 | 0 | 0 | 0 | 0 | N/A |")
    report.append("")

    report.append("## 异常 5 分钟桶与跨域相关性")
    report.append("")
    report.append(
        "| 桶开始 | 严格故障/attempts | 疑似代理/策略拒绝 | 受影响 host | 跨 host 同窗 | 分类 |"
    )
    report.append("|---|---:|---:|---|---|---|")
    abnormal_five = [
        row
        for row in five_rows
        if row["strict_network_failures"] > 0
        or row["proxy_suspected_or_policy"] > 0
    ]
    for row in abnormal_five[:300]:
        report.append(
            f"| {row['local_bucket_start']} | {row['strict_network_failures']}/{row['attempts']} | "
            f"{row['proxy_suspected_or_policy']} | {row['affected_hosts'] or 'N/A'} | "
            f"{'是' if row['cross_host_correlated'] else '否'} | "
            f"`{row['categories']}` |"
        )
    if len(abnormal_five) > 300:
        report.append(
            f"| … | 另有 {len(abnormal_five) - 300} 个异常桶，见 buckets_5m.csv | … | … | … | … |"
        )
    if not abnormal_five:
        report.append("| 无 | 0 | 0 | N/A | 否 | `{}` |")
    report.append("")

    report.append("## 故障片段（连续两次路由成功后确认恢复）")
    report.append("")
    report.append("| 目标 | 首次失败 | 最后失败 | 稳定恢复 | 失败 attempts | 片段跨度 | 分类 |")
    report.append("|---|---|---|---|---:|---:|---|")
    for episode in episodes[:300]:
        report.append(
            f"| {episode['target']} | {episode['first_failure']} | {episode['last_failure']} | "
            f"{episode['stable_recovery']} | {episode['failure_count']} | "
            f"{episode['duration_s'] / 60.0:.1f} min | "
            f"`{json.dumps(episode['categories'], ensure_ascii=False, sort_keys=True)}` |"
        )
    if not episodes:
        report.append("| 无 | N/A | N/A | N/A | 0 | 0 min | `{}` |")
    report.append("")

    report.append("## 延迟（仅 route_ok attempts）")
    report.append("")
    report.append("| 目标 | 样本 | p50 | p90 | p95 | p99 | max |")
    report.append("|---|---:|---:|---:|---:|---:|---:|")
    for target, items in sorted(by_target.items()):
        values = [float(x["duration_ms"]) for x in items if x.get("route_ok")]
        report.append(
            f"| {target} | {len(values)} | {fmt_number(percentile(values, 50))} ms | "
            f"{fmt_number(percentile(values, 90))} ms | {fmt_number(percentile(values, 95))} ms | "
            f"{fmt_number(percentile(values, 99))} ms | {fmt_number(max(values) if values else None)} ms |"
        )
    if not by_target:
        report.append("| N/A | 0 | N/A | N/A | N/A | N/A | N/A |")
    report.append("")

    report.append("## 代表性失败样本（已脱敏）")
    report.append("")
    samples_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_fingerprints: dict[str, set[str]] = defaultdict(set)
    for item in events:
        if item.get("app_ok"):
            continue
        category = str(item.get("category", "UNKNOWN"))
        fingerprint = str(item.get("error_fingerprint", ""))
        if len(samples_by_category[category]) >= 3:
            continue
        if fingerprint and fingerprint in seen_fingerprints[category]:
            continue
        samples_by_category[category].append(item)
        if fingerprint:
            seen_fingerprints[category].add(fingerprint)
    for category, samples in sorted(samples_by_category.items()):
        report.append(f"### `{category}`")
        report.append("")
        for sample in samples:
            message = sample.get("message") or f"HTTP {sample.get('http_status')} {sample.get('http_reason', '')}"
            report.append(
                f"- {sample.get('started_at_local')} | target=`{sample.get('target_name')}` | "
                f"attempt={sample.get('attempt_no')}/{sample.get('max_attempts')} | "
                f"fingerprint=`{sample.get('error_fingerprint') or 'N/A'}` | {message}"
            )
        report.append("")

    report.append("## 数据文件")
    report.append("")
    report.append(
        f"- `{events_path.name}`：逐 HTTP attempt 完整记录，推荐作为 MT 取证原始数据。"
    )
    if (output_dir / "events.csv").exists():
        report.append("- `events.csv`：逐 attempt 表格版。")
    report.append("- `hourly_summary.csv`：逐小时、逐目标及 ALL 汇总。")
    report.append("- `buckets_5m.csv`：5 分钟桶与跨 host 相关性。")
    if events_path.exists():
        report.append(f"- `{events_path.name}` SHA256：`{file_sha256(events_path)}`")
    report.append("")
    report.append(
        "判读重点：若 Yibu 与无关 HTTPS control 在同一 5 分钟桶同时出现 "
        "`TLS_WRONG_VERSION`、`PROXY_AUTH_407` 或 tunnel 504，可高可信指向共享出口/代理；"
        "若仅单一目标持续返回 403/412/429，则优先按目标站策略或 API 配额处理。"
    )

    report_path = output_dir / "network_report.md"
    atomic_write_text(report_path, "\n".join(report) + "\n")
    return {
        "attempts": attempts,
        "strict_failures": strict_failures,
        "confirmed_proxy_failures": confirmed_proxy,
        "app_successes": app_successes,
        "categories": dict(categories),
        "report_path": str(report_path),
    }


def build_targets(args: argparse.Namespace, scrubber: SecretScrubber) -> tuple[list[ProbeTarget], list[str]]:
    targets: list[ProbeTarget] = []
    warnings: list[str] = []
    verify_tls = bool(args.verify_tls)

    api_key = os.environ.get("BRAVE_API_KEY") or os.environ.get("YIBU_BRAVE_API_KEY")
    search_url = os.environ.get("BRAVE_YIBU_BASE_URL") or args.search_url
    if not args.no_search:
        if api_key:
            targets.append(
                ProbeTarget(
                    name="yibu_search",
                    kind="yibu_api",
                    url=search_url,
                    headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
                    params={"q": args.search_query, "count": 1},
                    timeout_s=args.search_timeout,
                    max_attempts=args.search_attempts,
                    retry_delay_s=args.retry_delay,
                    expect_json=True,
                    verify_tls=verify_tls,
                )
            )
        else:
            raise ValueError(
                "missing BRAVE_API_KEY/YIBU_BRAVE_API_KEY. Provide it through the "
                "environment or --env-file so the real Yibu request is tested; use "
                "--no-search only for an intentional HTTPS-control-only run"
            )

    if not args.no_wikipedia:
        targets.append(
            ProbeTarget(
                name="wikipedia_api",
                kind="wikipedia_api",
                url=args.wikipedia_url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": DEFAULT_USER_AGENT,
                },
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": args.wikipedia_query,
                    "format": "json",
                    "srlimit": 1,
                    "utf8": 1,
                },
                timeout_s=args.wikipedia_timeout,
                max_attempts=args.wikipedia_attempts,
                retry_delay_s=args.retry_delay,
                expect_json=True,
                verify_tls=verify_tls,
            )
        )

    custom_urls = list(args.url)
    if not args.no_default_control:
        custom_urls.insert(0, ("https_control", DEFAULT_CONTROL_URL))
    seen_names: set[str] = {target.name for target in targets}
    for name, url in custom_urls:
        if name in seen_names:
            raise ValueError(f"duplicate target name: {name}")
        seen_names.add(name)
        targets.append(
            ProbeTarget(
                name=name,
                kind="web_fetch" if name != "https_control" else "stable_control",
                url=url,
                headers={
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                },
                params={},
                timeout_s=args.fetch_timeout,
                max_attempts=args.fetch_attempts,
                retry_delay_s=args.retry_delay,
                expect_json=False,
                verify_tls=verify_tls,
            )
        )

    if not targets:
        raise ValueError("no enabled targets; provide an API key or at least one --url")
    return targets, warnings


def proxy_metadata(scrubber: SecretScrubber) -> dict[str, str]:
    result: dict[str, str] = {}
    for scheme in ("http", "https", "all"):
        raw = os.environ.get(f"{scheme}_proxy") or os.environ.get(f"{scheme.upper()}_PROXY")
        endpoint, auth_present = proxy_endpoint(raw, scrubber)
        suffix = " (auth present)" if auth_present else ""
        result[scheme] = f"{endpoint}{suffix}"
    return result


def update_meta(path: Path, meta: dict[str, Any]) -> None:
    meta["updated_at_utc"] = iso_utc(utc_now())
    atomic_write_json(path, meta)


def install_signal_handlers(stop_event: threading.Event, stop_reason: dict[str, str]) -> None:
    def handler(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        if not stop_event.is_set():
            stop_reason["value"] = name
            print(
                f"[{iso_utc(utc_now())}] received {name}; stopping new probes and finalizing report...",
                flush=True,
            )
            stop_event.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def run_monitor(args: argparse.Namespace) -> int:
    os.umask(0o077)
    try:
        local_tz = ZoneInfo(args.timezone)
    except ZoneInfoNotFoundError as exc:
        raise SystemExit(f"unknown timezone {args.timezone!r}: {exc}")

    loaded_env_names: list[str] = []
    if args.env_file:
        env_path = Path(args.env_file).expanduser().resolve()
        if not env_path.is_file():
            raise SystemExit(f"env file does not exist: {env_path}")
        loaded_env_names = load_selected_env_file(env_path)

    api_key = os.environ.get("BRAVE_API_KEY") or os.environ.get("YIBU_BRAVE_API_KEY")
    scrubber = SecretScrubber(collect_secret_values(api_key))
    try:
        targets, warnings = build_targets(args, scrubber)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    target_routes: dict[str, tuple[str, bool]] = {
        target.name: effective_proxy_for(target.url, scrubber) for target in targets
    }
    unproxied_targets = [
        name
        for name, (endpoint, _auth_present) in target_routes.items()
        if endpoint in {"DIRECT", "UNKNOWN"}
    ]
    if unproxied_targets and not args.allow_direct:
        raise SystemExit(
            "these targets are not using an environment proxy: "
            + ", ".join(unproxied_targets)
            + ". Run `source train-roma/con_to_web.sh` first. "
            "Use --allow-direct only for an intentional direct-route control test."
        )
    if unproxied_targets:
        warnings.append(
            "以下目标未走代理（显式 --allow-direct）：" + ", ".join(unproxied_targets)
        )

    started = utc_now()
    run_id = f"netmon-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = Path.cwd() / f"network_probe_{started.strftime('%Y%m%d_%H%M%S')}"
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"output path exists and is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicts = [
        output_dir / name
        for name in ("events.jsonl", "events.csv", "run_meta.json", "monitor.pid")
        if (output_dir / name).exists()
    ]
    if conflicts:
        raise SystemExit(
            "output directory already contains monitor data: "
            + ", ".join(str(path) for path in conflicts)
        )
    os.chmod(output_dir, 0o700)
    pid_path = output_dir / "monitor.pid"
    atomic_write_text(pid_path, f"{os.getpid()}\n")

    target_meta = []
    for target in targets:
        endpoint, auth_present = target_routes[target.name]
        target_meta.append(
            {
                "name": target.name,
                "kind": target.kind,
                "url": scrubber.sanitize_url(target.url),
                "timeout_s": target.timeout_s,
                "max_attempts": target.max_attempts,
                "effective_proxy": endpoint,
                "proxy_auth_present": auth_present,
            }
        )
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "stop_reason": "running",
        "started_at_utc": iso_utc(started),
        "started_at_local": iso_local(started, local_tz),
        "timezone": args.timezone,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "requests_version": requests.__version__,
        "duration_s": args.duration,
        "interval_s": args.interval,
        "summary_interval_s": args.summary_interval,
        "concurrency": args.concurrency,
        "verify_tls": bool(args.verify_tls),
        "max_bytes": args.max_bytes,
        "targets": target_meta,
        "proxy_endpoints": proxy_metadata(scrubber),
        "proxy_environment_present": any(
            endpoint not in {"DIRECT", "UNKNOWN"}
            for endpoint, _auth_present in target_routes.values()
        ),
        "loaded_env_names": loaded_env_names,
        "warnings": warnings,
        "cycles_executed": 0,
        "cycles_skipped": 0,
        "output_dir": str(output_dir),
    }
    meta_path = output_dir / "run_meta.json"
    update_meta(meta_path, meta)

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    stop_event = threading.Event()
    stop_reason = {"value": "duration_complete"}
    install_signal_handlers(stop_event, stop_reason)
    sink = EventSink(output_dir, args.fsync_every)
    engine = ProbeEngine(
        run_id, local_tz, scrubber, sink, args.max_bytes, stop_event
    )

    print(f"run_id={run_id}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(
        "targets=" + ", ".join(f"{target.name}:{scrubber.sanitize_url(target.url)}" for target in targets),
        flush=True,
    )
    print("proxy=" + json.dumps(meta["proxy_endpoints"], ensure_ascii=False), flush=True)
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)

    start_mono = time.monotonic()
    deadline_mono = start_mono + args.duration
    next_scheduled_mono = start_mono
    next_summary_mono = start_mono + args.summary_interval
    cycle_id = 0
    last_summary: dict[str, Any] = {}

    try:
        with ThreadPoolExecutor(
            max_workers=args.concurrency, thread_name_prefix="network-probe"
        ) as executor:
            while not stop_event.is_set():
                now_mono = time.monotonic()
                if now_mono >= deadline_mono:
                    break
                wait_s = min(next_scheduled_mono - now_mono, deadline_mono - now_mono)
                if wait_s > 0 and stop_event.wait(wait_s):
                    break
                if time.monotonic() >= deadline_mono:
                    break

                actual_start_mono = time.monotonic()
                scheduled_at = started + timedelta(seconds=cycle_id * args.interval)
                lag_ms = max(0.0, (actual_start_mono - next_scheduled_mono) * 1000.0)
                futures = [
                    executor.submit(
                        engine.probe_target,
                        target,
                        cycle_id,
                        scheduled_at,
                        lag_ms,
                    )
                    for target in targets
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        safe = scrubber.sanitize_text(f"{type(exc).__name__}: {exc}")
                        stop_reason["value"] = f"monitor_error:probe_worker:{type(exc).__name__}"
                        stop_event.set()
                        raise RuntimeError(f"probe worker failed: {safe}") from exc

                meta["cycles_executed"] += 1
                cycle_id += 1
                next_scheduled_mono = start_mono + cycle_id * args.interval
                now_mono = time.monotonic()
                if now_mono >= next_scheduled_mono + args.interval:
                    skipped = int((now_mono - next_scheduled_mono) // args.interval) + 1
                    cycle_id += skipped
                    meta["cycles_skipped"] += skipped
                    next_scheduled_mono = start_mono + cycle_id * args.interval

                if now_mono >= next_summary_mono:
                    sink.flush(durable=True)
                    meta["stop_reason"] = "running"
                    update_meta(meta_path, meta)
                    last_summary = generate_report(
                        output_dir, meta, local_tz, completed=False
                    )
                    local_now = iso_local(utc_now(), local_tz)
                    print(
                        f"[{local_now}] cycles={meta['cycles_executed']} "
                        f"attempts={last_summary['attempts']} "
                        f"strict_failures={last_summary['strict_failures']} "
                        f"categories={json.dumps(last_summary['categories'], sort_keys=True)}",
                        flush=True,
                    )
                    while next_summary_mono <= now_mono:
                        next_summary_mono += args.summary_interval
    except KeyboardInterrupt:
        stop_reason["value"] = "SIGINT"
        stop_event.set()
    except Exception as exc:
        if not str(stop_reason["value"]).startswith("monitor_error:"):
            stop_reason["value"] = f"monitor_error:{type(exc).__name__}"
        safe = scrubber.sanitize_text(f"{type(exc).__name__}: {exc}")
        print(f"fatal monitor error: {safe}", file=sys.stderr, flush=True)
    finally:
        sink.close()
        ended = utc_now()
        meta["status"] = "completed" if stop_reason["value"] == "duration_complete" else "stopped"
        meta["stop_reason"] = stop_reason["value"]
        meta["ended_at_utc"] = iso_utc(ended)
        meta["ended_at_local"] = iso_local(ended, local_tz)
        meta["actual_duration_s"] = round((ended - started).total_seconds(), 3)
        update_meta(meta_path, meta)
        last_summary = generate_report(
            output_dir,
            meta,
            local_tz,
            completed=stop_reason["value"] == "duration_complete",
        )
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass
        print(
            f"final_report={last_summary['report_path']} attempts={last_summary['attempts']} "
            f"strict_failures={last_summary['strict_failures']}",
            flush=True,
        )

    return 0 if not str(stop_reason["value"]).startswith("monitor_error:") else 1


def report_only(args: argparse.Namespace) -> int:
    os.umask(0o077)
    events_path = Path(args.report_only).expanduser().resolve()
    if events_path.is_dir():
        output_dir = events_path
        events_path = output_dir / "events.jsonl"
    else:
        output_dir = events_path.parent
    if not events_path.is_file():
        raise SystemExit(f"events JSONL does not exist: {events_path}")
    meta_path = output_dir / "run_meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    else:
        meta = {
            "run_id": "recovered-from-events",
            "timezone": args.timezone,
            "stop_reason": "report_only_recovery",
            "targets": [],
            "proxy_endpoints": {},
            "warnings": ["run_meta.json 缺失，报告仅从 events.jsonl 恢复。"],
        }
    timezone_name = str(meta.get("timezone") or args.timezone)
    try:
        local_tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        local_tz = ZoneInfo("Asia/Shanghai")
        meta["warnings"] = list(meta.get("warnings", [])) + [
            f"未知时区 {timezone_name}，回退为 Asia/Shanghai。"
        ]
    meta["stop_reason"] = meta.get("stop_reason") or "report_only_recovery"
    summary = generate_report(
        output_dir,
        meta,
        local_tz,
        completed=meta.get("status") == "completed",
        events_path=events_path,
    )
    print(
        f"report={summary['report_path']} attempts={summary['attempts']} "
        f"strict_failures={summary['strict_failures']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously probe the AgentFlow Yibu search, Wikipedia API, and HTTPS "
            "fetch paths, then generate timestamped JSONL/CSV/Markdown evidence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=parse_duration, default=parse_duration("24h"))
    parser.add_argument("--interval", type=parse_duration, default=parse_duration("60s"))
    parser.add_argument(
        "--summary-interval", type=parse_duration, default=parse_duration("5m")
    )
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--env-file",
        default="",
        help="optional .env file; only BRAVE/YIBU key and endpoint variables are loaded",
    )
    parser.add_argument("--search-url", default=DEFAULT_SEARCH_URL)
    parser.add_argument("--search-query", default="AgentFlow network path health check")
    parser.add_argument("--search-timeout", type=float, default=20.0)
    parser.add_argument("--search-attempts", type=int, default=3)
    parser.add_argument("--wikipedia-url", default=DEFAULT_WIKIPEDIA_URL)
    parser.add_argument("--wikipedia-query", default="Moon")
    parser.add_argument("--wikipedia-timeout", type=float, default=10.0)
    parser.add_argument("--wikipedia-attempts", type=int, default=1)
    parser.add_argument("--fetch-timeout", type=float, default=10.0)
    parser.add_argument("--fetch-attempts", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--max-bytes", type=int, default=65536)
    parser.add_argument("--fsync-every", type=int, default=10)
    parser.add_argument("--verify-tls", action="store_true")
    parser.add_argument(
        "--allow-direct",
        action="store_true",
        help=(
            "allow targets that do not resolve to an environment proxy; by default the "
            "monitor fails fast to avoid accidentally testing a route unlike training"
        ),
    )
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--no-wikipedia", action="store_true")
    parser.add_argument("--no-default-control", action="store_true")
    parser.add_argument(
        "--url",
        action="append",
        type=parse_named_url,
        default=[],
        metavar="NAME=URL",
        help="additional web-fetch target; may be supplied more than once",
    )
    parser.add_argument(
        "--report-only",
        default="",
        metavar="EVENTS_OR_DIR",
        help="regenerate report from an existing events.jsonl or result directory",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.concurrency < 1 or args.concurrency > 64:
        raise SystemExit("--concurrency must be between 1 and 64")
    for name in ("search_attempts", "wikipedia_attempts", "fetch_attempts"):
        value = getattr(args, name)
        if value < 1 or value > 20:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 1 and 20")
    for name in ("search_timeout", "wikipedia_timeout", "fetch_timeout"):
        value = getattr(args, name)
        if value <= 0 or value > 600:
            raise SystemExit(f"--{name.replace('_', '-')} must be in (0, 600]")
    if args.retry_delay < 0 or args.retry_delay > 600:
        raise SystemExit("--retry-delay must be between 0 and 600 seconds")
    if args.max_bytes < 1 or args.max_bytes > 100 * 1024 * 1024:
        raise SystemExit("--max-bytes must be between 1 and 104857600")
    if args.fsync_every < 1:
        raise SystemExit("--fsync-every must be at least 1")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    if args.report_only:
        return report_only(args)
    return run_monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())

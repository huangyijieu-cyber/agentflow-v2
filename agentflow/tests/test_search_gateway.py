"""Offline tests for the EC2/Windows Search Gateway client."""

import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from agentflow.tools.search_gateway import (
    SearchGatewayClient,
    SearchGatewayConfigurationError,
    SearchGatewayError,
    _health_check,
    _readiness_check,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload


class RecordingSession:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.trust_env = True
        self.closed = False

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.outcomes:
            raise AssertionError("No fake response remains")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


def make_client(session, **kwargs):
    options = {
        "base_url": "http://7.150.10.123/agentflow-search/",
        "token": "test-gateway-token",
        "connect_timeout": 3,
        "read_timeout": 17,
        "max_retries": 0,
        "session": session,
    }
    options.update(kwargs)
    return SearchGatewayClient(**options)


class SearchGatewayClientTests(unittest.TestCase):
    def test_configuration_is_required_and_validated(self):
        with self.assertRaises(SearchGatewayConfigurationError):
            SearchGatewayClient.from_env(environ={})
        with self.assertRaises(SearchGatewayConfigurationError):
            SearchGatewayClient.from_env(
                environ={"SEARCH_GATEWAY_BASE_URL": "https://gateway.example"}
            )
        with self.assertRaises(SearchGatewayConfigurationError):
            SearchGatewayClient(
                "https://user:pass@gateway.example/path", "token"
            )
        with self.assertRaises(SearchGatewayConfigurationError):
            SearchGatewayClient("https://gateway.example/path?q=1", "token")
        with self.assertRaises(SearchGatewayConfigurationError):
            SearchGatewayClient("https://gateway.example/path", "bad token")
        with self.assertRaises(SearchGatewayConfigurationError):
            SearchGatewayClient("https://gateway.example/path", "token\n")
        with self.assertRaises(SearchGatewayConfigurationError):
            SearchGatewayClient(
                "https://gateway.example/path", "token", read_timeout="nan"
            )

    def test_all_protocol_calls_keep_prefix_and_use_only_gateway(self):
        session = RecordingSession(
            FakeResponse({"status": "ok"}),
            FakeResponse({"text": "page text", "final_url": "https://example.com"}),
            FakeResponse(
                [
                    {
                        "title": "Moon",
                        "url": "https://en.wikipedia.org/wiki/Moon",
                        "abstract": "Earth's natural satellite",
                    }
                ]
            ),
            FakeResponse({"web": {"results": []}}),
        )
        client = make_client(session)

        self.assertFalse(session.trust_env)
        self.assertEqual(client.health(), {"status": "ok"})
        self.assertEqual(client.fetch("https://example.com")["text"], "page text")
        self.assertEqual(
            client.wikipedia_search("Moon", max_pages=5, max_length=500)[
                "results"
            ][0]["title"],
            "Moon",
        )
        client.brave_search(
            "latest research",
            count=7,
            country="US",
            search_lang="en",
            ui_lang="en-US",
            freshness="pw",
        )

        expected_urls = [
            "http://7.150.10.123/agentflow-search/healthz",
            "http://7.150.10.123/agentflow-search/v1/fetch",
            "http://7.150.10.123/agentflow-search/v1/search/wikipedia",
            "http://7.150.10.123/agentflow-search/v1/search/brave",
        ]
        self.assertEqual([call[1] for call in session.calls], expected_urls)
        self.assertEqual(session.calls[0][0], "GET")
        self.assertEqual(session.calls[1][0], "POST")
        self.assertNotIn("json", session.calls[0][2])
        self.assertEqual(
            session.calls[1][2]["json"], {"url": "https://example.com"}
        )
        self.assertEqual(
            session.calls[2][2]["json"],
            {
                "query": "Moon",
                "max_pages": 5,
                "max_length": 500,
                "language": "en",
            },
        )
        self.assertEqual(
            session.calls[3][2]["json"],
            {
                "query": "latest research",
                "count": 7,
                "country": "US",
                "search_lang": "en",
                "ui_lang": "en-US",
                "freshness": "pw",
            },
        )

        for _, url, kwargs in session.calls:
            self.assertTrue(url.startswith(client.base_url + "/"))
            self.assertEqual(
                kwargs["headers"]["Authorization"], "Bearer test-gateway-token"
            )
            self.assertEqual(kwargs["timeout"], (3.0, 17.0))
            self.assertIs(kwargs["verify"], True)

    def test_ca_bundle_and_context_manager(self):
        session = RecordingSession(FakeResponse({"status": "ok"}))
        with tempfile.NamedTemporaryFile() as ca_file:
            with make_client(session, ca_bundle=ca_file.name) as client:
                client.health()
                self.assertEqual(client.verify, os.path.abspath(ca_file.name))
        self.assertTrue(session.closed)

    def test_missing_ca_bundle_is_rejected(self):
        with self.assertRaises(SearchGatewayConfigurationError):
            make_client(RecordingSession(), ca_bundle="/not/a/real/ca.pem")

    def test_response_schema_is_checked(self):
        with self.assertRaisesRegex(SearchGatewayError, "missing text"):
            make_client(RecordingSession(FakeResponse({}))).fetch(
                "https://example.com"
            )
        with self.assertRaisesRegex(SearchGatewayError, "missing results"):
            make_client(RecordingSession(FakeResponse({"results": None}))).wikipedia_search(
                "Moon"
            )
        with self.assertRaisesRegex(SearchGatewayError, "JSON object"):
            make_client(RecordingSession(FakeResponse([]))).brave_search("Moon")
        with self.assertRaisesRegex(SearchGatewayError, "non-JSON"):
            make_client(
                RecordingSession(FakeResponse(ValueError("not json")))
            ).health()

    def test_early_wikipedia_pages_alias_is_normalized(self):
        result = make_client(
            RecordingSession(FakeResponse({"pages": [{"title": "Moon"}]}))
        ).wikipedia_search("Moon")
        self.assertEqual(result["results"], [{"title": "Moon"}])

    def test_request_arguments_are_checked_before_network(self):
        session = RecordingSession()
        client = make_client(session)
        with self.assertRaises(SearchGatewayError):
            client.fetch("file:///etc/passwd")
        with self.assertRaises(SearchGatewayError):
            client.wikipedia_search("Moon", max_pages=0)
        with self.assertRaises(SearchGatewayError):
            client.wikipedia_search("Moon", max_length="invalid")
        with self.assertRaises(SearchGatewayError):
            client.brave_search("Moon", count=21)
        self.assertEqual(session.calls, [])

    def test_connection_error_is_retried_without_proxy_fallback(self):
        session = RecordingSession(
            requests.ConnectionError("offline"),
            FakeResponse({"status": "ok"}),
        )
        client = make_client(session, max_retries=1)
        with patch("agentflow.tools.search_gateway.time.sleep") as sleep:
            self.assertEqual(client.health()["status"], "ok")
        self.assertEqual(len(session.calls), 2)
        sleep.assert_called_once()
        self.assertTrue(
            all("7.150.10.123/agentflow-search" in call[1] for call in session.calls)
        )

    def test_retryable_http_status_is_retried(self):
        session = RecordingSession(
            FakeResponse({}, status_code=502, headers={"Retry-After": "99"}),
            FakeResponse({"status": "ok"}),
        )
        client = make_client(session, max_retries=1)
        with patch("agentflow.tools.search_gateway.time.sleep") as sleep:
            client.health()
        sleep.assert_called_once_with(5.0)
        self.assertEqual(len(session.calls), 2)

    def test_explicit_non_retryable_error_is_not_retried(self):
        session = RecordingSession(
            FakeResponse(
                {
                    "error": {
                        "code": "upstream_unavailable",
                        "message": "No Brave channel is available",
                        "retryable": False,
                        "request_id": "request-123",
                    }
                },
                status_code=503,
            ),
            FakeResponse({"status": "ok"}),
        )
        client = make_client(session, max_retries=1)
        with self.assertRaises(SearchGatewayError) as context:
            client.health()
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(context.exception.code, "upstream_unavailable")
        self.assertEqual(context.exception.request_id, "request-123")
        self.assertNotIn("test-gateway-token", str(context.exception))


class HealthCheckTests(unittest.TestCase):
    def test_health_check_exit_codes(self):
        class HealthyClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def health(self):
                return {"status": "ok"}

        with patch(
            "agentflow.tools.search_gateway.SearchGatewayClient.from_env",
            return_value=HealthyClient(),
        ):
            self.assertEqual(_health_check(), 0)

        with patch(
            "agentflow.tools.search_gateway.SearchGatewayClient.from_env",
            side_effect=SearchGatewayError("offline"),
        ):
            self.assertEqual(_health_check(), 1)

    def test_readiness_checks_public_egress(self):
        class ReadyClient:
            def __init__(self):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def health(self):
                self.calls.append("health")
                return {"status": "ok"}

            def fetch(self, url):
                self.calls.append(("fetch", url))
                return {"text": "Baidu"}

            def wikipedia_search(self, query, **kwargs):
                self.calls.append(("wikipedia", query, kwargs))
                return {"results": [{"title": "Moon"}]}

            def brave_search(self, query, **kwargs):
                self.calls.append(("brave", query, kwargs))
                return {"web": {"results": []}}

        client = ReadyClient()
        env = {
            "SEARCH_GATEWAY_BASE_URL": "http://gateway/agentflow-search",
            "SEARCH_GATEWAY_TOKEN": "token",
            "SEARCH_GATEWAY_SMOKE_URL": "https://example.com/ready",
            "SEARCH_GATEWAY_SMOKE_WIKIPEDIA_QUERY": "Earth",
            "SEARCH_GATEWAY_CHECK_BRAVE": "true",
            "SEARCH_GATEWAY_SMOKE_BRAVE_QUERY": "ready",
        }
        with patch(
            "agentflow.tools.search_gateway.SearchGatewayClient.from_env",
            return_value=client,
        ):
            self.assertEqual(_readiness_check(env), 0)

        self.assertEqual(
            client.calls,
            [
                "health",
                ("fetch", "https://example.com/ready"),
                (
                    "wikipedia",
                    "Earth",
                    {"max_pages": 1, "max_length": 64, "language": "en"},
                ),
                ("brave", "ready", {"count": 1}),
            ],
        )

    def test_readiness_fails_when_public_egress_fails(self):
        class NotReadyClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def health(self):
                return {"status": "ok"}

            def fetch(self, url):
                raise SearchGatewayError("public egress is unavailable")

        with patch(
            "agentflow.tools.search_gateway.SearchGatewayClient.from_env",
            return_value=NotReadyClient(),
        ):
            self.assertEqual(_readiness_check({}), 1)


if __name__ == "__main__":
    unittest.main()

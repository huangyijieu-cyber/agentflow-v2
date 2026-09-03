"""Regression checks that active search tools contain no direct web egress."""

from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "agentflow"


class SearchEgressStaticTests(unittest.TestCase):
    def test_active_tools_do_not_contain_direct_egress(self):
        files = [
            PACKAGE_ROOT / "tools" / "brave_search" / "tool.py",
            PACKAGE_ROOT / "tools" / "web_search" / "tool.py",
            PACKAGE_ROOT / "tools" / "wikipedia_search" / "tool.py",
            PACKAGE_ROOT / "tools" / "wikipedia_search" / "web_rag.py",
        ]
        forbidden = (
            "requests.get(",
            "urlopen(",
            "wikipedia.search(",
            "wikipedia.page(",
            "verify=False",
            "verify = False",
            "yibuapi.com",
            "ssl._create_default_https_context",
        )
        for path in files:
            source = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, source, f"{marker!r} found in {path}")

    def test_gateway_client_disables_environment_proxies(self):
        source = (PACKAGE_ROOT / "tools" / "search_gateway.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.session.trust_env = False", source)


if __name__ == "__main__":
    unittest.main()

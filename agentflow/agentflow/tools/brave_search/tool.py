from typing import Any, Dict, List, Optional

from agentflow.tools.base import BaseTool
from agentflow.tools.search_gateway import SearchGatewayClient, SearchGatewayError

TOOL_NAME = "Yibu_Brave_Search_Tool"
LIMITATIONS = """
1. This tool is suitable for general web search through Yibu's Brave-compatible API.
2. It returns search result snippets, not full webpage content.
3. It may require Web_RAG_Search_Tool for reading a specific result URL in depth.
4. Search quality depends on Brave/Yibu availability and quota.
"""

BEST_PRACTICES = """
1. Use this tool for up-to-date web search and general information retrieval.
2. Use concise keyword queries for best recall.
3. Use count to request multiple candidates when the answer may require comparison.
4. Follow up with Web_RAG_Search_Tool on specific URLs when detailed page content is needed.
"""


class Brave_Search_Tool(BaseTool):
    def __init__(self, gateway_client=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "A web search tool powered by Yibu's Brave-compatible API. "
                "It returns Brave-like web search results with titles, URLs, and snippets."
            ),
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query to find information on the web.",
                "count": "int - Number of search results to return. Default is 10.",
                "country": "str - Optional country code, such as US, CN, JP.",
                "search_lang": "str - Optional search language, such as en, zh-hans.",
                "ui_lang": "str - Optional UI language.",
                "freshness": "str - Optional freshness filter supported by Brave-compatible API.",
            },
            output_type="str - Formatted web search results with titles, URLs, and descriptions.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Who won the euro 2024?", count=5)',
                    "description": "Search recent public web information."
                },
                {
                    "command": 'execution = tool.execute(query="Physics and Society arXiv August 11 2016", count=10)',
                    "description": "Search for a specific article or web page."
                },
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            },
        )

        # The Yibu/Brave credential and upstream endpoint live only on the
        # Windows Gateway.  The training host knows only the Gateway token.
        self.gateway = gateway_client or SearchGatewayClient.from_env()

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _extract_results(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        web = data.get("web")
        web_results = web.get("results") if isinstance(web, dict) else None
        if isinstance(web_results, list):
            normalized = [item for item in web_results if isinstance(item, dict)]
            if normalized:
                return normalized

        organic_results = data.get("organic_results")
        if isinstance(organic_results, list):
            normalized = [
                item for item in organic_results if isinstance(item, dict)
            ]
            if normalized:
                return normalized

        discussions = data.get("discussions")
        discussion_results = (
            discussions.get("results") if isinstance(discussions, dict) else None
        )
        if isinstance(discussion_results, list):
            normalized = [
                item for item in discussion_results if isinstance(item, dict)
            ]
            if normalized:
                return normalized

        return []

    def _format_results(self, query: str, data: Dict[str, Any], count: int) -> str:
        results = self._extract_results(data)
        if not results:
            return f"No Brave search results found for query: {query}"

        lines = [f"Search results for: {query}", ""]
        for idx, item in enumerate(results[:count], start=1):
            title = self._as_text(item.get("title"))
            url = self._as_text(item.get("url") or item.get("link"))
            description = self._as_text(item.get("description") or item.get("snippet"))

            lines.append(f"[{idx}] {title or 'Untitled'}")
            if url:
                lines.append(f"URL: {url}")
            if description:
                lines.append(f"Description: {description}")

            extra_snippets = item.get("extra_snippets")
            if isinstance(extra_snippets, list) and extra_snippets:
                snippet_text = " ".join(self._as_text(s) for s in extra_snippets if s)
                if snippet_text:
                    lines.append(f"Extra snippets: {snippet_text}")

            lines.append("")

        return "\n".join(lines).strip()

    def _execute_search(
        self,
        query: str,
        count: int = 10,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            return "Error: Brave search query must be a non-empty string."
        try:
            normalized_count = max(1, min(int(count), 20))
        except (TypeError, ValueError):
            return "Error: Brave search count must be an integer."

        try:
            data = self.gateway.brave_search(
                query=query,
                count=normalized_count,
                country=country,
                search_lang=search_lang,
                ui_lang=ui_lang,
                freshness=freshness,
            )
            return self._format_results(query, data, normalized_count)
        except SearchGatewayError as e:
            # Fail closed: never retry by contacting Yibu from the training host.
            return f"Error: Brave search via Search Gateway failed: {e}"

    def execute(
        self,
        query: str,
        count: int = 10,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
    ) -> str:
        return self._execute_search(
            query=query,
            count=count,
            country=country,
            search_lang=search_lang,
            ui_lang=ui_lang,
            freshness=freshness,
        )


if __name__ == "__main__":
    tool = Brave_Search_Tool()
    print(tool.execute(query="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.?", count=3))

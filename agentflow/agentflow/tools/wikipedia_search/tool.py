"""Wikipedia RAG tool backed by the AgentFlow Search Gateway."""

from typing import Any, Dict, List

from pydantic import BaseModel

from agentflow.engine.factory import create_llm_engine
from agentflow.models.utils import robust_json_loads
from agentflow.tools.base import BaseTool
from agentflow.tools.search_gateway import SearchGatewayClient, SearchGatewayError
from agentflow.tools.web_search.tool import Web_Search_Tool


TOOL_NAME = "Wikipedia_RAG_Search_Tool"


LIMITATION = f"""
{TOOL_NAME} has the following limitations:
1. It is designed specifically for retrieving grounded information from Wikipedia pages only.
2. Filtering of relevant pages depends on LLM model performance and may not always select optimal pages.
3. The returned information accuracy depends on Wikipedia content quality.
"""


BEST_PRACTICE = f"""
For optimal results with {TOOL_NAME}:
1. Use specific, targeted queries rather than broad or ambiguous questions.
2. The tool automatically filters for relevant pages using LLM-based selection - trust the "relevant_pages" results.
3. If initial results are insufficient, examine the "other_pages" section for additional potentially relevant content.
4. Use it as part of a multi-step research process rather than a single source of truth.
5. You can use the {TOOL_NAME} to get more information from the URLs.
"""


class Select_Relevant_Queries(BaseModel):
    matched_queries: list[str]
    matched_query_ids: list[int]


def select_relevant_queries(
    original_query: str,
    query_candidates: list[str],
    llm_engine: Any,
):
    formatted_candidates = "\n".join(
        f"{index}. {query}" for index, query in enumerate(query_candidates)
    )

    prompt = f"""
You are an expert AI assistant. Your task is to identify and select the most relevant queries from a list of Wikipedia search results that are most likely to address the user’s original question.

## Input

Original Query: `{original_query}`
Query Candidates from Wikipedia Search: `{formatted_candidates}`

## Instructions

1. Carefully read the original query and the list of query candidates.
2. Select the query candidates that are most relevant to the original query — i.e., those most likely to contain the information needed to answer the question.
3. Return the most relevant queries. If you think multiple queries are helpful, you can return up to 3 queries.
4. Return your output in the following format:

```
Matched Queries: <list of matched queries>
Matched Query IDs: <list of matched query ids>. Please make sure the ids are integers. And do not return empty list.
```

## Examples

Original Query: What is the capital of France?
Query Candidates from Wikipedia Search:
0. Closed-ended question
1. France
2. What Is a Nation?
3. Capital city
4. London
5. WhatsApp
6. French Revolution
7. Communes of France
8. Capital punishment
9. Louis XIV

Output:
- Matched Queries: France
- Matched Query IDs: [1]


Original Query: What is the mass of the moon?
Query Candidates from Wikipedia Search:
0. Moon
1. Planetary-mass moon
2. What If the Moon Didn't Exist
3. Earth mass
4. Moon landing
5. Mass
6. Colonization of the Moon
7. Planetary mass
8. Hollow Moon
9. Gravitation of the Moon

Output:
- Matched Queries: Moon, Planetary-mass moon
- Matched Query IDs: [0, 1]
"""

    try:
        response = llm_engine.generate(
            prompt,
            response_format=Select_Relevant_Queries,
        )

        # The local vLLM engine can return a raw JSON string instead of the
        # requested Pydantic model.
        if isinstance(response, str):
            response = Select_Relevant_Queries(**robust_json_loads(response))

        matched_queries = [str(item) for item in response.matched_queries]
        matched_query_ids = [int(item) for item in response.matched_query_ids]
        return matched_queries, matched_query_ids
    except Exception as exc:
        print(f"Error selecting relevant Wikipedia queries: {exc}")
        return [], []


class Wikipedia_Search_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini", gateway_client=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "A tool that searches Wikipedia and returns relevant pages with "
                "their titles, URLs, abstracts, and retrieved information."
            ),
            tool_version="1.0.0",
            input_types={"query": "str - The search query for Wikipedia."},
            output_type=(
                "dict - Search results, relevant pages, URLs, and grounded content."
            ),
            demo_commands=[
                {
                    "command": (
                        'execution = tool.execute(query="What is the exact mass in '
                        'kg of the moon")'
                    ),
                    "description": "Search Wikipedia for the mass of the Moon.",
                },
                {
                    "command": (
                        'execution = tool.execute(query="Funtion of human kidney")'
                    ),
                    "description": (
                        "Search Wikipedia and get information about the function "
                        "of the human kidney."
                    ),
                },
                {
                    "command": (
                        'execution = tool.execute(query="When was the first moon '
                        'landing?")'
                    ),
                    "description": "Search Wikipedia for the first Moon landing.",
                },
            ],
            user_metadata={
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE,
            },
        )

        self.model_string = model_string
        self.gateway = gateway_client or SearchGatewayClient.from_env()
        self.llm_engine = create_llm_engine(
            model_string=model_string,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        self._web_rag_tool = None

    @staticmethod
    def _empty_result(query: str, error: str | None = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "query": query,
            "relevant_pages (to the query)": [],
            "other_pages (may be irrelevant to the query)": [],
        }
        if error:
            result["error"] = error
        return result

    @staticmethod
    def _normalize_page(item: Any) -> Dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        title = item.get("title")
        url = item.get("url") or item.get("fullurl") or item.get("page_url")
        abstract = item.get("abstract")
        if abstract is None:
            abstract = item.get("extract")
        if abstract is None:
            abstract = item.get("text")

        if title is None and url is None and abstract is None:
            return None
        return {
            "title": str(title) if title is not None else None,
            "url": str(url) if url is not None else None,
            "abstract": str(abstract) if abstract is not None else "",
        }

    def search_wikipedia(
        self,
        query: str,
        max_length: int = 256,
        max_pages: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search Wikipedia through EC2 and the Windows Gateway only."""
        response = self.gateway.wikipedia_search(
            query=query,
            max_pages=max_pages,
            max_length=max_length,
            language="en",
        )
        pages = []
        for item in response["results"]:
            normalized = self._normalize_page(item)
            if normalized is not None:
                pages.append(normalized)
        return pages

    def _get_web_rag_tool(self) -> Web_Search_Tool:
        if self._web_rag_tool is None:
            self._web_rag_tool = Web_Search_Tool(
                model_string=self.model_string,
                gateway_client=self.gateway,
            )
        return self._web_rag_tool

    @staticmethod
    def _valid_page_ids(candidate_ids: List[int], page_count: int) -> List[int]:
        valid_ids = []
        seen = set()
        for candidate_id in candidate_ids:
            if 0 <= candidate_id < page_count and candidate_id not in seen:
                seen.add(candidate_id)
                valid_ids.append(candidate_id)
            if len(valid_ids) == 3:
                break
        return valid_ids

    def execute(self, query):
        if not isinstance(query, str) or not query.strip():
            return self._empty_result(
                str(query or ""),
                "Wikipedia search query must be a non-empty string.",
            )

        try:
            search_results = self.search_wikipedia(query)
        except SearchGatewayError as exc:
            # The error is exposed to the Agent, but there is intentionally no
            # training-host Wikipedia fallback.
            return self._empty_result(
                query,
                f"Wikipedia search via Search Gateway failed: {exc}",
            )

        if not search_results:
            return self._empty_result(
                query,
                f"No Wikipedia results found for query: {query}",
            )

        titled_pages = [
            (index, page)
            for index, page in enumerate(search_results)
            if isinstance(page.get("title"), str) and page["title"]
        ]
        if not titled_pages:
            result = self._empty_result(query, "Wikipedia results contain no titles.")
            result["other_pages (may be irrelevant to the query)"] = search_results
            return result

        titles = [page["title"] for _, page in titled_pages]
        _, candidate_ids = select_relevant_queries(query, titles, self.llm_engine)
        matched_ids = self._valid_page_ids(candidate_ids, len(titled_pages))
        # If the local selector fails to emit valid JSON or valid indices, use
        # the first Wikipedia result instead of crashing the rollout.
        if not matched_ids:
            matched_ids = [0]

        selected_source_ids = [titled_pages[index][0] for index in matched_ids]
        selected_source_id_set = set(selected_source_ids)
        pages_data = [search_results[index] for index in selected_source_ids]
        other_pages = [
            page
            for index, page in enumerate(search_results)
            if index not in selected_source_id_set
        ]

        print("model_string:", self.model_string)
        web_rag_tool = self._get_web_rag_tool()
        gateway_failed = False
        for page in pages_data:
            if gateway_failed:
                page["retrieved_information"] = (
                    "Error: Page retrieval was skipped because the Search Gateway "
                    "failed for another selected Wikipedia page."
                )
                continue

            url = page.get("url")
            if not url:
                page["retrieved_information"] = (
                    "Error: Wikipedia result did not include a page URL."
                )
                continue

            # Web_Search_Tool performs the fetch through the same injected
            # Gateway client, then keeps chunk/embed/rank/summarize local.
            retrieved_information = web_rag_tool.execute(
                query=query,
                url=url,
            )
            page["retrieved_information"] = retrieved_information
            if (
                isinstance(retrieved_information, str)
                and retrieved_information.startswith(
                    "Error: Web fetch via Search Gateway failed:"
                )
            ):
                # One failed tunnel/Gateway call is sufficient evidence.  Do
                # not spend the rest of the Executor's timeout retrying every
                # selected page against the same unavailable dependency.
                gateway_failed = True

        return {
            "query": query,
            "relevant_pages (to the query)": pages_data,
            "other_pages (may be irrelevant to the query)": other_pages,
        }

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    tool = Wikipedia_Search_Tool(
        model_string="vllm-Qwen3-30B-A3B-Instruct-2507"
    )
    print(tool.execute(query="When was the first moon landing?"))

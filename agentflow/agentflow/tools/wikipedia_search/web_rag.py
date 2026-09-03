"""Backward-compatible import for the canonical Web RAG implementation.

The previous copy in this module downloaded pages directly from the training
host.  Keeping only this re-export ensures old imports use SearchGatewayClient
as well.
"""

from agentflow.tools.web_search.tool import Web_Search_Tool

__all__ = ["Web_Search_Tool"]

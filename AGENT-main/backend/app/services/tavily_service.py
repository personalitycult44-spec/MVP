
import os
import logging
from typing import Any, Dict, List, Optional

from tavily import TavilyClient

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_MAX_RESULTS = 5
MAX_QUERY_LENGTH = 1000
MAX_CONTENT_PER_RESULT = 4000
MAX_TOTAL_CONTEXT_CHARS = 12000

DEFAULT_SEARCH_DEPTH = os.getenv(
    "TAVILY_SEARCH_DEPTH", "basic"
).strip().lower()

if DEFAULT_SEARCH_DEPTH not in {"basic", "advanced"}:
    DEFAULT_SEARCH_DEPTH = "basic"


# ============================================================
# TAVILY CLIENT
# ============================================================
_tavily_client: Optional[TavilyClient] = None
_cached_api_key: Optional[str] = None


def _get_tavily_client() -> Optional[TavilyClient]:
    """Initialize and reuse the Tavily client without exposing its key."""
    global _tavily_client, _cached_api_key

    api_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not api_key:
        logger.warning("Tavily web search is not configured.")
        return None

    if _tavily_client is not None and _cached_api_key == api_key:
        return _tavily_client

    try:
        _tavily_client = TavilyClient(api_key=api_key)
        _cached_api_key = api_key
        logger.info("Tavily search client initialized.")
        return _tavily_client

    except Exception:
        logger.exception("Could not initialize Tavily client.")
        _tavily_client = None
        _cached_api_key = None
        return None


# ============================================================
# RESULT CLEANING
# ============================================================
def _clean_text(value: Any, limit: int) -> str:
    """Normalize external text and limit its size."""
    if not isinstance(value, str):
        return ""

    text = " ".join(value.split())
    return text[:limit].strip()


def _format_results(
    results: List[Dict[str, Any]],
    max_context_chars: int = MAX_TOTAL_CONTEXT_CHARS,
) -> str:
    """
    Convert Tavily results into readable, source-labelled context.

    Output remains a string for compatibility with the existing RAG
    pipeline and LLM context processing.
    """
    sections = []
    used_chars = 0

    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue

        title = _clean_text(result.get("title"), 300)
        url = _clean_text(result.get("url"), 1000)
        content = _clean_text(
            result.get("content"),
            MAX_CONTENT_PER_RESULT,
        )

        if not content:
            continue

        score = result.get("score")
        score_text = ""

        if isinstance(score, (int, float)):
            score_text = f"\nRelevance score: {score:.3f}"

        section = (
            f"SOURCE {index}\n"
            f"Title: {title or 'Untitled result'}\n"
            f"URL: {url or 'URL unavailable'}"
            f"{score_text}\n"
            f"Content: {content}"
        )

        remaining = max_context_chars - used_chars

        if remaining <= 0:
            break

        if len(section) > remaining:
            section = section[:remaining]

        sections.append(section)
        used_chars += len(section)

        if used_chars >= max_context_chars:
            break

    return "\n\n".join(sections)


# ============================================================
# WEB SEARCH
# ============================================================
def search_the_web(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
) -> str:
    """
    Search the web through Tavily when local RAG context is insufficient.

    Returns source-labelled text for the existing LLM pipeline.
    Returns an empty string when no usable results are available.
    """
    if not isinstance(query, str):
        logger.warning("Web search rejected a non-text query.")
        return ""

    query = " ".join(query.split()).strip()

    if not query:
        return ""

    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]

    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        max_results = DEFAULT_MAX_RESULTS

    search_depth = str(search_depth).strip().lower()

    if search_depth not in {"basic", "advanced"}:
        search_depth = DEFAULT_SEARCH_DEPTH

    client = _get_tavily_client()

    if client is None:
        return ""

    try:
        response = client.search(
            query=query,
            search_depth=search_depth,
            topic="general",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )

        if not isinstance(response, dict):
            logger.warning("Tavily returned an unexpected response format.")
            return ""

        results = response.get("results", [])

        if not isinstance(results, list) or not results:
            logger.info("Tavily returned no results for the query.")
            return ""

        formatted_context = _format_results(results)

        if not formatted_context:
            logger.info("Tavily results contained no usable content.")
            return ""

        logger.info(
            "Tavily search completed: %d results, %d context characters.",
            len(results),
            len(formatted_context),
        )

        return formatted_context

    except Exception:
        # Avoid logging API keys, full client objects, or sensitive data.
        logger.exception("Tavily web search failed.")
        return ""

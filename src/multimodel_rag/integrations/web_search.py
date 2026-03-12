"""
web_search.py - Web Search Tool v6
====================================
✅ Wikipedia, ArXiv, DuckDuckGo — graceful fallback if missing
✅ Returns real URLs / citation links, not just domain names
✅ DuckDuckGo generator safety (list() wrapping)
✅ sync-first design + async wrapper (no nested event-loop issues)
✅ Timeout + result length limits
✅ Clean combined output with source headers
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Optional LangChain imports ────────────────────────────────────────────────
try:
    from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    logger.warning(
        "langchain_community not installed — Wikipedia/ArXiv search unavailable. "
        "Install with: pip install langchain-community wikipedia arxiv"
    )

# DuckDuckGo — try ddgs (new name) then duckduckgo_search (old name)
HAS_DDGS = False
DDGS = None
try:
    from ddgs import DDGS  # new package name
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS  # old package name
        HAS_DDGS = True
    except ImportError:
        pass

if not HAS_DDGS:
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        HAS_LANGCHAIN_DDG = True
    except ImportError:
        HAS_LANGCHAIN_DDG = False
else:
    HAS_LANGCHAIN_DDG = False

# ─────────────────────────────────────────────────────────────────────────────
#  INDIVIDUAL SEARCH FUNCTIONS  (all synchronous — easy to thread)
# ─────────────────────────────────────────────────────────────────────────────

def _search_wikipedia(query: str, max_chars: int, top_k: int) -> Dict[str, Any]:
    """Search Wikipedia and return text + URL."""
    if not HAS_LANGCHAIN:
        return {}
    try:
        import wikipedia as _wp          # comes with langchain-community deps
        _wp.set_lang("en")
        page = _wp.page(query, auto_suggest=True)
        snippet = page.summary[:max_chars]
        return {
            "text": snippet,
            "url":  page.url,
            "title": page.title,
        }
    except Exception:
        # Fallback: use LangChain wrapper (no URL available but still useful)
        try:
            wrapper = WikipediaAPIWrapper(
                top_k_results=top_k,
                doc_content_chars_max=max_chars,
            )
            text = wrapper.run(query)
            return {"text": text[:max_chars], "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}", "title": query}
        except Exception as exc:
            logger.debug("Wikipedia search failed: %s", exc)
            return {}


def _search_arxiv(query: str, max_chars: int, top_k: int) -> Dict[str, Any]:
    """Search ArXiv and return text + paper URLs."""
    if not HAS_LANGCHAIN:
        return {}
    try:
        import arxiv as _arxiv
        try:
            # New API (arxiv >= 2.0)
            client = _arxiv.Client()
            search = _arxiv.Search(query=query, max_results=top_k)
            results = list(client.results(search))
        except AttributeError:
            # Fallback for older arxiv package
            search = _arxiv.Search(query=query, max_results=top_k)
            results = list(search.results())
        if not results:
            return {}
        parts: List[str] = []
        urls:  List[str] = []
        for r in results:
            parts.append(f"[{r.title}]\n{r.summary[:300]}")
            urls.append(r.entry_id)           # canonical arxiv URL
        text = "\n\n".join(parts)[:max_chars]
        return {"text": text, "url": urls[0] if urls else "", "urls": urls, "title": results[0].title}
    except Exception:
        # Fallback: LangChain wrapper
        try:
            wrapper = ArxivAPIWrapper(
                top_k_results=top_k,
                doc_content_chars_max=max_chars,
            )
            text = wrapper.run(query)
            return {"text": text[:max_chars], "url": f"https://arxiv.org/search/?query={query.replace(' ', '+')}", "title": query}
        except Exception as exc:
            logger.debug("ArXiv search failed: %s", exc)
            return {}


def _search_duckduckgo(query: str, max_chars: int, top_k: int) -> Dict[str, Any]:
    """Search DuckDuckGo and return text + URLs."""
    # Prefer duckduckgo_search package (more stable)
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                # .text() is a generator — must wrap in list()
                raw = list(ddgs.text(query, max_results=top_k))
            if not raw:
                return {}
            parts: List[str] = []
            urls:  List[str] = []
            for item in raw:
                title = item.get("title", "")
                body  = item.get("body",  "")
                href  = item.get("href",  "")
                parts.append(f"[{title}]\n{body[:300]}")
                if href:
                    urls.append(href)
            text = "\n\n".join(parts)[:max_chars]
            return {"text": text, "url": urls[0] if urls else "", "urls": urls, "title": raw[0].get("title", query)}
        except Exception as exc:
            logger.debug("DDGS search failed: %s", exc)
            return {}

    # Fallback: LangChain DuckDuckGoSearchRun (text only, no URLs)
    if HAS_LANGCHAIN_DDG:
        try:
            tool = DuckDuckGoSearchRun()
            text = tool.run(query)
            return {"text": text[:max_chars], "url": "", "title": query}
        except Exception as exc:
            logger.debug("LangChain DuckDuckGo failed: %s", exc)
            return {}

    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  WEB SEARCH TOOL
# ─────────────────────────────────────────────────────────────────────────────

class WebSearchTool:
    """
    Concurrent multi-source web search with real citation URLs.

    Sources: Wikipedia · ArXiv · DuckDuckGo
    Degrades gracefully when dependencies are missing.

    Usage
    -----
    sync :  result = tool.search_sync(query)
    async:  result = await tool.search(query)
    """

    def __init__(
        self,
        max_chars_per_source: int = 1500,
        top_k:                int = 2,
        timeout:              float = 10.0,
    ):
        self.max_chars = max_chars_per_source
        self.top_k     = top_k
        self.timeout   = timeout

        # Pre-check which sources are potentially available
        self._has_wikipedia  = HAS_LANGCHAIN
        self._has_arxiv      = HAS_LANGCHAIN
        self._has_duckduckgo = HAS_DDGS or (not HAS_DDGS and "HAS_LANGCHAIN_DDG" in globals() and HAS_LANGCHAIN_DDG)

        logger.info(
            "WebSearchTool ready — wikipedia=%s arxiv=%s duckduckgo=%s",
            self._has_wikipedia, self._has_arxiv, self._has_duckduckgo,
        )

    # ── availability ─────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return any([self._has_wikipedia, self._has_arxiv, self._has_duckduckgo])

    def status(self) -> Dict[str, bool]:
        return {
            "wikipedia":   self._has_wikipedia,
            "arxiv":       self._has_arxiv,
            "duckduckgo":  self._has_duckduckgo,
            "langchain":   HAS_LANGCHAIN,
        }

    # ── sync search (primary) ────────────────────────────────────────────────

    def search_sync(
        self,
        query:       str,
        limit_chars: int = 2000,
    ) -> Dict[str, Any]:
        """
        Run query against all available sources concurrently (thread pool).

        Returns:
            {
                "sources":  {"Wikipedia": {"text": …, "url": …}, …},
                "combined": "<all sources concatenated>",
                "urls":     ["https://…", …],
                "available": bool,
            }
        """
        if not self.is_available:
            return {"sources": {}, "combined": "", "urls": [], "available": False}

        tasks = {
            "Wikipedia":  (_search_wikipedia,  self.max_chars, self.top_k),
            "ArXiv":      (_search_arxiv,      self.max_chars, self.top_k),
            "DuckDuckGo": (_search_duckduckgo, self.max_chars, self.top_k),
        }

        sources: Dict[str, Dict[str, Any]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            future_to_name = {
                pool.submit(fn, query, mc, tk): name
                for name, (fn, mc, tk) in tasks.items()
            }
            for future in concurrent.futures.as_completed(
                future_to_name, timeout=self.timeout
            ):
                name = future_to_name[future]
                try:
                    result = future.result()
                    if result and result.get("text", "").strip():
                        sources[name] = {
                            "text":  result["text"][:limit_chars],
                            "url":   result.get("url",  ""),
                            "urls":  result.get("urls", [result.get("url", "")]),
                            "title": result.get("title", name),
                        }
                except concurrent.futures.TimeoutError:
                    logger.debug("%s search timed out", name)
                except Exception as exc:
                    logger.debug("%s search error: %s", name, exc)

        # Build combined text with clear source headers + citation links
        parts: List[str] = []
        all_urls: List[str] = []
        for name, data in sources.items():
            header = f"[{name}]"
            if data.get("url"):
                header += f" — {data['url']}"
            parts.append(f"{header}\n{data['text']}")
            all_urls.extend(u for u in data.get("urls", []) if u)

        combined = "\n\n".join(parts)

        return {
            "sources":   sources,
            "combined":  combined,
            "urls":      all_urls,
            "available": bool(sources),
        }

    # ── async wrapper ────────────────────────────────────────────────────────

    async def search(
        self,
        query:       str,
        limit_chars: int = 2000,
    ) -> Dict[str, Any]:
        """
        Async wrapper around search_sync.
        Runs blocking I/O in a thread-pool executor so the event loop
        is never blocked.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.search_sync,
            query,
            limit_chars,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tool = WebSearchTool()
    print("Status:", tool.status())
    print("is_available:", tool.is_available)

    if tool.is_available:
        result = tool.search_sync("Retrieval Augmented Generation RAG NLP")
        print(f"\nSources found: {list(result['sources'].keys())}")
        print(f"URLs: {result['urls'][:3]}")
        print(f"Combined preview:\n{result['combined'][:400]}")
    else:
        print("⚠️ No search sources available — install langchain-community and/or duckduckgo-search")

    print("\n✅ web_search.py v6 ready")
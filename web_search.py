"""
Web search for supplemental medical literature.
"""

from .config import WEB_SEARCH_AVAILABLE

if WEB_SEARCH_AVAILABLE:
    from duckduckgo_search import DDGS


class WebSearcher:
    """Searches the web for additional medical literature."""

    def search_medical_literature(self, query: str, num_results: int = 5) -> str:
        """Search for additional medical information via web."""
        if not WEB_SEARCH_AVAILABLE:
            return ""

        try:
            with DDGS() as ddgs:
                # Target clinical/medical sources
                medical_query = f"{query} clinical guidelines diagnosis treatment site:ncbi.nlm.nih.gov OR site:uptodate.com OR site:cdc.gov"
                results = list(ddgs.text(medical_query, max_results=num_results))

                if not results:
                    # Fallback to general medical search
                    results = list(ddgs.text(f"{query} diagnosis treatment clinical guidelines", max_results=num_results))

                if not results:
                    return ""

                formatted = []
                for r in results:
                    title = r.get('title', '')
                    body = r.get('body', '')
                    formatted.append(f"• **{title}**: {body}")

                return "\n".join(formatted)
        except:
            return ""

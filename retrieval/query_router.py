# routing/query_router.py

from enum import Enum


class QueryType(Enum):
    DOCUMENT = "document"
    WEB = "web"
    HYBRID = "hybrid"


WEB_KEYWORDS = [
    "latest",
    "recent",
    "current",
    "today",
    "news",
    "2024",
    "2025",
    "update",
    "trend",
    "statistics",
]


def classify_query(query: str) -> QueryType:
    """
    Classifies a query into DOCUMENT, WEB, or HYBRID.
    """

    query_lower = query.lower()

    web_score = sum(1 for kw in WEB_KEYWORDS if kw in query_lower)

    if web_score == 0:
        return QueryType.DOCUMENT

    if web_score >= 2:
        return QueryType.WEB

    return QueryType.HYBRID

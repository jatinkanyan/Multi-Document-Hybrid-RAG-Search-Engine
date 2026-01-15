from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class UnifiedDocument:
    source_id: str
    source_type: str        # pdf / web / wikipedia
    title: str
    content: str
    metadata: Dict


@dataclass
class DocumentChunk:
    chunk_id: str
    source_id: str
    source_type: str
    title: str
    content: str
    chunk_index: int
    metadata: Dict


@dataclass
class WebSearchResult:
    title: str
    content: str
    url: str


@dataclass
class AnswerSource:
    source_type: str        # Doc / Web
    reference: str

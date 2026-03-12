"""
models.py - Shared Data Models
===============================
Single source of truth for all dataclasses used across the project.
Imported by: pdf_processor, document_chunker, vector_store,
             hybrid_retrieval, multimodal_agentic_rag, response_formatter.

DO NOT add business logic here — pure data containers only.
"""

from __future__ import annotations

import time
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  PDF PROCESSING DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessedEquation:
    equation_id: str
    global_number: int
    text: str
    latex: Optional[str]
    page_number: int
    bbox: Tuple[float, float, float, float]
    section: str = ""
    context: str = ""
    confidence: float = 0.9
    raw_text: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedTable:
    table_id: str
    global_number: int
    caption: str
    page_number: int
    markdown: str
    section: str = ""
    html_table: Optional[str] = None
    table_image_path: Optional[str] = None
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    confidence: float = 0.85
    raw_text: str = ""
    parsed_data: Optional[Any] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if 'parsed_data' in d:
            d.pop('parsed_data')
            d['has_parsed_data'] = self.parsed_data is not None
        return d


@dataclass
class ProcessedFigure:
    figure_id: str
    global_number: int
    caption: str
    page_number: int
    saved_path: Optional[str]
    bbox: Tuple[float, float, float, float]
    section: str = ""
    page_width: float = 0
    page_height: float = 0
    visual_content_score: float = 1.0
    confidence: float = 0.85
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedSection:
    """Section with structure awareness"""
    section_id: str
    title: str
    page_number: int
    content: str
    subsections: List[str] = field(default_factory=list)
    equations: List[int] = field(default_factory=list)
    tables: List[int] = field(default_factory=list)
    figures: List[int] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    doc_id: str
    filename: str
    num_pages: int
    page_texts: List[str]
    enriched_page_texts: List[str]
    sections: List[ProcessedSection]
    equations: List[ProcessedEquation]
    tables: List[ProcessedTable]
    figures: List[ProcessedFigure]
    equation_registry: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    table_registry: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    figure_registry: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    title: str = "Unknown"
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    date: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.equation_registry = {eq.global_number: eq.to_dict() for eq in self.equations}
        self.table_registry = {tb.global_number: tb.to_dict() for tb in self.tables}
        self.figure_registry = {fig.global_number: fig.to_dict() for fig in self.figures}


# ─────────────────────────────────────────────────────────────────────────────
#  CHUNK / RETRIEVAL DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """Generic chunk (backward compat)."""
    chunk_id:    str
    text:        str
    doc_id:      str
    page_number: int
    chunk_type:  str
    metadata:    Dict[str, Any]
    image_path:  Optional[str] = None


@dataclass
class MultimodalChunk:
    """Primary chunk used throughout the pipeline."""
    chunk_id:   str
    text:       str
    doc_id:     str
    page_num:   int
    chunk_type: str
    metadata:   Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str]  = None

    @property
    def page_number(self) -> int:
        """Backward-compat alias for page_num."""
        return self.page_num


@dataclass
class SearchResult:
    """Result from vector search — iterable as (chunk, score)."""
    chunk:            MultimodalChunk
    similarity_score: float
    rank:             int

    def __iter__(self):
        yield self.chunk
        yield self.similarity_score


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL ELEMENT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class GlobalElementRegistry:
    """
    O(1) registry for numbered elements: Eq N, Table N, Fig N.
    Also tracks document versions for reproducible runs.
    """

    def __init__(self):
        # {doc_id: {type: {number: chunk_id}}}
        self._registry: Dict[str, Dict[str, Dict[int, str]]] = {}
        # versioning
        self._versions: Dict[str, List[Dict[str, Any]]] = {}

    def register(self, doc_id: str, chunk: MultimodalChunk) -> None:
        """Register a chunk by its type and global_number if available."""
        num = chunk.metadata.get('global_number')
        if num is None:
            return
        self._registry.setdefault(doc_id, {}).setdefault(chunk.chunk_type, {})[int(num)] = chunk.chunk_id

    def lookup(
        self,
        element_type: str,
        number: int,
        doc_id: Optional[str] = None,
    ) -> Optional[str]:
        """O(1) lookup → chunk_id or None."""
        if doc_id:
            return self._registry.get(doc_id, {}).get(element_type, {}).get(number)
        # Search across all docs
        for doc_registry in self._registry.values():
            cid = doc_registry.get(element_type, {}).get(number)
            if cid:
                return cid
        return None

    def register_version(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """Register a processing version; returns version_id."""
        version_id = f"v{len(self._versions.get(doc_id, [])) + 1}_{int(time.time())}"
        self._versions.setdefault(doc_id, []).append({
            'version_id': version_id,
            'timestamp':  time.time(),
            'metadata':   metadata,
        })
        logger.info("📌 Registry version %s for doc %s", version_id, doc_id)
        return version_id

    def get_versions(self, doc_id: str) -> List[Dict[str, Any]]:
        return self._versions.get(doc_id, [])

    def clear(self, doc_id: Optional[str] = None) -> None:
        if doc_id:
            self._registry.pop(doc_id, None)
            self._versions.pop(doc_id, None)
        else:
            self._registry.clear()
            self._versions.clear()

    def stats(self) -> Dict[str, Any]:
        total = sum(
            sum(len(nums) for nums in types.values())
            for types in self._registry.values()
        )
        return {'total_registered': total, 'documents': list(self._registry.keys())}


if __name__ == "__main__":
    # Sanity check
    chunk = MultimodalChunk(
        chunk_id="test_001", text="Hello world", doc_id="doc1",
        page_num=0, chunk_type="text", metadata={"global_number": 1}
    )
    result = SearchResult(chunk=chunk, similarity_score=0.95, rank=0)
    ch, sc = result  # test __iter__
    reg = GlobalElementRegistry()
    reg.register("doc1", chunk)
    assert reg.lookup("text", 1, "doc1") == "test_001"
    print("✅ models.py sanity check passed")
    print(f"   chunk.page_number = {chunk.page_number}")
    print(f"   registry lookup   = {reg.lookup('text', 1, 'doc1')}")
    print(f"   registry stats    = {reg.stats()}")
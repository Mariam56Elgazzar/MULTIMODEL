"""
smart_retriever.py - Enhanced Smart Retrieval System V3.1
=========================================================
✅ Fixed Vector Store V6 compatibility (removed search_type argument)
✅ Automatically routes between search() and hybrid_search()
✅ All required exports for enhanced_rag_system.py
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  RAG TOKEN HELPERS
# ═══════════════════════════════════════════════════════════════════════════

RAG_TOKEN_TERMS = [
    "rag token", "rag-token", "p_rag-token", "p rag-token",
    "p_rag_token", "ragtoken", "top-k", "top k", "p_eta", "p_theta",
    "pη", "pθ"
]

def _contains_rag_token_query(q: str) -> bool:
    q = (q or "").lower()
    return any(term in q for term in ["rag token", "rag-token", "p_rag-token", "ragtoken"])

def _keyword_boost_score(text: str) -> float:
    if not text: return 0.0
    t = text.lower()
    score = sum(1.0 for term in RAG_TOKEN_TERMS if term in t)
    if ("p_rag-token" in t) or ("rag-token(y|x" in t) or ("rag-token (y|x" in t):
        score += 5.0
    return score

# ═══════════════════════════════════════════════════════════════════════════
#  QUERY TYPES
# ═══════════════════════════════════════════════════════════════════════════

class QueryType(Enum):
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"
    GENERAL = "general"
    SPECIFIC_ELEMENT = "specific_element"
    LIST_ALL = "list_all"
    COMPARISON = "comparison"
    CROSS_REFERENCE = "cross_reference"

@dataclass
class QueryIntent:
    query_type: QueryType
    target_type: Optional[str] = None
    target_number: Optional[int] = None
    target_numbers: Optional[List[int]] = None
    keywords: List[str] = field(default_factory=list)
    requires_context: bool = True
    confidence: float = 1.0
    search_strategy: str = "hybrid"

# ═══════════════════════════════════════════════════════════════════════════
#  QUERY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

class QueryClassifier:
    EQUATION_PATTERNS = [r'\bequation\s+(\d+)\b', r'\beq\.?\s+(\d+)\b', r'\bformula\s+(\d+)\b', r'explain.*equation\s+(\d+)', r'show.*equation\s+(\d+)', r'what.*equation\s+(\d+)']
    TABLE_PATTERNS = [r'\btable\s+(\d+)\b', r'\btbl\.?\s+(\d+)\b', r'show.*table\s+(\d+)', r'display.*table\s+(\d+)']
    FIGURE_PATTERNS = [r'\bfigure\s+(\d+)\b', r'\bfig\.?\s+(\d+)\b', r'show.*figure\s+(\d+)']
    LIST_ALL_PATTERNS = [r'(?:show|list|display)\s+all\s+(equation|table|figure)s?', r'how many\s+(equation|table|figure)s?', r'all\s+(?:the\s+)?(equation|table|figure)s?']
    COMPARISON_PATTERNS = [r'compare\s+(equation|table|figure)s?\s+(\d+)\s+(?:and|with)\s+(\d+)', r'difference\s+between\s+(equation|table|figure)s?\s+(\d+)\s+and\s+(\d+)']
    EQUATION_KEYWORDS = ['equation', 'formula', 'mathematical', 'calculation', 'expression', 'variable', 'function', 'compute', 'solve']
    TABLE_KEYWORDS = ['table', 'data', 'results', 'statistics', 'values', 'columns', 'rows', 'dataset', 'comparison']
    FIGURE_KEYWORDS = ['figure', 'image', 'diagram', 'graph', 'plot', 'chart', 'illustration', 'visualization', 'picture']
    
    def classify(self, query: str) -> QueryIntent:
        if not query: return QueryIntent(query_type=QueryType.GENERAL, confidence=0.5)
        query_lower = query.lower().strip()
        
        if any(re.search(p, query_lower) for p in [r'relation.*between', r'how.*relate', r'connection.*between', r'link.*between']):
            return QueryIntent(query_type=QueryType.CROSS_REFERENCE, keywords=query.split(), confidence=0.9)
            
        for pattern in self.COMPARISON_PATTERNS:
            match = re.search(pattern, query_lower)
            if match: return QueryIntent(query_type=QueryType.COMPARISON, target_type=match.group(1), target_numbers=[int(match.group(2)), int(match.group(3))], keywords=query.split(), confidence=0.95)
            
        for pattern in self.LIST_ALL_PATTERNS:
            match = re.search(pattern, query_lower)
            if match: return QueryIntent(query_type=QueryType.LIST_ALL, target_type=match.group(1).rstrip('s'), requires_context=False, confidence=1.0)
            
        for pattern in self.EQUATION_PATTERNS:
            match = re.search(pattern, query_lower)
            if match: return QueryIntent(query_type=QueryType.SPECIFIC_ELEMENT, target_type='equation', target_number=int(match.group(1)), keywords=query.split(), confidence=1.0)
            
        for pattern in self.TABLE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match: return QueryIntent(query_type=QueryType.SPECIFIC_ELEMENT, target_type='table', target_number=int(match.group(1)), keywords=query.split(), confidence=1.0)
            
        for pattern in self.FIGURE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match: return QueryIntent(query_type=QueryType.SPECIFIC_ELEMENT, target_type='figure', target_number=int(match.group(1)), keywords=query.split(), confidence=1.0)
            
        return self._classify_by_keywords(query_lower)
    
    def _classify_by_keywords(self, query: str) -> QueryIntent:
        eq_score = sum(1 for kw in self.EQUATION_KEYWORDS if kw in query)
        tbl_score = sum(1 for kw in self.TABLE_KEYWORDS if kw in query)
        fig_score = sum(1 for kw in self.FIGURE_KEYWORDS if kw in query)
        max_score = max(eq_score, tbl_score, fig_score)
        
        if max_score == 0: return QueryIntent(query_type=QueryType.GENERAL, keywords=query.split(), confidence=0.7)
        if eq_score == max_score: return QueryIntent(query_type=QueryType.EQUATION, target_type='equation', keywords=query.split(), confidence=0.8)
        elif tbl_score == max_score: return QueryIntent(query_type=QueryType.TABLE, target_type='table', keywords=query.split(), confidence=0.8)
        else: return QueryIntent(query_type=QueryType.FIGURE, target_type='figure', keywords=query.split(), confidence=0.8)

# ═══════════════════════════════════════════════════════════════════════════
#  SMART RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════

class SmartRetriever:
    def __init__(self, vector_store: Any):
        self.vector_store = vector_store
        self.classifier = QueryClassifier()
        logger.info("✅ SmartRetriever initialized")
    
    def retrieve(self, query: str, top_k: int = 5, enable_self_rag: bool = True, use_hybrid: bool = True) -> Dict[str, Any]:
        intent = self.classifier.classify(query)
        logger.info(f"Query classified: {intent}")
        
        # 1. Intercept SPECIFIC_ELEMENT
        if intent.query_type == QueryType.SPECIFIC_ELEMENT and intent.target_number:
            chunks = self._retrieve_specific_element(intent.target_type, intent.target_number, top_k)
            if chunks:
                return {'chunks': chunks, 'scores': [1.0] * len(chunks), 'intent': intent, 'strategy': 'registry_lookup', 'success': True}
        
        # 2. ✅ FIX: Intercept LIST_ALL so it actually gets ALL elements, not a random search!
        if intent.query_type == QueryType.LIST_ALL and intent.target_type:
            if hasattr(self.vector_store, 'get_all_chunks_by_type'):
                chunks = self.vector_store.get_all_chunks_by_type(intent.target_type)
                if chunks:
                    return {'chunks': chunks, 'scores': [1.0] * len(chunks), 'intent': intent, 'strategy': 'list_all', 'success': True}
        
        # 3. Standard Search (Hybrid or Dense)
        try:
            target_chunk_type = intent.target_type if intent.query_type != QueryType.GENERAL else None
            
            if use_hybrid and hasattr(self.vector_store, 'hybrid_search'):
                results = self.vector_store.hybrid_search(
                    query=query, 
                    top_k=top_k, 
                    chunk_type=target_chunk_type
                )
                search_strategy = "hybrid"
            elif hasattr(self.vector_store, 'search'):
                results = self.vector_store.search(
                    query=query, 
                    top_k=top_k, 
                    chunk_type=target_chunk_type
                )
                search_strategy = "dense"
            else:
                results = self.vector_store.similarity_search(query, k=top_k)
                search_strategy = "fallback"
            
            if _contains_rag_token_query(query):
                results = self._apply_rag_token_boost(results)
            
            chunks, scores = [], []
            for r in results:
                if isinstance(r, dict):
                    chunks.append(r.get('chunk'))
                    scores.append(r.get('score', 0.0))
                elif hasattr(r, 'chunk'):
                    chunks.append(r.chunk)
                    scores.append(getattr(r, 'similarity_score', 0.0))
                else:
                    chunks.append(r)
                    scores.append(0.5)
            
            return {'chunks': chunks, 'scores': scores, 'intent': intent, 'strategy': search_strategy, 'success': len(chunks) > 0}
        
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            # If it fails, return empty so it doesn't crash the entire app
            return {'chunks': [], 'scores': [], 'intent': intent, 'strategy': 'error', 'success': False, 'error': str(e)}
    
    def _retrieve_specific_element(self, element_type: str, element_number: int, top_k: int) -> List[Any]:
        if hasattr(self.vector_store, 'registry'):
            chunk_id = self.vector_store.registry.lookup(element_type=element_type, number=element_number)
            if chunk_id and hasattr(self.vector_store, 'get_chunk_by_id'):
                chunk = self.vector_store.get_chunk_by_id(chunk_id)
                if chunk: return [chunk]
        
        query = f"{element_type} {element_number}"
        try:
            # ✅ FIX: Removed search_type="dense"
            if hasattr(self.vector_store, 'search'):
                results = self.vector_store.search(query, top_k=top_k, chunk_type=element_type)
            else:
                results = self.vector_store.similarity_search(query, k=top_k)
            for r in results:
                chunk = getattr(r, 'chunk', r)
                if chunk.chunk_type == element_type and chunk.metadata.get('global_number') == element_number:
                    return [chunk]
        except: pass
        return []
    
    def _apply_rag_token_boost(self, results: List[Any]) -> List[Any]:
        boosted = []
        for r in results:
            chunk = getattr(r, 'chunk', r)
            score = getattr(r, 'similarity_score', 0.5)
            new_score = min(1.0, score + (_keyword_boost_score(chunk.text) * 0.1))
            if hasattr(r, 'similarity_score'): r.similarity_score = new_score
            boosted.append(r)
        boosted.sort(key=lambda x: getattr(x, 'similarity_score', 0.5), reverse=True)
        return boosted

__all__ = ['SmartRetriever', 'QueryClassifier', 'QueryType', 'QueryIntent']
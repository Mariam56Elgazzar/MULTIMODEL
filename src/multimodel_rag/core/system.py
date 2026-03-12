"""
enhanced_rag_system.py - Enhanced RAG System V4.0
=================================================
✅ FIXED: Invincible Dictionary Key Handling
✅ FIXED: AsyncGroq Event Loop Blocking
✅ ADDED: Equation OCR config pass-through to PDF processor
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Core imports
from multimodel_rag.core.models import ProcessedDocument, GlobalElementRegistry
from multimodel_rag.guardrails.hallucination_guard import DocumentElementRegistry

# Specialized components
from multimodel_rag.processing.specialized_chunker import SpecializedChunker, SpecializedEmbedder
from multimodel_rag.retrieval.smart_retriever import SmartRetriever, QueryClassifier, QueryType
from multimodel_rag.formatting.advanced_formatter import AdvancedResponseFormatter, ResponseMode
from multimodel_rag.guardrails.self_rag_validator import SelfRAGValidator, ValidationLevel, ResponseQualityAssessor

# Existing components
from multimodel_rag.processing.pdf_processor import EnhancedPDFProcessor
from multimodel_rag.retrieval.vector_store import UnifiedVectorStore
from multimodel_rag.infra.rate_limiter import get_rate_limiter

try:
    # Use AsyncGroq to prevent Streamlit freezing
    from groq import AsyncGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.error("❌ groq package not installed - pip install groq")


@dataclass
class EnhancedRAGConfig:
    groq_api_key: str = field(default_factory=lambda: os.getenv('GROQ_API_KEY', ''))
    groq_model: str = "llama-3.3-70b-versatile"

    # Equation OCR
    enable_equation_ocr: bool = True
    equation_ocr_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Retrieval / formatting
    enable_specialized_chunking: bool = True
    enable_hybrid_search: bool = True
    top_k: int = 5
    enable_self_rag: bool = True
    validation_level: str = "strict"
    max_response_length: int = 4000
    include_references: bool = True
    format_tables_markdown: bool = True
    format_equations_latex: bool = True

    # Paths / runtime
    artifacts_dir: str = "./artifacts"
    cache_dir: str = "./cache"
    enable_rate_limiting: bool = True
    enable_caching: bool = True

    def __post_init__(self):
        if not self.groq_api_key:
            try:
                from multimodel_rag.core.config import GROQ_API_KEY
                if GROQ_API_KEY and GROQ_API_KEY != "your-api-key":
                    self.groq_api_key = GROQ_API_KEY
            except ImportError:
                pass

        Path(self.artifacts_dir).mkdir(exist_ok=True, parents=True)
        Path(self.cache_dir).mkdir(exist_ok=True, parents=True)


class EnhancedRAGSystem:
    def __init__(self, config: EnhancedRAGConfig):
        self.config = config
        self.doc_registry = DocumentElementRegistry()
        self.global_registry = GlobalElementRegistry()
        logger.info("🚀 Initializing Enhanced RAG System V4.0...")

        # Enable OCR only if explicitly enabled and API key exists
        ocr_enabled = bool(config.enable_equation_ocr and config.groq_api_key)

        if config.enable_equation_ocr and not config.groq_api_key:
            logger.warning(
                "⚠️ Equation OCR enabled but GROQ_API_KEY is missing. OCR fallback will be disabled."
            )

        self.pdf_processor = EnhancedPDFProcessor(config={
            'extract_equations': True,
            'extract_tables': True,
            'extract_images': True,
            'output_dir': config.artifacts_dir,
            'save_images': True,
            'equation_confidence_threshold': 0.45,
            'table_confidence_threshold': 0.65,

            # Equation OCR integration
            'enable_equation_ocr': ocr_enabled,
            'equation_ocr_model': config.equation_ocr_model,
            'groq_api_key': config.groq_api_key,
        })

        self.chunker = SpecializedChunker()
        self.embedder = SpecializedEmbedder()
        self.vector_store = UnifiedVectorStore(enable_hybrid_search=config.enable_hybrid_search)
        self.retriever = SmartRetriever(vector_store=self.vector_store)
        self.formatter = AdvancedResponseFormatter(registry=self.doc_registry)

        validation_level_map = {
            'strict': ValidationLevel.STRICT,
            'moderate': ValidationLevel.MODERATE,
            'lenient': ValidationLevel.LENIENT
        }
        self.validator = SelfRAGValidator(
            registry=self.doc_registry,
            level=validation_level_map.get(config.validation_level, ValidationLevel.STRICT)
        )

        if HAS_GROQ and config.groq_api_key:
            self.groq_client = AsyncGroq(api_key=config.groq_api_key)
        else:
            self.groq_client = None
            logger.warning("⚠️ Groq client not available")

        self.rate_limiter = get_rate_limiter() if config.enable_rate_limiting else None
        self.processed_doc: Optional[ProcessedDocument] = None
        self.doc_id: Optional[str] = None
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ Enhanced RAG System V4.0 initialized")

    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        logger.info(f"📄 Processing document: {pdf_path}")
        self.processed_doc = self.pdf_processor.process_pdf(pdf_path)
        self.doc_id = self.processed_doc.doc_id

        chunks = self.chunker.build_all_chunks(self.processed_doc)
        for chunk in chunks:
            chunk.metadata['embedding_text'] = self.embedder.prepare_text_for_embedding(chunk)

        self.vector_store.add_chunks(chunks, doc_id=self.doc_id)
        for chunk in chunks:
            self.global_registry.register(self.doc_id, chunk)

        self.doc_registry.load_from_processed_document(self.processed_doc)
        self._query_cache.clear()

        return {
            'doc_id': self.doc_id,
            'title': self.processed_doc.title,
            'num_pages': self.processed_doc.num_pages,
            'num_equations': len(self.processed_doc.equations),
            'num_tables': len(self.processed_doc.tables),
            'num_figures': len(self.processed_doc.figures),
            'total_chunks': len(chunks),
        }

    async def query(
        self,
        query: str,
        enable_self_rag: Optional[bool] = None,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        if not self.processed_doc:
            return {
                'answer': "⚠️ Please process a document first.",
                'error': 'no_document_loaded',
                'success': False
            }

        if enable_self_rag is None:
            enable_self_rag = self.config.enable_self_rag

        cache_key = f"{query}:{enable_self_rag}"
        if self.config.enable_caching and cache_key in self._query_cache:
            return self._query_cache[cache_key]

        try:
            retrieval_result = self.retriever.retrieve(
                query=query,
                top_k=self.config.top_k,
                enable_self_rag=enable_self_rag,
                use_hybrid=self.config.enable_hybrid_search
            )

            intent = retrieval_result.get('intent')

            # Safely handle both possible response keys
            if 'chunks' in retrieval_result:
                retrieved_chunks = retrieval_result['chunks']
            elif 'results' in retrieval_result:
                retrieved_chunks = retrieval_result['results']
            else:
                retrieved_chunks = []

            self_rag_passed = retrieval_result.get('self_rag_passed', True)

            llm_response = await self._generate_llm_response(query, intent, retrieved_chunks)

            validation = None
            if enable_self_rag and self.validator:
                validation = self.validator.validate_response(
                    response=llm_response,
                    query=query,
                    intent=intent,
                    retrieved_chunks=retrieved_chunks
                )
                if not validation.passed and validation.corrections:
                    llm_response = self.validator.auto_correct_response(llm_response, validation)

            if enable_self_rag and validation is not None and (not validation.passed):
                ql = query.lower().strip()
                if any(k in ql for k in ['rag token', 'rag-token', 'p_rag-token', 'ragtoken']):
                    retry_result = self.retriever.retrieve(
                        query=query,
                        top_k=self.config.top_k,
                        enable_self_rag=False,
                        use_hybrid=False
                    )
                    intent = retry_result.get('intent', intent)

                    if 'chunks' in retry_result:
                        retrieved_chunks = retry_result['chunks']
                    elif 'results' in retry_result:
                        retrieved_chunks = retry_result['results']
                    else:
                        retrieved_chunks = []

                    llm_response = await self._generate_llm_response(query, intent, retrieved_chunks)
                    validation = self.validator.validate_response(
                        response=llm_response,
                        query=query,
                        intent=intent,
                        retrieved_chunks=retrieved_chunks
                    )

            formatted = self.formatter.format_response(
                query=query,
                intent=intent,
                retrieved_chunks=retrieved_chunks,
                llm_response=llm_response
            )
            quality = ResponseQualityAssessor.assess_quality(formatted.content, query)

            result = {
                'answer': formatted.content,
                'mode': formatted.mode.value,
                'success': True
            }

            if return_metadata:
                result['metadata'] = {
                    'intent': intent.query_type.value if intent else "General",
                    'num_sources': len(retrieved_chunks),
                    'self_rag_passed': self_rag_passed,
                    'quality_scores': quality,
                    'latex_equations': formatted.latex_equations,
                    'tables': formatted.tables,
                    'figures': formatted.figures
                }

            if self.config.enable_caching and len(self._query_cache) < 100:
                self._query_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"❌ Query processing error: {e}", exc_info=True)
            return {
                'answer': f"⚠️ An error occurred: {str(e)}",
                'error': str(e),
                'success': False
            }

    async def _generate_llm_response(self, query: str, intent: Any, chunks: List[Any]) -> str:
        if not self.groq_client:
            return "⚠️ LLM not available. Please configure Groq API key."

        context = self._build_context(chunks, intent)
        prompt = self._build_prompt(query, intent, context)

        if self.rate_limiter:
            await self.rate_limiter.acquire(estimated_tokens=1000)

        try:
            response = await self.groq_client.chat.completions.create(
                model=self.config.groq_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(intent)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=self.config.max_response_length,
                top_p=0.9
            )
            answer = response.choices[0].message.content.strip()
            if self.rate_limiter:
                self.rate_limiter.release(response.usage.total_tokens)
            return answer
        except Exception as e:
            if self.rate_limiter:
                self.rate_limiter.release(0)
            return f"⚠️ Error generating response: {str(e)}"

    def _build_context(self, chunks: List[Any], intent: Any) -> str:
        if not chunks:
            return "No relevant information found in the document."

        context_parts = []
        for i, chunk in enumerate(chunks[:self.config.top_k], 1):
            if chunk.chunk_type == 'equation':
                context_parts.append(
                    f"[EQUATION {chunk.metadata.get('global_number', '?')}] "
                    f"(Page {chunk.metadata.get('page_number', '?')})\n"
                    f"LaTeX: {chunk.metadata.get('latex', '')}\n"
                    f"Context: {chunk.metadata.get('context', '')}\n"
                )
            elif chunk.chunk_type == 'table':
                markdown = chunk.metadata.get('markdown', '')
                if len(markdown) > 1000:
                    markdown = markdown[:1000] + "\n[...table truncated...]"
                context_parts.append(
                    f"[TABLE {chunk.metadata.get('global_number', '?')}] "
                    f"(Page {chunk.metadata.get('page_number', '?')})\n"
                    f"Caption: {chunk.metadata.get('caption', 'No caption')}\n"
                    f"{markdown}\n"
                )
            elif chunk.chunk_type == 'figure':
                context_parts.append(
                    f"[FIGURE {chunk.metadata.get('global_number', '?')}] "
                    f"(Page {chunk.metadata.get('page_number', '?')})\n"
                    f"Caption: {chunk.metadata.get('caption', 'No caption')}\n"
                )
            else:
                text = chunk.text[:800] + "..." if len(chunk.text) > 800 else chunk.text
                context_parts.append(
                    f"[TEXT] (Page {chunk.metadata.get('page_number', '?')})\n{text}\n"
                )

        return "\n" + "=" * 70 + "\n".join(context_parts)

    def _build_prompt(self, query: str, intent: Any, context: str) -> str:
        instruction = """Answer the question using ONLY the provided context.
Instructions:
1. Be direct and concise
2. Cite specific equations/tables/figures by number when relevant
3. Include page numbers
4. DO NOT invent or assume information"""
        return (
            f"{instruction}\n\n"
            f"Context from document:\n{context}\n\n"
            f"User Question: {query}\n\n"
            f"Answer (be concise and accurate):"
        )

    def _get_system_prompt(self, intent: Any) -> str:
        return """You are a precise academic assistant specialized in document analysis.
Core Principles:
1. ACCURACY: Use ONLY information from the provided context
2. CONCISENESS: Be direct, avoid unnecessary explanations
3. CITATIONS: Reference specific equations, tables, figures by number
NEVER hallucinate or invent information not in the context."""

    def get_document_info(self) -> Dict[str, Any]:
        if not self.processed_doc:
            return {'error': 'No document loaded'}

        return {
            'doc_id': self.doc_id,
            'title': self.processed_doc.title,
            'num_pages': self.processed_doc.num_pages,
            'equations': [
                {'number': eq.global_number, 'page': eq.page_number}
                for eq in self.processed_doc.equations
            ],
            'tables': [
                {'number': tbl.global_number, 'page': tbl.page_number}
                for tbl in self.processed_doc.tables
            ],
            'figures': [
                {'number': fig.global_number, 'page': fig.page_number}
                for fig in self.processed_doc.figures
            ]
        }

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'document': {
                'loaded': self.processed_doc is not None,
                'doc_id': self.doc_id
            }
        }
        return stats

    def reset(self):
        self.processed_doc, self.doc_id = None, None
        self.vector_store.clear()
        self._query_cache.clear()
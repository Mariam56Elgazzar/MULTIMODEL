"""
advanced_formatter.py - V2.0 (FINAL)
====================================
✅ Seamless integration for perfectly extracted tables
✅ Safe Math rendering
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

def safe_text(text, fallback='[text unavailable]'):
    if text is None: return fallback
    try:
        if not isinstance(text, str): text = str(text)
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except Exception: return fallback

class ResponseMode(Enum):
    SHOW_EQUATION = "show_equation"
    SHOW_TABLE = "show_table"
    SHOW_FIGURE = "show_figure"
    LIST_ALL = "list_all"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    GENERAL = "general"

@dataclass
class FormattedResponse:
    mode: ResponseMode
    content: str
    metadata: Dict[str, Any]
    latex_equations: List[str] = None
    tables: List[Dict[str, Any]] = None
    figures: List[Dict[str, Any]] = None
    warnings: List[str] = None

class AdvancedResponseFormatter:
    def __init__(self, registry: Any = None):
        self.registry = registry

    def format_response(self, query: str, intent: Any, retrieved_chunks: List[Any], llm_response: str) -> FormattedResponse:
        from multimodel_rag.retrieval.smart_retriever import QueryType
        if intent.query_type == QueryType.LIST_ALL:
            return self._format_list_all_response(intent.target_type, retrieved_chunks)
        return self._format_general_response(retrieved_chunks, llm_response)

    def _format_list_all_response(self, element_type: str, chunks: List[Any]) -> FormattedResponse:
        if not chunks:
            return FormattedResponse(mode=ResponseMode.LIST_ALL, content=f"⚠️ No {element_type}s found in the document.", metadata={})

        sorted_chunks = sorted(chunks, key=lambda c: c.metadata.get('global_number', 0))
        content = f"## 📋 All {element_type.title()}s in Document\n\n**Total Count:** {len(sorted_chunks)}\n\n"

        latex_eqs, tables, figures = [], [], []

        for chunk in sorted_chunks:
            num = chunk.metadata.get('global_number', '?')
            page = chunk.metadata.get('page_num', '?')
            section = chunk.metadata.get('section', 'Unknown')

            if element_type == 'equation':
                latex = chunk.metadata.get('latex', chunk.metadata.get('raw_text', ''))
                content += f"**Equation {num}** (Page {page})\n"
                if latex:
                    content += f"$$\n{latex}\n$$\n\n---\n\n"
                    latex_eqs.append(latex)
                else:
                    content += "\n---\n\n"

            elif element_type == 'table':
                caption = chunk.metadata.get('caption', f'Table {num}')
                content += f"### 📊 Table {num} (Page {page})\n"
                md = chunk.metadata.get('markdown', '')
                if md: 
                    content += f"\n{md}\n\n---\n\n"
                tables.append({'number': num, 'caption': caption, 'markdown': md})

            elif element_type == 'figure':
                caption = chunk.metadata.get('caption', f'Figure {num}')
                content += f"### 🖼️ Figure {num} (Page {page})\n*{caption}*\n\n---\n\n"
                figures.append({'number': num, 'caption': caption, 'path': chunk.metadata.get('image_path', '')})

        return FormattedResponse(
            mode=ResponseMode.LIST_ALL, 
            content=content, 
            metadata={'count': len(sorted_chunks)}, 
            latex_equations=latex_eqs, 
            tables=tables, 
            figures=figures
        )

    def _format_general_response(self, chunks: List[Any], llm_response: str) -> FormattedResponse:
        response = re.sub(r'(The document|This document|As mentioned|As shown)\s+', '', llm_response, flags=re.IGNORECASE)
        explanation = '\n\n'.join([line.strip() for line in response.split('\n') if line.strip()])

        references = []
        latex_equations, tables, figures = [], [], []

        for chunk in chunks:
            page = chunk.metadata.get('page_num', '?')
            section = chunk.metadata.get('section', 'Unknown')
            num = chunk.metadata.get('global_number', '?')

            if chunk.chunk_type == 'equation':
                references.append(f"- Equation {num} (Page {page})")
                if chunk.metadata.get('latex'): latex_equations.append(chunk.metadata['latex'])
            elif chunk.chunk_type == 'table':
                references.append(f"- Table {num} (Page {page})")
                if chunk.metadata.get('markdown'): tables.append({'number': num, 'caption': chunk.metadata.get('caption', ''), 'markdown': chunk.metadata['markdown']})
            elif chunk.chunk_type == 'figure':
                references.append(f"- Figure {num} (Page {page})")
                figures.append({'number': num, 'caption': chunk.metadata.get('caption', ''), 'path': chunk.metadata.get('image_path', '')})
            else:
                references.append(f"- Page {page}, Section: {section}")

        ref_text = '\n'.join(references) if references else "No specific references"
        content = f"{explanation}\n\n---\n\n### 📚 Sources\n{ref_text}"

        return FormattedResponse(
            mode=ResponseMode.GENERAL, 
            content=content.strip(), 
            metadata={'num_sources': len(chunks)}, 
            latex_equations=latex_equations, 
            tables=tables, 
            figures=figures
        )
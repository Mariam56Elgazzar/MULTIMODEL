"""
specialized_chunker.py - نظام Chunking & Embedding متخصص V2.0
================================================================
✅ استراتيجية منفصلة لكل نوع محتوى (معادلات/جداول/نصوص/صور)
✅ Metadata غنية لتحسين الاسترجاع
✅ Context window ذكي لكل نوع
✅ Priority scoring تلقائي
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib

logger = logging.getLogger(__name__)

try:
    from multimodel_rag.core.models import MultimodalChunk, ProcessedDocument, ProcessedEquation, ProcessedTable, ProcessedFigure
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False
    logger.warning("⚠️ models.py not found")


@dataclass
class ChunkingStrategy:
    """استراتيجية chunking لنوع محتوى معين"""
    chunk_type: str
    chunk_size: int
    overlap: int
    context_window: int  # كم سطر قبل/بعد للسياق
    priority_boost: float  # boost للأهمية
    metadata_extractors: List[str] = field(default_factory=list)


class SpecializedChunker:
    """
    نظام chunking متخصص لكل نوع محتوى
    """
    
    # استراتيجيات منفصلة لكل نوع
    STRATEGIES = {
        'equation': ChunkingStrategy(
            chunk_type='equation',
            chunk_size=800,  # المعادلات غالباً قصيرة
            overlap=100,
            context_window=3,  # 3 أسطر قبل/بعد
            priority_boost=1.5,
            metadata_extractors=['equation_number', 'section', 'variables']
        ),
        'table': ChunkingStrategy(
            chunk_type='table',
            chunk_size=2000,  # الجداول أكبر
            overlap=0,  # لا overlap للجداول
            context_window=2,
            priority_boost=1.3,
            metadata_extractors=['table_number', 'caption', 'columns']
        ),
        'figure': ChunkingStrategy(
            chunk_type='figure',
            chunk_size=1000,
            overlap=0,
            context_window=2,
            priority_boost=1.2,
            metadata_extractors=['figure_number', 'caption']
        ),
        'text': ChunkingStrategy(
            chunk_type='text',
            chunk_size=1500,
            overlap=300,
            context_window=0,
            priority_boost=1.0,
            metadata_extractors=['section', 'keywords']
        )
    }
    
    def __init__(self):
        logger.info("✅ SpecializedChunker initialized")
    
    def chunk_equation(
        self,
        equation: ProcessedEquation,
        doc_id: str,
        page_text: str,
        section: str = ""
    ) -> MultimodalChunk:
        """
        Chunking متخصص للمعادلات
        """
        strategy = self.STRATEGIES['equation']
        
        # استخراج السياق المحيط
        context = self._extract_context_around_equation(
            equation, page_text, strategy.context_window
        )
        
        # بناء النص الكامل للـ chunk
        chunk_text = f"""
[EQUATION {equation.global_number}]

LaTeX: {equation.latex or equation.text}

Context: {context}

Description: {equation.description or 'Mathematical equation'}

Section: {section or equation.section}
""".strip()
        
        # استخراج المتغيرات
        variables = self._extract_variables(equation.text)
        
        # بناء metadata غنية
        metadata = {
            'global_number': equation.global_number,
            'equation_id': equation.equation_id,
            'section': section or equation.section,
            'page_num': equation.page_number,
            'latex': equation.latex or equation.text,
            'raw_text': equation.text,
            'variables': variables,
            'context': context[:200],  # أول 200 حرف من السياق
            'content_priority': strategy.priority_boost,
            'has_description': bool(equation.description),
            'bbox': equation.bbox,
            'chunk_type': 'equation'
        }
        
        chunk_id = self._generate_chunk_id(doc_id, 'equation', equation.global_number)
        
        return MultimodalChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            doc_id=doc_id,
            page_num=equation.page_number,
            chunk_type='equation',
            metadata=metadata,
            image_path=None
        )
    
    def chunk_table(
        self,
        table: ProcessedTable,
        doc_id: str,
        page_text: str,
        section: str = ""
    ) -> MultimodalChunk:
        """
        Chunking متخصص للجداول
        """
        strategy = self.STRATEGIES['table']
        
        # استخراج السياق
        context = self._extract_context_around_table(
            table, page_text, strategy.context_window
        )
        
        # تحليل هيكل الجدول
        table_structure = self._analyze_table_structure(table.markdown)
        
        # بناء نص الـ chunk
        chunk_text = f"""
[TABLE {table.global_number}]

Caption: {table.caption}

Data:
{table.markdown}

Context: {context}

Section: {section or table.section}

Structure: {table_structure['rows']} rows × {table_structure['cols']} columns
""".strip()
        
        metadata = {
            'global_number': table.global_number,
            'table_id': table.table_id,
            'caption': table.caption,
            'section': section or table.section,
            'page_num': table.page_number,
            'markdown': table.markdown,
            'num_rows': table_structure['rows'],
            'num_cols': table_structure['cols'],
            'headers': table_structure['headers'],
            'context': context[:200],
            'content_priority': strategy.priority_boost,
            'has_image': bool(table.table_image_path),
            'bbox': table.bbox,
            'chunk_type': 'table'
        }
        
        chunk_id = self._generate_chunk_id(doc_id, 'table', table.global_number)
        
        return MultimodalChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            doc_id=doc_id,
            page_num=table.page_number,
            chunk_type='table',
            metadata=metadata,
            image_path=table.table_image_path
        )
    
    def chunk_figure(
        self,
        figure: ProcessedFigure,
        doc_id: str,
        page_text: str,
        section: str = ""
    ) -> MultimodalChunk:
        """
        Chunking متخصص للصور/الرسومات
        """
        strategy = self.STRATEGIES['figure']
        
        # استخراج السياق
        context = self._extract_context_around_figure(
            figure, page_text, strategy.context_window
        )
        
        # بناء نص الـ chunk
        chunk_text = f"""
[FIGURE {figure.global_number}]

Caption: {figure.caption}

Description: {figure.description or 'Visual content'}

Context: {context}

Section: {section or figure.section}
""".strip()
        
        metadata = {
            'global_number': figure.global_number,
            'figure_id': figure.figure_id,
            'caption': figure.caption,
            'section': section or figure.section,
            'page_num': figure.page_number,
            'has_image': bool(figure.saved_path),
            'image_path': figure.saved_path,
            'visual_score': figure.visual_content_score,
            'context': context[:200],
            'content_priority': strategy.priority_boost,
            'bbox': figure.bbox,
            'chunk_type': 'figure'
        }
        
        chunk_id = self._generate_chunk_id(doc_id, 'figure', figure.global_number)
        
        return MultimodalChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            doc_id=doc_id,
            page_num=figure.page_number,
            chunk_type='figure',
            metadata=metadata,
            image_path=figure.saved_path
        )
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        page_num: int,
        section: str = "",
        chunk_idx: int = 0
    ) -> MultimodalChunk:
        """
        Chunking متخصص للنصوص العادية
        """
        strategy = self.STRATEGIES['text']
        
        # استخراج keywords
        keywords = self._extract_keywords(text)
        
        # تحديد نوع المحتوى
        content_type = self._classify_text_content(text)
        
        metadata = {
            'section': section,
            'page_num': page_num,
            'chunk_idx': chunk_idx,
            'keywords': keywords[:10],  # أول 10 keywords
            'content_type': content_type,
            'word_count': len(text.split()),
            'char_count': len(text),
            'content_priority': strategy.priority_boost,
            'chunk_type': 'text'
        }
        
        chunk_id = self._generate_chunk_id(doc_id, 'text', f"{page_num}_{chunk_idx}")
        
        return MultimodalChunk(
            chunk_id=chunk_id,
            text=text.strip(),
            doc_id=doc_id,
            page_num=page_num,
            chunk_type='text',
            metadata=metadata,
            image_path=None
        )
    
    def build_all_chunks(self, processed_doc: ProcessedDocument) -> List[MultimodalChunk]:
        """
        بناء جميع الـ chunks من الوثيقة المعالجة
        """
        chunks = []
        doc_id = processed_doc.doc_id
        
        # 1. معادلات (أولوية عالية)
        logger.info(f"📐 Processing {len(processed_doc.equations)} equations...")
        for eq in processed_doc.equations:
            page_text = processed_doc.page_texts[eq.page_number] if eq.page_number < len(processed_doc.page_texts) else ""
            chunk = self.chunk_equation(eq, doc_id, page_text, eq.section)
            chunks.append(chunk)
        
        # 2. جداول
        logger.info(f"📊 Processing {len(processed_doc.tables)} tables...")
        for table in processed_doc.tables:
            page_text = processed_doc.page_texts[table.page_number] if table.page_number < len(processed_doc.page_texts) else ""
            chunk = self.chunk_table(table, doc_id, page_text, table.section)
            chunks.append(chunk)
        
        # 3. صور/رسومات
        logger.info(f"🖼️ Processing {len(processed_doc.figures)} figures...")
        for fig in processed_doc.figures:
            page_text = processed_doc.page_texts[fig.page_number] if fig.page_number < len(processed_doc.page_texts) else ""
            chunk = self.chunk_figure(fig, doc_id, page_text, fig.section)
            chunks.append(chunk)
        
        # 4. نصوص (بعد إزالة المعادلات/الجداول/الصور)
        logger.info(f"📝 Processing {len(processed_doc.enriched_page_texts)} pages of text...")
        strategy = self.STRATEGIES['text']
        
        for page_num, page_text in enumerate(processed_doc.enriched_page_texts):
            # تقسيم النص إلى chunks
            text_chunks = self._split_text_with_overlap(
                page_text,
                strategy.chunk_size,
                strategy.overlap
            )
            
            # تحديد القسم الحالي
            section = self._find_section_for_page(processed_doc.sections, page_num)
            
            for idx, text_chunk in enumerate(text_chunks):
                if len(text_chunk.strip()) < 100:  # تجاهل chunks صغيرة جداً
                    continue
                chunk = self.chunk_text(text_chunk, doc_id, page_num, section, idx)
                chunks.append(chunk)
        
        logger.info(f"✅ Created {len(chunks)} total chunks")
        logger.info(f"   - Equations: {sum(1 for c in chunks if c.chunk_type == 'equation')}")
        logger.info(f"   - Tables: {sum(1 for c in chunks if c.chunk_type == 'table')}")
        logger.info(f"   - Figures: {sum(1 for c in chunks if c.chunk_type == 'figure')}")
        logger.info(f"   - Text: {sum(1 for c in chunks if c.chunk_type == 'text')}")
        
        return chunks
    
    # ─── Helper Methods ───────────────────────────────────────────────────────
    
    def _generate_chunk_id(self, doc_id: str, chunk_type: str, identifier: Any) -> str:
        """توليد معرّف فريد للـ chunk"""
        content = f"{doc_id}_{chunk_type}_{identifier}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _extract_context_around_equation(
        self, equation: ProcessedEquation, page_text: str, window: int
    ) -> str:
        """استخراج السياق المحيط بالمعادلة"""
        lines = page_text.split('\n')
        
        # محاولة إيجاد المعادلة في النص
        eq_text = equation.text[:50]  # أول 50 حرف
        
        for i, line in enumerate(lines):
            if eq_text in line or equation.latex in line if equation.latex else False:
                start = max(0, i - window)
                end = min(len(lines), i + window + 1)
                context_lines = lines[start:end]
                return ' '.join(context_lines)
        
        # fallback: أخذ سياق عشوائي
        return page_text[:500]
    
    def _extract_context_around_table(
        self, table: ProcessedTable, page_text: str, window: int
    ) -> str:
        """استخراج السياق المحيط بالجدول"""
        lines = page_text.split('\n')
        
        # البحث عن caption
        for i, line in enumerate(lines):
            if table.caption[:30] in line:
                start = max(0, i - window)
                end = min(len(lines), i + window + 1)
                return ' '.join(lines[start:end])
        
        return page_text[:500]
    
    def _extract_context_around_figure(
        self, figure: ProcessedFigure, page_text: str, window: int
    ) -> str:
        """استخراج السياق المحيط بالصورة"""
        lines = page_text.split('\n')
        
        for i, line in enumerate(lines):
            if figure.caption[:30] in line:
                start = max(0, i - window)
                end = min(len(lines), i + window + 1)
                return ' '.join(lines[start:end])
        
        return page_text[:500]
    
    def _extract_variables(self, text: str) -> List[str]:
        """استخراج المتغيرات من المعادلة"""
        # البحث عن متغيرات شائعة
        variables = set()
        
        # متغيرات يونانية
        greek = re.findall(r'[αβγδεζηθικλμνξπρστυφχψω]', text)
        variables.update(greek)
        
        # متغيرات لاتينية (حرف واحد)
        latin = re.findall(r'\b[a-zA-Z]\b', text)
        variables.update(latin)
        
        return list(variables)[:20]  # حد أقصى 20 متغير
    
    def _analyze_table_structure(self, markdown: str) -> Dict[str, Any]:
        """تحليل هيكل الجدول"""
        lines = markdown.strip().split('\n')
        
        # عد الصفوف
        rows = len([l for l in lines if '|' in l and not l.strip().startswith('|-')])
        
        # عد الأعمدة (من أول صف)
        if lines:
            first_row = lines[0]
            cols = len([c for c in first_row.split('|') if c.strip()])
        else:
            cols = 0
        
        # استخراج headers
        headers = []
        if lines:
            header_row = lines[0]
            headers = [h.strip() for h in header_row.split('|') if h.strip()]
        
        return {
            'rows': rows,
            'cols': cols,
            'headers': headers
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """استخراج keywords من النص"""
        # إزالة stop words بسيطة
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        words = re.findall(r'\b\w{4,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # عد التكرارات
        word_counts = {}
        for word in keywords:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # ترتيب حسب التكرار
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, count in sorted_keywords[:20]]
    
    def _classify_text_content(self, text: str) -> str:
        """تصنيف نوع المحتوى النصي"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['method', 'approach', 'algorithm', 'procedure']):
            return 'methodology'
        elif any(kw in text_lower for kw in ['result', 'experiment', 'evaluation', 'performance']):
            return 'results'
        elif any(kw in text_lower for kw in ['introduction', 'background', 'motivation']):
            return 'introduction'
        elif any(kw in text_lower for kw in ['conclusion', 'summary', 'future work']):
            return 'conclusion'
        else:
            return 'general'
    
    def _split_text_with_overlap(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """تقسيم النص مع overlap"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunks.append(' '.join(chunk_words))
            i += chunk_size - overlap
        
        return chunks
    
    def _find_section_for_page(self, sections: List, page_num: int) -> str:
        """إيجاد اسم القسم للصفحة"""
        for section in sections:
            if section.page_number == page_num:
                return section.title
        return ""


# ═══════════════════════════════════════════════════════════════════════════
#  EMBEDDING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

class SpecializedEmbedder:
    """
    نظام Embedding متخصص لكل نوع محتوى
    """
    
    def __init__(self, base_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.base_model = base_model
        logger.info(f"✅ SpecializedEmbedder initialized with {base_model}")
    
    def prepare_text_for_embedding(self, chunk: MultimodalChunk) -> str:
        """
        تحضير النص للـ embedding حسب النوع
        """
        if chunk.chunk_type == 'equation':
            return self._prepare_equation_text(chunk)
        elif chunk.chunk_type == 'table':
            return self._prepare_table_text(chunk)
        elif chunk.chunk_type == 'figure':
            return self._prepare_figure_text(chunk)
        else:
            return self._prepare_general_text(chunk)
    
    def _prepare_equation_text(self, chunk: MultimodalChunk) -> str:
        """تحضير نص المعادلة للـ embedding"""
        # التركيز على: اللاتكس + المتغيرات + السياق
        latex = chunk.metadata.get('latex', '')
        variables = ' '.join(chunk.metadata.get('variables', []))
        context = chunk.metadata.get('context', '')
        section = chunk.metadata.get('section', '')
        
        return f"Equation: {latex} Variables: {variables} Context: {context} Section: {section}"
    
    def _prepare_table_text(self, chunk: MultimodalChunk) -> str:
        """تحضير نص الجدول للـ embedding"""
        # التركيز على: caption + headers + بعض البيانات
        caption = chunk.metadata.get('caption', '')
        headers = ' '.join(chunk.metadata.get('headers', []))
        context = chunk.metadata.get('context', '')
        
        # أخذ أول 500 حرف من الـ markdown
        markdown = chunk.metadata.get('markdown', '')[:500]
        
        return f"Table: {caption} Headers: {headers} Data: {markdown} Context: {context}"
    
    def _prepare_figure_text(self, chunk: MultimodalChunk) -> str:
        """تحضير نص الصورة للـ embedding"""
        caption = chunk.metadata.get('caption', '')
        context = chunk.metadata.get('context', '')
        section = chunk.metadata.get('section', '')
        
        return f"Figure: {caption} Context: {context} Section: {section}"
    
    def _prepare_general_text(self, chunk: MultimodalChunk) -> str:
        """تحضير نص عام للـ embedding"""
        # استخدام النص كما هو مع إضافة keywords
        keywords = ' '.join(chunk.metadata.get('keywords', []))
        section = chunk.metadata.get('section', '')
        
        return f"{chunk.text} Keywords: {keywords} Section: {section}"


if __name__ == "__main__":
    print("✅ SpecializedChunker & Embedder V2.0 Ready")
    print("\nFeatures:")
    print("  - Separate chunking strategies per content type")
    print("  - Rich metadata extraction")
    print("  - Context-aware chunking")
    print("  - Priority-based scoring")
    print("  - Specialized embedding preparation")
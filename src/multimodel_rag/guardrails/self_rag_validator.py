"""
self_rag_validator.py - نظام Self-RAG للحماية من الهلوسة V2.0
================================================================
✅ التحقق من وجود العناصر المذكورة في الرد
✅ التحقق من صحة الأرقام
✅ التحقق من عدم الخلط بين الأنواع
✅ التحقق من عدم اختراع معلومات
✅ تقييم جودة الرد
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)




def _simple_relevance(query: str, chunks: List[Any]) -> float:
    """Simple relevance: fraction of query terms that appear in retrieved chunks."""
    if not query or not chunks:
        return 0.0
    q_terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
    if not q_terms:
        return 0.0
    ctx_parts = []
    for c in chunks:
        if hasattr(c, 'text') and isinstance(getattr(c, 'text'), str):
            ctx_parts.append(getattr(c, 'text'))
        else:
            ctx_parts.append(str(c))
    ctx = " ".join(ctx_parts).lower()
    hits = sum(1 for t in q_terms if t in ctx)
    return hits / max(1, len(q_terms))
class ValidationLevel(Enum):
    """مستويات التحقق"""
    STRICT = "strict"  # صارم جداً
    MODERATE = "moderate"  # متوسط
    LENIENT = "lenient"  # متساهل


@dataclass
class ValidationResult:
    """نتيجة التحقق"""
    passed: bool
    confidence: float  # 0.0 - 1.0
    issues: List[str]
    corrections: Dict[str, str]
    metadata: Dict[str, Any]


class SelfRAGValidator:
    """
    نظام Self-RAG للتحقق من جودة الردود
    """
    
    # Patterns للكشف عن الإشارات
    EQUATION_REF_PATTERN = re.compile(
        r'\b(?:Equation|Eq\.|equation)\s*#?\s*(\d+)\b',
        re.IGNORECASE
    )
    
    TABLE_REF_PATTERN = re.compile(
        r'\b(?:Table|Tbl\.|table)\s*#?\s*(\d+)\b',
        re.IGNORECASE
    )
    
    FIGURE_REF_PATTERN = re.compile(
        r'\b(?:Figure|Fig\.|figure)\s*#?\s*(\d+)\b',
        re.IGNORECASE
    )
    
    # كلمات تدل على هلوسة محتملة
    HALLUCINATION_INDICATORS = [
        'probably', 'might be', 'possibly', 'perhaps', 'maybe',
        'could be', 'seems to be', 'appears to be', 'likely',
        'I think', 'I believe', 'I assume', 'presumably'
    ]
    
    def __init__(
        self,
        registry: Any,
        level: ValidationLevel = ValidationLevel.STRICT
    ):
        self.registry = registry
        self.level = level
        
        logger.info(f"✅ SelfRAGValidator initialized (level: {level.value})")
    
    def validate_response(
        self,
        response: str,
        query: str,
        intent: Any,  # QueryIntent
        retrieved_chunks: List[Any]
    ) -> ValidationResult:
        """
        التحقق الشامل من الرد
        """
        issues = []
        corrections = {}
        confidence = 1.0
        
        # 1. التحقق من الإشارات للعناصر
        ref_check = self._validate_element_references(response)
        if not ref_check['valid']:
            issues.extend(ref_check['issues'])
            corrections.update(ref_check['corrections'])
            confidence *= 0.8
        
        # 2. التحقق من عدم الخلط بين الأنواع
        type_check = self._validate_type_consistency(response, intent)
        if not type_check['valid']:
            issues.extend(type_check['issues'])
            confidence *= 0.7
        
        # 3. التحقق من عدم اختراع معلومات
        hallucination_check = self._detect_hallucinations(response, retrieved_chunks)
        if not hallucination_check['valid']:
            issues.extend(hallucination_check['issues'])
            confidence *= 0.6
        
        # 4. التحقق من الاكتمال
        completeness_check = self._validate_completeness(response, query, intent)
        if not completeness_check['valid']:
            issues.extend(completeness_check['issues'])
            confidence *= 0.9
        
        # 5. التحقق من عدم التكرار
        repetition_check = self._detect_repetition(response)
        if not repetition_check['valid']:
            issues.extend(repetition_check['issues'])
            confidence *= 0.95
        # ===================== HARD RELEVANCE GATE =====================
        relevance = _simple_relevance(query, retrieved_chunks)
        min_rel = 0.60 if self.level == ValidationLevel.STRICT else 0.45
        if relevance < min_rel:
            issues.append(
                f"Low relevance to query (relevance={relevance:.2f} < {min_rel:.2f})."
            )
            confidence *= 0.4


        
        # قرار نهائي
        passed = self._make_decision(confidence, issues)
        
        # force_fail_on_low_relevance: if we flagged low relevance, do not pass.
        if any('Low relevance to query' in s for s in issues):
            passed = False
        return ValidationResult(
            passed=passed,
            confidence=confidence,
            issues=issues,
            corrections=corrections,
            metadata={
                'total_issues': len(issues),
                'has_corrections': bool(corrections),
                'validation_level': self.level.value
            }
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    #  VALIDATION METHODS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _validate_element_references(self, response: str) -> Dict[str, Any]:
        """
        التحقق من صحة الإشارات للعناصر
        """
        issues = []
        corrections = {}
        
        # فحص المعادلات
        eq_refs = self.EQUATION_REF_PATTERN.findall(response)
        for eq_num_str in eq_refs:
            eq_num = int(eq_num_str)
            if not self._element_exists('equation', eq_num):
                issues.append(f"Equation {eq_num} does not exist")
                # إيجاد أقرب معادلة موجودة
                nearest = self._find_nearest_element('equation', eq_num)
                if nearest:
                    corrections[f"Equation {eq_num}"] = f"Equation {nearest}"
        
        # فحص الجداول
        table_refs = self.TABLE_REF_PATTERN.findall(response)
        for table_num_str in table_refs:
            table_num = int(table_num_str)
            if not self._element_exists('table', table_num):
                issues.append(f"Table {table_num} does not exist")
                nearest = self._find_nearest_element('table', table_num)
                if nearest:
                    corrections[f"Table {table_num}"] = f"Table {nearest}"
        
        # فحص الصور
        fig_refs = self.FIGURE_REF_PATTERN.findall(response)
        for fig_num_str in fig_refs:
            fig_num = int(fig_num_str)
            if not self._element_exists('figure', fig_num):
                issues.append(f"Figure {fig_num} does not exist")
                nearest = self._find_nearest_element('figure', fig_num)
                if nearest:
                    corrections[f"Figure {fig_num}"] = f"Figure {nearest}"
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'corrections': corrections
        }
    
    def _validate_type_consistency(
        self,
        response: str,
        intent: Any
    ) -> Dict[str, Any]:
        """
        التحقق من عدم الخلط بين الأنواع
        """
        from multimodel_rag.retrieval.smart_retriever import QueryType
        
        issues = []
        
        # إذا كان السؤال عن معادلات فقط
        if intent.query_type == QueryType.EQUATION:
            # لا يجب أن يذكر جداول أو صور
            if self.TABLE_REF_PATTERN.search(response):
                issues.append("Response mentions tables when query is about equations")
            if self.FIGURE_REF_PATTERN.search(response):
                issues.append("Response mentions figures when query is about equations")
        
        # إذا كان السؤال عن جداول فقط
        elif intent.query_type == QueryType.TABLE:
            if self.EQUATION_REF_PATTERN.search(response):
                issues.append("Response mentions equations when query is about tables")
            if self.FIGURE_REF_PATTERN.search(response):
                issues.append("Response mentions figures when query is about tables")
        
        # إذا كان السؤال عن صور فقط
        elif intent.query_type == QueryType.FIGURE:
            if self.EQUATION_REF_PATTERN.search(response):
                issues.append("Response mentions equations when query is about figures")
            if self.TABLE_REF_PATTERN.search(response):
                issues.append("Response mentions tables when query is about figures")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _detect_hallucinations(
        self,
        response: str,
        retrieved_chunks: List[Any]
    ) -> Dict[str, Any]:
        """
        كشف الهلوسة (اختراع معلومات)
        """
        issues = []
        
        # 1. فحص كلمات مشبوهة
        response_lower = response.lower()
        suspicious_phrases = [
            phrase for phrase in self.HALLUCINATION_INDICATORS
            if phrase in response_lower
        ]
        
        if len(suspicious_phrases) > 2:
            issues.append(f"Response contains uncertain language: {', '.join(suspicious_phrases[:3])}")
        
        # 2. فحص الأرقام المحددة
        # إذا ذكر أرقاماً محددة، يجب أن تكون من الـ chunks
        numbers_in_response = re.findall(r'\b\d+\.?\d*\b', response)
        numbers_in_chunks = []
        
        for chunk in retrieved_chunks:
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            numbers_in_chunks.extend(re.findall(r'\b\d+\.?\d*\b', chunk_text))
        
        # فحص عينة من الأرقام (لا نفحص الكل لأنه قد يكون بطيئاً)
        suspicious_numbers = []
        for num in numbers_in_response[:10]:  # أول 10 أرقام فقط
            if num not in numbers_in_chunks and len(num) > 2:  # تجاهل الأرقام الصغيرة جداً
                suspicious_numbers.append(num)
        
        if len(suspicious_numbers) > 3:
            issues.append(f"Response contains numbers not found in source: {', '.join(suspicious_numbers[:3])}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _validate_completeness(
        self,
        response: str,
        query: str,
        intent: Any
    ) -> Dict[str, Any]:
        """
        التحقق من اكتمال الرد
        """
        from multimodel_rag.retrieval.smart_retriever import QueryType
        
        issues = []
        
        # التحقق من الحد الأدنى للطول
        if len(response.strip()) < 50:
            issues.append("Response is too short")
        
        # إذا كان سؤالاً عن عنصر محدد
        if intent.query_type == QueryType.SPECIFIC_ELEMENT:
            # يجب أن يذكر رقم العنصر
            target_num = intent.target_number
            if str(target_num) not in response:
                issues.append(f"Response does not mention {intent.target_type} {target_num}")
        
        # إذا كان "show all"
        elif intent.query_type == QueryType.LIST_ALL:
            # يجب أن يحتوي على قائمة
            if not any(marker in response for marker in ['1.', '2.', '-', '*', '•']):
                issues.append("Response should contain a list format")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _detect_repetition(self, response: str) -> Dict[str, Any]:
        """
        كشف التكرار المفرط
        """
        issues = []
        
        # تقسيم إلى جمل
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        
        # فحص التكرار
        unique_sentences = set(sentences)
        
        if len(sentences) > 5 and len(unique_sentences) < len(sentences) * 0.7:
            repetition_rate = 1 - (len(unique_sentences) / len(sentences))
            issues.append(f"High repetition rate: {repetition_rate:.1%}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    #  HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _element_exists(self, element_type: str, number: int) -> bool:
        """التحقق من وجود عنصر"""
        if not self.registry:
            return True  # تجاهل إذا لم يكن هناك registry
        
        # استخدام الطريقة الصحيحة حسب نوع العنصر
        try:
            if element_type == 'equation':
                return number in self.registry.equations
            elif element_type == 'table':
                return number in self.registry.tables
            elif element_type == 'figure':
                return number in self.registry.figures
            else:
                return True
        except Exception:
            return True  # في حالة الخطأ، نعتبره موجود لتجنب false positives
    
    def _find_nearest_element(
        self,
        element_type: str,
        target_number: int
    ) -> Optional[int]:
        """إيجاد أقرب عنصر موجود"""
        if not self.registry:
            return None
        
        # الحصول على جميع الأرقام المتاحة
        registry_dict = getattr(self.registry, f"{element_type}s", {})
        if not registry_dict:
            return None
        
        available_numbers = list(registry_dict.keys())
        if not available_numbers:
            return None
        
        # إيجاد الأقرب
        nearest = min(available_numbers, key=lambda x: abs(x - target_number))
        return nearest
    
    def _make_decision(self, confidence: float, issues: List[str]) -> bool:
        """اتخاذ القرار النهائي"""
        
        if self.level == ValidationLevel.STRICT:
            # صارم: لا يسمح بأي مشاكل
            return confidence >= 0.95 and len(issues) == 0
        
        elif self.level == ValidationLevel.MODERATE:
            # متوسط: يسمح ببعض المشاكل البسيطة
            return confidence >= 0.80 and len(issues) <= 2
        
        else:  # LENIENT
            # متساهل: يسمح بمشاكل أكثر
            return confidence >= 0.60 and len(issues) <= 5
    
    def auto_correct_response(
        self,
        response: str,
        validation: ValidationResult
    ) -> str:
        """
        تصحيح تلقائي للرد
        """
        corrected = response
        
        # تطبيق التصحيحات
        for wrong, correct in validation.corrections.items():
            corrected = corrected.replace(wrong, correct)
        
        # إضافة تحذيرات إذا لزم الأمر
        if validation.issues and not validation.passed:
            warning = "\n\n⚠️ **Note:** Some information in this response may be uncertain."
            corrected += warning
        
        return corrected


class ResponseQualityAssessor:
    """
    تقييم جودة الردود
    """
    
    @staticmethod
    def assess_quality(response: str, query: str) -> Dict[str, Any]:
        """
        تقييم جودة الرد
        """
        scores = {
            'completeness': 0.0,
            'relevance': 0.0,
            'clarity': 0.0,
            'conciseness': 0.0,
            'overall': 0.0
        }
        
        # 1. Completeness (هل الرد كامل؟)
        if len(response.strip()) > 100:
            scores['completeness'] = min(1.0, len(response) / 500)
        
        # 2. Relevance (هل الرد ذو صلة؟)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        scores['relevance'] = min(1.0, overlap / max(len(query_words), 1))
        
        # 3. Clarity (هل الرد واضح؟)
        # عدد الجمل ÷ عدد الكلمات (كلما كانت الجمل أقصر كان أوضح)
        sentences = len(re.split(r'[.!?]+', response))
        words = len(response.split())
        avg_sentence_length = words / max(sentences, 1)
        scores['clarity'] = 1.0 / (1 + (avg_sentence_length / 20))  # أفضل طول: 20 كلمة/جملة
        
        # 4. Conciseness (هل الرد مختصر؟)
        # عقوبة للطول المفرط
        if len(response) > 1000:
            scores['conciseness'] = max(0.5, 1.0 - (len(response) - 1000) / 2000)
        else:
            scores['conciseness'] = 1.0
        
        # Overall score
        scores['overall'] = sum(scores.values()) / 4
        
        return scores


if __name__ == "__main__":
    print("✅ SelfRAGValidator V2.0 Ready")
    print("\nFeatures:")
    print("  - Element reference validation")
    print("  - Type consistency checking")
    print("  - Hallucination detection")
    print("  - Completeness validation")
    print("  - Repetition detection")
    print("  - Auto-correction")
    print("  - Quality assessment")
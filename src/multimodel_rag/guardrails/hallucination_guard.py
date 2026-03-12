"""
hallucination_guard.py - Hallucination Detection & Reference Validation
========================================================================
Extracted from response_formatter.py (Step 2 refactor).

Contains:
  • DocumentElementRegistry  — int-keyed registry of equations/tables/figures
  • HallucinationGuard       — validates and auto-corrects LLM references
  • _nearest()               — closest-number helper

Imported by: response_formatter.py, multimodal_agentic_rag.py (if needed directly)
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _nearest(ref: int, valid: Set[int]) -> Optional[int]:
    """Return the closest integer in *valid* to *ref*, or None if empty."""
    if not valid:
        return None
    return min(valid, key=lambda v: abs(v - ref))


# ─────────────────────────────────────────────────────────────────────────────
#  DOCUMENT ELEMENT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class DocumentElementRegistry:
    """
    Registry of all valid elements in the loaded document.
    Keyed by integer element number for O(1) lookup.
    """

    def __init__(self):
        self.equations: Dict[int, Dict[str, Any]] = {}
        self.figures:   Dict[int, Dict[str, Any]] = {}
        self.tables:    Dict[int, Dict[str, Any]] = {}
        self.initialized: bool = False

    def load_from_processed_document(self, processed_doc: Any) -> None:
        """Populate registries from a ProcessedDocument object."""

        def _safe_int_dict(src: Any) -> Dict[int, Dict[str, Any]]:
            if not src:
                return {}
            try:
                return {int(k): v for k, v in src.items()}
            except Exception:
                return {}

        self.equations = _safe_int_dict(
            getattr(processed_doc, "equation_registry", None)
        )
        self.figures = _safe_int_dict(
            getattr(processed_doc, "figure_registry", None)
        )
        self.tables = _safe_int_dict(
            getattr(processed_doc, "table_registry", None)
        )
        self.initialized = True
        logger.info(
            "Registry loaded: %d eqs, %d figs, %d tables",
            len(self.equations), len(self.figures), len(self.tables),
        )

    def register_equation(self, num: int, data: Dict[str, Any]) -> None:
        self.equations[num] = data
        self.initialized = True

    def register_figure(self, num: int, data: Dict[str, Any]) -> None:
        self.figures[num] = data
        self.initialized = True

    def register_table(self, num: int, data: Dict[str, Any]) -> None:
        self.tables[num] = data
        self.initialized = True

    def clear(self) -> None:
        self.equations = {}
        self.figures   = {}
        self.tables    = {}
        self.initialized = False

    def get_all_equations(self) -> List[Dict[str, Any]]:
        """Get all equations sorted by number"""
        return [self.equations[k] for k in sorted(self.equations.keys())]

    def get_all_tables(self) -> List[Dict[str, Any]]:
        """Get all tables sorted by number"""
        return [self.tables[k] for k in sorted(self.tables.keys())]

    def get_all_figures(self) -> List[Dict[str, Any]]:
        """Get all figures sorted by number"""
        return [self.figures[k] for k in sorted(self.figures.keys())]


# ─────────────────────────────────────────────────────────────────────────────
#  HALLUCINATION GUARD
# ─────────────────────────────────────────────────────────────────────────────

class HallucinationGuard:
    """
    Validate and correct references to equations/tables/figures in LLM output.
    Auto-corrects invalid numbers to the nearest valid element,
    or flags them as [NOT FOUND] when no valid alternative exists.
    """

    # Reference patterns — catches "Equation 3", "Eq. 3", "eq. 3", etc.
    EQ_PATTERN  = re.compile(r'\b(?:Equation|Eq\.|eq\.|equation)\s*#?\s*(\d+)\b', re.IGNORECASE)
    TBL_PATTERN = re.compile(r'\b(?:Table|Tbl\.|table)\s*#?\s*(\d+)\b',           re.IGNORECASE)
    FIG_PATTERN = re.compile(r'\b(?:Figure|Fig\.|figure)\s*#?\s*(\d+)\b',          re.IGNORECASE)

    def __init__(self, registry: DocumentElementRegistry):
        self.registry = registry

    def validate_and_correct(self, text: str, strict: bool = False) -> str:
        """
        Validate references and correct hallucinations.

        Args:
            text:   LLM-generated text to validate.
            strict: If True, raise on invalid refs. If False, auto-correct.

        Returns:
            Corrected text.
        """
        if not self.registry.initialized:
            return text

        eq_refs  = list(self.EQ_PATTERN.finditer(text))
        tbl_refs = list(self.TBL_PATTERN.finditer(text))
        fig_refs = list(self.FIG_PATTERN.finditer(text))

        corrections: List[Tuple[Tuple[int, int], str]] = []

        # Validate equations
        valid_eqs = set(self.registry.equations.keys())
        for match in eq_refs:
            num = int(match.group(1))
            if num not in valid_eqs:
                nearest = _nearest(num, valid_eqs)
                corrections.append((match.span(), f"Equation {nearest}" if nearest else "[Equation NOT FOUND]"))

        # Validate tables
        valid_tbls = set(self.registry.tables.keys())
        for match in tbl_refs:
            num = int(match.group(1))
            if num not in valid_tbls:
                nearest = _nearest(num, valid_tbls)
                corrections.append((match.span(), f"Table {nearest}" if nearest else "[Table NOT FOUND]"))

        # Validate figures
        valid_figs = set(self.registry.figures.keys())
        for match in fig_refs:
            num = int(match.group(1))
            if num not in valid_figs:
                nearest = _nearest(num, valid_figs)
                corrections.append((match.span(), f"Figure {nearest}" if nearest else "[Figure NOT FOUND]"))

        # Apply corrections in reverse span order to preserve positions
        if corrections:
            corrections.sort(key=lambda x: x[0][0], reverse=True)
            for (start, end), replacement in corrections:
                text = text[:start] + replacement + text[end:]

        return text

    def validate_and_fix(self, text: str, strict: bool = False) -> Tuple[str, List[str]]:
        """
        Alias returning (corrected_text, warnings_list).
        Backward-compatible with code expecting a tuple.
        """
        corrected = self.validate_and_correct(text, strict)
        warnings: List[str] = []
        if corrected != text:
            if "[Equation NOT FOUND]" in corrected:
                warnings.append("Some equation references were invalid")
            if "[Table NOT FOUND]" in corrected:
                warnings.append("Some table references were invalid")
            if "[Figure NOT FOUND]" in corrected:
                warnings.append("Some figure references were invalid")
        return corrected, warnings

    def detect_hallucinations(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect invalid references without correcting them.

        Returns:
            (has_hallucinations, list_of_issues)
        """
        if not self.registry.initialized:
            return False, []

        issues: List[str] = []

        for match in self.EQ_PATTERN.finditer(text):
            num = int(match.group(1))
            if num not in self.registry.equations:
                issues.append(f"Equation {num} not found")

        for match in self.TBL_PATTERN.finditer(text):
            num = int(match.group(1))
            if num not in self.registry.tables:
                issues.append(f"Table {num} not found")

        for match in self.FIG_PATTERN.finditer(text):
            num = int(match.group(1))
            if num not in self.registry.figures:
                issues.append(f"Figure {num} not found")

        return bool(issues), issues


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick sanity check
    reg = DocumentElementRegistry()
    reg.register_equation(1, {"global_number": 1, "text": "E=mc^2"})
    reg.register_equation(3, {"global_number": 3, "text": "KL(p||q)"})
    reg.register_table(1, {"global_number": 1, "caption": "Results"})
    reg.register_figure(2, {"global_number": 2, "caption": "Architecture"})

    guard = HallucinationGuard(reg)

    test_text = "As shown in Equation 5 and Table 3, and Figure 9..."
    corrected, warnings = guard.validate_and_fix(test_text)

    print("Original : ", test_text)
    print("Corrected:", corrected)
    print("Warnings :", warnings)
    assert "[Equation NOT FOUND]" not in corrected  # Eq 5 → nearest is Eq 3
    assert "Equation 3" in corrected
    assert "[Table NOT FOUND]" in corrected          # Table 3 doesn't exist
    print("✅ hallucination_guard.py sanity check passed")
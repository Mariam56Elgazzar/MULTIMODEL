"""
pdf_processor_v2.py - Production PDF Processor (THE PARAGRAPH KILLER)
=====================================================================
✅ Paragraph Killer (Stops extracting when it hits paragraph text)
✅ Layout Engine (Perfect spatial representation)
✅ Column-Cropping (Prevents Table 1 & 2 from merging)
✅ FIXED: Safe optional OCR import
✅ FIXED: Figure section/image_path positional bug
✅ FIXED: Safer OCR fallback behavior
✅ FIXED: Keyword-based Processed* construction
"""

import os
import re
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import fitz  # PyMuPDF
import pdfplumber

from multimodel_rag.utils.equation_latex import to_latex
from multimodel_rag.core.models import (
    ProcessedDocument,
    ProcessedEquation,
    ProcessedTable,
    ProcessedFigure,
)

logger = logging.getLogger(__name__)

# Optional OCR agent import so the app still runs if the file is not added yet
try:
    from multimodel_rag.processing.equation_ocr import EquationOCRService
    HAS_EQUATION_OCR = True
except Exception as e:
    EquationOCRService = None
    HAS_EQUATION_OCR = False
    logger.warning("Equation OCR service not available: %s", e)


class StrictEquationDetector:
    STRONG_OPERATORS = ['=', '≈', '≤', '≥', '∫', '∑', '∏', '∝', '→', '∈']

    @classmethod
    def is_equation(cls, text: str, bbox: Tuple[float, float, float, float] = None) -> bool:
        text = (text or "").strip()
        if len(text) < 8 or len(text) > 300:
            return False
        if re.search(r'(https?://|www\.|doi:|arxiv:|github\.com|\.pdf)', text, re.IGNORECASE):
            return False
        if re.match(r'^\[\d+\]', text):
            return False
        if re.match(r'^\s*(Figure|Fig|Table|Appendix|Algorithm|Listing)\b', text, re.IGNORECASE):
            return False
        if 'id=' in text.lower() or 'id =' in text.lower():
            return False
        if text.startswith(')'):
            return False
        if not any(op in text for op in cls.STRONG_OPERATORS):
            return False

        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        if len(words) > 6:
            return False
        return True

    @classmethod
    def extract_latex_from_text(cls, text: str) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return ""
        if '\\' in text:
            return text

        latex = text.replace('\n', ' ')

        greek_map = {
            'α': r'\alpha ',
            'β': r'\beta ',
            'γ': r'\gamma ',
            'δ': r'\delta ',
            'η': r'\eta ',
            'θ': r'\theta ',
            'λ': r'\lambda ',
            'μ': r'\mu ',
        }
        for greek, tex in greek_map.items():
            latex = latex.replace(greek, tex)

        latex = re.sub(r'≈\s*X', r'\\approx \\sum', latex)
        latex = re.sub(r'=\s*X', r'= \\sum', latex)
        latex = latex.replace('∑', r'\sum ').replace('∏', r'\prod ')

        latex = latex.replace('\ufffd', '').replace('\x01', '').replace('\x00', '')
        latex = latex.replace('exp  d(z)', 'exp(d(z))').replace('exp d(z)', 'exp(d(z))')
        return latex.strip()


class TableDetector:

    @staticmethod
    def _cells_to_markdown(rows: List[List[Any]]) -> str:
        def clean(cell):
            if cell is None:
                return ""
            return re.sub(r'\s+', ' ', str(cell)).strip()

        norm = [[clean(c) for c in row] for row in rows if any(c for c in row)]
        if not norm:
            return ""

        ncols = max(len(r) for r in norm)
        norm = [r + [""] * (ncols - len(r)) for r in norm]

        widths = [max(len(row[i]) for row in norm) for i in range(ncols)]
        widths = [max(w, 3) for w in widths]

        def fmt_row(row):
            return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

        header = fmt_row(norm[0])
        separator = "| " + " | ".join("-" * widths[i] for i in range(ncols)) + " |"
        data_rows = [fmt_row(r) for r in norm[1:]]

        return "\n".join([header, separator] + data_rows)

    @staticmethod
    def _words_to_markdown(words: List[Dict]) -> str:
        if not words:
            return ""

        words_sorted = sorted(words, key=lambda w: (round(w["top"] / 5) * 5, w["x0"]))
        rows_of_words: List[List[Dict]] = []
        cur_row: List[Dict] = []
        prev_top = None
        ROW_GAP = 8

        for w in words_sorted:
            if prev_top is None or abs(w["top"] - prev_top) <= ROW_GAP:
                cur_row.append(w)
            else:
                if cur_row:
                    rows_of_words.append(sorted(cur_row, key=lambda x: x["x0"]))
                cur_row = [w]
            prev_top = w["top"]

        if cur_row:
            rows_of_words.append(sorted(cur_row, key=lambda x: x["x0"]))

        if len(rows_of_words) < 2:
            return ""

        def looks_like_prose_row(row: List[Dict]) -> bool:
            texts = [w["text"] for w in row]
            total_chars = sum(len(t) for t in texts)

            if len(texts) == 1:
                t = texts[0]
                if len(t) > 15:
                    alpha = sum(c.isalpha() for c in t)
                    digits = sum(c.isdigit() for c in t)
                    if len(t) > 0 and alpha / len(t) > 0.80 and digits < 3:
                        return True

            if total_chars > 60:
                digit_count = sum(c.isdigit() for t in texts for c in t)
                special_count = sum(1 for t in texts for c in t if c in '.-/+*%')
                if digit_count == 0 and special_count < 2:
                    return True

            return False

        keep_rows: List[List[Dict]] = []
        consecutive_prose = 0

        for row in rows_of_words:
            if looks_like_prose_row(row):
                consecutive_prose += 1
                if consecutive_prose >= 1 and keep_rows:
                    break
            else:
                consecutive_prose = 0
                keep_rows.append(row)

        rows_of_words = keep_rows
        if len(rows_of_words) < 2:
            return ""

        def gaps_from_row(row: List[Dict], min_gap: float) -> List[float]:
            if not row:
                return []

            splits = []
            prev_x1 = row[0].get("x1", row[0]["x0"] + 4)
            for w in row[1:]:
                gap_start = prev_x1
                gap_end = w["x0"]
                gap_size = gap_end - gap_start
                if gap_size >= min_gap:
                    splits.append((gap_start + gap_end) / 2.0)
                prev_x1 = max(prev_x1, w.get("x1", w["x0"] + 4))
            return splits

        densest_row = max(rows_of_words, key=len)
        MIN_GAP = 4.0
        splits = gaps_from_row(densest_row, MIN_GAP)

        if not splits:
            all_splits = set()
            for row in rows_of_words:
                for s in gaps_from_row(row, MIN_GAP):
                    all_splits.add(round(s / 2) * 2)
            splits = sorted(all_splits)

        if not splits:
            return ""

        x_min = min(w["x0"] for w in words)
        x_max = max(w.get("x1", w["x0"] + 1) for w in words)
        boundaries = [x_min - 1] + splits + [x_max + 1]
        ncols = len(boundaries) - 1

        if ncols < 2:
            return ""

        def col_index(cx: float) -> int:
            for i in range(ncols):
                if boundaries[i] <= cx < boundaries[i + 1]:
                    return i
            return ncols - 1

        grid: List[List[str]] = []
        for row in rows_of_words:
            cells: List[List[str]] = [[] for _ in range(ncols)]
            for w in row:
                cx = (w["x0"] + w.get("x1", w["x0"])) / 2.0
                cells[col_index(cx)].append(w["text"])
            row_strs = [" ".join(c) for c in cells]
            if any(s for s in row_strs):
                grid.append(row_strs)

        while ncols > 1 and all(r[ncols - 1] == "" for r in grid):
            grid = [r[:-1] for r in grid]
            ncols -= 1

        if ncols < 2:
            return ""

        return TableDetector._cells_to_markdown(grid)

    @staticmethod
    def _layout_text_to_markdown(raw_text: str, paragraph_killer: bool = True) -> str:
        lines = []
        empty_streak = 0

        for line in raw_text.split('\n'):
            stripped = line.strip()

            if not stripped:
                empty_streak += 1
                if empty_streak >= 2 and lines:
                    break
                continue

            empty_streak = 0
            if paragraph_killer:
                if len(stripped) > 30 and ' ' not in stripped:
                    if lines:
                        break
                if re.match(r'^\d+\.\d+\s+[A-Z]', stripped):
                    if lines:
                        break
                alpha_ratio = sum(c.isalpha() for c in stripped) / len(stripped) if stripped else 0
                if len(stripped) > 50 and alpha_ratio > 0.85 and not re.search(r'\s{3,}', line):
                    if lines:
                        break

            lines.append(line.rstrip())

        if len(lines) < 2:
            return ""

        return "```text\n" + "\n".join(lines) + "\n```"

    @staticmethod
    def extract_table_markdown(plumber_page, fitz_page) -> List[Dict[str, Any]]:
        tables = []
        try:
            table_captions = []
            blocks = fitz_page.get_text("blocks")

            for b in blocks:
                if b[6] == 0:
                    text = b[4].strip()
                    if re.match(r'^Table\s+\d+[:\.]', text, re.IGNORECASE) and len(text.split()) < 40:
                        table_captions.append({"text": text.replace('\n', ' '), "bbox": b[:4]})

            if not table_captions:
                return tables

            page_width = fitz_page.rect.width
            page_height = fitz_page.rect.height

            try:
                native_tables = plumber_page.find_tables()
            except Exception:
                native_tables = []

            for cap in table_captions:
                cx0, cy0, cx1, cy1 = cap["bbox"]

                is_left = cx1 < (page_width / 2) + 20
                is_right = cx0 > (page_width / 2) - 20
                crop_x0 = 0 if is_left else (page_width / 2)
                crop_x1 = (page_width / 2) if (is_left and not is_right) else page_width
                if is_left and is_right:
                    crop_x0, crop_x1 = 0, page_width

                search_box = (crop_x0, cy1, crop_x1, min(page_height, cy1 + 350))
                matched_native = None

                for nt in native_tables:
                    tb = nt.bbox
                    if (
                        tb[0] < search_box[2] and tb[2] > search_box[0]
                        and tb[1] < search_box[3] and tb[3] > search_box[1]
                    ):
                        matched_native = nt
                        break

                md_text = ""
                crop_box = (crop_x0, max(0, cy1 - 2), crop_x1, min(page_height, cy1 + 350))

                if matched_native is not None:
                    try:
                        rows = matched_native.extract()
                        md_body = TableDetector._cells_to_markdown(rows)
                        if md_body:
                            md_text = md_body
                            crop_box = (
                                matched_native.bbox[0],
                                matched_native.bbox[1],
                                matched_native.bbox[2],
                                matched_native.bbox[3],
                            )
                    except Exception as e:
                        logger.debug("Native table extraction failed, falling back: %s", e)

                if not md_text:
                    try:
                        cropped_b = plumber_page.within_bbox(crop_box)
                        words = cropped_b.extract_words(
                            x_tolerance=4,
                            y_tolerance=4,
                            keep_blank_chars=False,
                            use_text_flow=False,
                        )
                        if words:
                            md_body = TableDetector._words_to_markdown(words)
                            if md_body:
                                md_text = md_body
                    except Exception as eb:
                        logger.debug("Word-cluster extraction failed: %s", eb)

                if not md_text:
                    try:
                        cropped_c = plumber_page.within_bbox(crop_box)
                        raw_text = cropped_c.extract_text(layout=True)
                        if raw_text and len(raw_text.strip()) >= 15:
                            md_text = TableDetector._layout_text_to_markdown(raw_text)
                    except Exception as ec:
                        logger.debug("Layout fallback failed: %s", ec)

                if not md_text:
                    continue

                plain = re.sub(r'[|`\-]+', ' ', md_text)
                plain = re.sub(r'\s+', ' ', plain).strip()

                tables.append({
                    "markdown": md_text,
                    "bbox": crop_box,
                    "caption": cap["text"],
                    "text": plain,
                })

        except Exception as e:
            logger.warning("Table extraction failed: %s", e)

        return tables


class SectionDetector:
    SECTION_PATTERNS = [
        r'^\d+\.?\s+[A-Z]',
        r'^[IVX]+\.?\s+[A-Z]',
        r'^(Abstract|Introduction|Methods|Results|Discussion|Conclusion|References)'
    ]

    @classmethod
    def detect_section(cls, text: str) -> Optional[str]:
        text = (text or "").strip()
        if len(text) > 100:
            return None
        for p in cls.SECTION_PATTERNS:
            if re.match(p, text, re.IGNORECASE):
                return text
        return None


@dataclass
class ExtractedEquation:
    text: str
    latex: str
    page_num: int
    bbox: Tuple[float, float, float, float]
    global_number: int
    section: str = ""


@dataclass
class ExtractedTable:
    text: str
    markdown: str
    page_num: int
    bbox: Tuple[float, float, float, float]
    global_number: int
    caption: str = ""
    section: str = ""


@dataclass
class ExtractedFigure:
    caption: str
    page_num: int
    bbox: Tuple[float, float, float, float]
    global_number: int
    image_path: Optional[str] = None
    section: str = ""


class PDFProcessorV2:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.eq_det = StrictEquationDetector()
        self.tbl_det = TableDetector()
        self.sec_det = SectionDetector()
        self.equation_ocr = None

        if (
            self.config.get("enable_equation_ocr")
            and self.config.get("groq_api_key")
            and HAS_EQUATION_OCR
        ):
            try:
                self.equation_ocr = EquationOCRService(
                    groq_api_key=self.config["groq_api_key"],
                    model=self.config.get(
                        "equation_ocr_model",
                        "meta-llama/llama-4-scout-17b-16e-instruct",
                    ),
                )
                logger.info("Equation OCR service initialized")
            except Exception as e:
                logger.warning("Failed to initialize Equation OCR service: %s", e)
                self.equation_ocr = None

    def _should_try_ocr(self, eq_text: str, latex: str) -> bool:
        eq_text = (eq_text or "").strip()
        latex = (latex or "").strip()

        if not eq_text:
            return True
        if '\ufffd' in eq_text or '\x00' in eq_text or '\x01' in eq_text:
            return True
        if len(eq_text) < 12:
            return True
        if re.search(r'\b[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\b', eq_text):
            return True
        if not latex:
            return True

        return False

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        self.eq_counter = 0
        self.tbl_counter = 0
        self.fig_counter = 0
        self.current_section = ""

        equations: List[ExtractedEquation] = []
        tables: List[ExtractedTable] = []
        figures: List[ExtractedFigure] = []
        page_texts: List[str] = []

        doc = None
        plumber_doc = None

        try:
            doc = fitz.open(pdf_path)
            plumber_doc = pdfplumber.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                plumber_page = plumber_doc.pages[page_num]

                page_tables = self.tbl_det.extract_table_markdown(plumber_page, page)
                for pt in page_tables:
                    self.tbl_counter += 1
                    tbl = ExtractedTable(
                        text=str(pt["text"]),
                        markdown=pt["markdown"],
                        page_num=page_num,
                        bbox=pt.get("bbox", (0, 0, 0, 0)),
                        global_number=self.tbl_counter,
                        caption=pt.get("caption", ""),
                        section=self.current_section,
                    )
                    tables.append(tbl)

                blocks = page.get_text("blocks")
                page_text_parts = []

                for b in blocks:
                    x0, y0, x1, y1, text, block_no, block_type = b
                    text = (text or "").strip()
                    bbox = (x0, y0, x1, y1)

                    if not text and block_type == 0:
                        continue

                    if block_type == 1:
                        self.fig_counter += 1
                        fig = ExtractedFigure(
                            caption=f"Image from page {page_num + 1}",
                            page_num=page_num,
                            bbox=bbox,
                            global_number=self.fig_counter,
                            image_path=None,
                            section=self.current_section,
                        )
                        figures.append(fig)
                        page_text_parts.append(f"[Figure {self.fig_counter}]")
                        continue

                    if block_type == 0:
                        if re.match(r'^\s*(Figure|Fig\.)\s*\d+', text, re.IGNORECASE):
                            self.fig_counter += 1
                            caption = text.replace('\n', ' ')
                            fig = ExtractedFigure(
                                caption=caption,
                                page_num=page_num,
                                bbox=bbox,
                                global_number=self.fig_counter,
                                image_path=None,
                                section=self.current_section,
                            )
                            figures.append(fig)
                            page_text_parts.append(f"[{caption}]")
                            continue

                        section_title = self.sec_det.detect_section(text)
                        if section_title:
                            self.current_section = section_title
                            page_text_parts.append(text)
                            continue

                        if self.eq_det.is_equation(text, bbox):
                            self.eq_counter += 1

                            eq_text = text.replace('\n', ' ').strip()
                            latex = self.eq_det.extract_latex_from_text(eq_text) or to_latex(eq_text)

                            if self.equation_ocr is not None and self._should_try_ocr(eq_text, latex):
                                try:
                                    ocr_result = self.equation_ocr.extract_from_bbox(
                                        pdf_path=pdf_path,
                                        page_num=page_num,
                                        bbox=bbox,
                                    )
                                    if ocr_result and getattr(ocr_result, "raw_text", "").strip():
                                        ocr_text = ocr_result.raw_text.strip()
                                        ocr_latex = getattr(ocr_result, "latex", "").strip()

                                        if len(ocr_text) >= max(8, len(eq_text) // 2):
                                            eq_text = ocr_text
                                            latex = ocr_latex or to_latex(eq_text) or latex
                                except Exception as e:
                                    logger.debug("Equation OCR failed on page %s: %s", page_num, e)

                            eq = ExtractedEquation(
                                text=eq_text,
                                latex=latex or "",
                                page_num=page_num,
                                bbox=bbox,
                                global_number=self.eq_counter,
                                section=self.current_section,
                            )
                            equations.append(eq)
                            page_text_parts.append(f"[Equation {self.eq_counter}]")
                            continue

                        page_text_parts.append(text)

                page_texts.append("\n\n".join(page_text_parts))

            return {
                'equations': equations,
                'tables': tables,
                'figures': figures,
                'page_texts': page_texts,
                'num_pages': len(doc),
            }

        finally:
            if doc is not None:
                doc.close()
            if plumber_doc is not None:
                plumber_doc.close()


class EnhancedPDFProcessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.processor = PDFProcessorV2(config or {})

    def process_pdf(self, pdf_path: str) -> ProcessedDocument:
        raw = self.processor.process_pdf(pdf_path)

        equations = [
            ProcessedEquation(
                equation_id=f"eq_{uuid.uuid4().hex[:8]}",
                global_number=e.global_number,
                text=e.text,
                latex=e.latex,
                page_number=e.page_num,
                bbox=e.bbox,
                section=e.section,
                raw_text=e.text,
            )
            for e in raw.get('equations', [])
        ]

        tables = [
            ProcessedTable(
                table_id=f"tb_{uuid.uuid4().hex[:8]}",
                global_number=t.global_number,
                caption=t.caption,
                page_number=t.page_num,
                markdown=t.markdown,
                section=t.section,
                bbox=t.bbox,
                raw_text=t.text,
            )
            for t in raw.get('tables', [])
        ]

        figures = [
            ProcessedFigure(
                figure_id=f"fig_{uuid.uuid4().hex[:8]}",
                global_number=f.global_number,
                caption=f.caption,
                page_number=f.page_num,
                saved_path=f.image_path,
                bbox=f.bbox,
                section=f.section,
            )
            for f in raw.get('figures', [])
        ]

        return ProcessedDocument(
            doc_id=f"doc_{uuid.uuid4().hex[:8]}",
            filename=os.path.basename(pdf_path),
            num_pages=raw.get('num_pages', 0),
            page_texts=raw.get('page_texts', []),
            enriched_page_texts=raw.get('page_texts', []),
            sections=[],
            equations=equations,
            tables=tables,
            figures=figures,
            title=os.path.basename(pdf_path),
        )
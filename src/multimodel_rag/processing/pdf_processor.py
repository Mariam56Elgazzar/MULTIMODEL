"""
pdf_processor_v2.py - Production PDF Processor (THE PARAGRAPH KILLER)
=====================================================================
✅ Paragraph Killer (Stops extracting when it hits paragraph text)
✅ Layout Engine (Perfect spatial representation)
✅ Column-Cropping (Prevents Table 1 & 2 from merging)
"""

import os
import re
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import fitz  # PyMuPDF
import pdfplumber

from multimodel_rag.core.models import ProcessedDocument, ProcessedEquation, ProcessedTable, ProcessedFigure

logger = logging.getLogger(__name__)

class StrictEquationDetector:
    STRONG_OPERATORS = ['=', '≈', '≤', '≥', '∫', '∑', '∏', '∝', '→', '∈']
    
    @classmethod
    def is_equation(cls, text: str, bbox: Tuple[float, float, float, float] = None) -> bool:
        text = text.strip()
        if len(text) < 8 or len(text) > 300: return False
        if re.search(r'(https?://|www\.|doi:|arxiv:|github\.com|\.pdf)', text, re.IGNORECASE): return False
        if re.match(r'^\[\d+\]', text): return False
        if re.match(r'^\s*(Figure|Fig|Table|Appendix|Algorithm|Listing)\b', text, re.IGNORECASE): return False
        if 'id=' in text.lower() or 'id =' in text.lower(): return False
        if text.startswith(')'): return False
        if not any(op in text for op in cls.STRONG_OPERATORS): return False

        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        if len(words) > 6: return False 
        return True

    @classmethod
    def extract_latex_from_text(cls, text: str) -> Optional[str]:
        if '\\' in text: return text
        latex = text.replace('\n', ' ')
        greek_map = {'α': r'\alpha ', 'β': r'\beta ', 'γ': r'\gamma ', 'δ': r'\delta ', 'η': r'\eta ', 'θ': r'\theta ', 'λ': r'\lambda ', 'μ': r'\mu '}
        for greek, tex in greek_map.items(): latex = latex.replace(greek, tex)
        
        latex = re.sub(r'≈\s*X', r'\\approx \\sum', latex)
        latex = re.sub(r'=\s*X', r'= \\sum', latex)
        latex = latex.replace('∑', r'\sum ').replace('∏', r'\prod ')
        
        latex = latex.replace('\ufffd', '').replace('\x01', '').replace('\x00', '')
        latex = latex.replace('exp  d(z)', 'exp(d(z))').replace('exp d(z)', 'exp(d(z))')
        return latex.strip()


class TableDetector:

    @staticmethod
    def _cells_to_markdown(rows: List[List[Any]]) -> str:
        """Convert a list-of-rows (each row is a list of cell strings) to a GitHub-flavored markdown table."""
        # Normalise every cell to a clean single-line string
        def clean(cell):
            if cell is None:
                return ""
            return re.sub(r'\s+', ' ', str(cell)).strip()

        norm = [[clean(c) for c in row] for row in rows if any(c for c in row)]
        if not norm:
            return ""

        # Determine column count
        ncols = max(len(r) for r in norm)
        # Pad short rows
        norm = [r + [""] * (ncols - len(r)) for r in norm]

        # Column widths for pretty alignment
        widths = [max(len(row[i]) for row in norm) for i in range(ncols)]
        widths = [max(w, 3) for w in widths]   # minimum 3 chars for separator dashes

        def fmt_row(row):
            return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

        header = fmt_row(norm[0])
        separator = "| " + " | ".join("-" * widths[i] for i in range(ncols)) + " |"
        data_rows = [fmt_row(r) for r in norm[1:]]

        return "\n".join([header, separator] + data_rows)

    @staticmethod
    def _words_to_markdown(words: List[Dict]) -> str:
        """
        Word-coordinate clustering for borderless academic tables.

        Algorithm:
        1. Group words into rows by y-proximity (ROW_GAP threshold).
        2. Use the row with the MOST words (typically the column-header row)
           to determine column split points by finding horizontal gaps between
           word spans in that row.
        3. If step 2 fails (only 1 column found), fall back to analysing ALL
           rows with an adaptive gap threshold.
        4. Assign every word to a column by checking which column interval
           contains the word's horizontal centre.
        5. Render as a GitHub-Flavored Markdown table.
        """
        if not words:
            return ""

        # ── 1. Cluster words into rows by y-coordinate ────────────────────
        words_sorted = sorted(words, key=lambda w: (round(w["top"] / 5) * 5, w["x0"]))
        rows_of_words: List[List[Dict]] = []
        cur_row: List[Dict] = []
        prev_top = None
        ROW_GAP = 8  # px – words within this vertical distance → same row

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

        # ── 1b. Paragraph-row killer ─────────────────────────────────────
        # Walk rows top-to-bottom; stop when we hit rows that look like prose
        # (squished long words from pdfplumber layout / paragraph continuation).
        def looks_like_prose_row(row: List[Dict]) -> bool:
            texts = [w["text"] for w in row]
            total_chars = sum(len(t) for t in texts)
            # Single very-long token with mostly letters and no digits
            if len(texts) == 1:
                t = texts[0]
                if len(t) > 15:
                    alpha = sum(c.isalpha() for c in t)
                    digits = sum(c.isdigit() for c in t)
                    if alpha / len(t) > 0.80 and digits < 3:
                        return True
            # Row of multiple words where the combined string is very long and
            # contains no digits at all (no numbers → no data → prose)
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
                    break  # first prose row after real data → stop
            else:
                consecutive_prose = 0
                keep_rows.append(row)

        rows_of_words = keep_rows
        if len(rows_of_words) < 2:
            return ""


        def gaps_from_row(row: List[Dict], min_gap: float) -> List[float]:
            """Return sorted list of column split x-midpoints from gaps in `row`."""
            if not row:
                return []
            splits = []
            prev_x1 = row[0].get("x1", row[0]["x0"] + 4)
            for w in row[1:]:
                gap_start = prev_x1
                gap_end   = w["x0"]
                gap_size  = gap_end - gap_start
                if gap_size >= min_gap:
                    splits.append((gap_start + gap_end) / 2.0)
                prev_x1 = max(prev_x1, w.get("x1", w["x0"] + 4))
            return splits

        # Try with the row that has the most words (most likely the header)
        densest_row = max(rows_of_words, key=len)
        MIN_GAP = 4.0  # pts – very permissive; pdfplumber coords are in pts
        splits = gaps_from_row(densest_row, MIN_GAP)

        # If densest row gives only 1 column, try EVERY row and union the splits
        if not splits:
            all_splits = set()
            for row in rows_of_words:
                for s in gaps_from_row(row, MIN_GAP):
                    # round to nearest 2 to cluster near-duplicates
                    all_splits.add(round(s / 2) * 2)
            splits = sorted(all_splits)

        # If still no splits, we can't build a table
        if not splits:
            return ""

        # Build boundary list
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

        # ── 3. Build 2-D grid ─────────────────────────────────────────────
        grid: List[List[str]] = []
        for row in rows_of_words:
            cells: List[List[str]] = [[] for _ in range(ncols)]
            for w in row:
                cx = (w["x0"] + w.get("x1", w["x0"])) / 2.0
                cells[col_index(cx)].append(w["text"])
            row_strs = [" ".join(c) for c in cells]
            # Skip completely empty rows
            if any(s for s in row_strs):
                grid.append(row_strs)


        # Trim trailing all-empty columns
        while ncols > 1 and all(r[ncols - 1] == "" for r in grid):
            grid = [r[:-1] for r in grid]
            ncols -= 1

        if ncols < 2:
            return ""

        return TableDetector._cells_to_markdown(grid)

    @staticmethod
    def _layout_text_to_markdown(raw_text: str, paragraph_killer: bool = True) -> str:
        """Fallback: convert pdfplumber layout-text to a best-effort markdown table."""
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
                    if lines: break
                if re.match(r'^\d+\.\d+\s+[A-Z]', stripped):
                    if lines: break
                alpha_ratio = sum(c.isalpha() for c in stripped) / len(stripped) if stripped else 0
                if len(stripped) > 50 and alpha_ratio > 0.85 and not re.search(r'\s{3,}', line):
                    if lines: break
            lines.append(line.rstrip())

        if len(lines) < 2:
            return ""
        return "```text\n" + "\n".join(lines) + "\n```"

    @staticmethod
    def extract_table_markdown(plumber_page, fitz_page) -> List[Dict[str, Any]]:
        tables = []
        try:
            # ── 1. CAPTION DETECTION via fitz ────────────────────────────────
            table_captions = []
            blocks = fitz_page.get_text("blocks")
            for b in blocks:
                if b[6] == 0:
                    text = b[4].strip()
                    if re.match(r'^Table\s+\d+[:\.]', text, re.IGNORECASE) and len(text.split()) < 40:
                        table_captions.append({"text": text.replace('\n', ' '), "bbox": b[:4]})

            if not table_captions:
                return tables

            page_width  = fitz_page.rect.width
            page_height = fitz_page.rect.height

            # ── 2. Try pdfplumber native table detection first ────────────────
            # Find all structured tables on the full page once (cheaper than per-caption)
            try:
                native_tables = plumber_page.find_tables()
            except Exception:
                native_tables = []

            for cap in table_captions:
                cx0, cy0, cx1, cy1 = cap["bbox"]

                # Column cropping: prevent left/right column tables from bleeding together
                is_left  = cx1 < (page_width / 2) + 20
                is_right = cx0 > (page_width / 2) - 20
                crop_x0 = 0 if is_left else (page_width / 2)
                crop_x1 = (page_width / 2) if (is_left and not is_right) else page_width
                if is_left and is_right:
                    crop_x0, crop_x1 = 0, page_width

                search_box = (crop_x0, cy1, crop_x1, min(page_height, cy1 + 350))

                # ── 2a. Check if any native table overlaps the caption region ─
                matched_native = None
                for nt in native_tables:
                    tb = nt.bbox  # (x0, top, x1, bottom)
                    # overlap check with our search window
                    if (tb[0] < search_box[2] and tb[2] > search_box[0] and
                            tb[1] < search_box[3] and tb[3] > search_box[1]):
                        matched_native = nt
                        break

                md_text = ""
                crop_box = (crop_x0, max(0, cy1 - 2), crop_x1, min(page_height, cy1 + 350))

                if matched_native is not None:
                    # ── PATH A: structured cell extraction ───────────────────
                    try:
                        rows = matched_native.extract()   # List[List[str|None]]
                        md_body = TableDetector._cells_to_markdown(rows)
                        if md_body:
                            md_text = md_body
                            crop_box = (matched_native.bbox[0], matched_native.bbox[1],
                                        matched_native.bbox[2], matched_native.bbox[3])
                    except Exception as e:
                        logger.debug(f"Native table extraction failed, falling back: {e}")

                # PATH B: word-coordinate clustering (handles borderless tables)
                if not md_text:
                    try:
                        cropped_b = plumber_page.within_bbox(crop_box)
                        words = cropped_b.extract_words(
                            x_tolerance=4, y_tolerance=4,
                            keep_blank_chars=False, use_text_flow=False,
                        )
                        if words:
                            md_body = TableDetector._words_to_markdown(words)
                            if md_body:
                                md_text = md_body
                    except Exception as eb:
                        logger.debug(f"Word-cluster extraction failed: {eb}")

                # PATH C: layout-text last resort
                if not md_text:
                    try:
                        cropped_c = plumber_page.within_bbox(crop_box)
                        raw_text = cropped_c.extract_text(layout=True)
                        if raw_text and len(raw_text.strip()) >= 15:
                            md_text = TableDetector._layout_text_to_markdown(raw_text)
                    except Exception as ec:
                        logger.debug(f"Layout fallback failed: {ec}")

                if not md_text:
                    continue

                # Build plain-text version for embedding/search
                plain = re.sub(r'[|`\-]+', ' ', md_text)
                plain = re.sub(r'\s+', ' ', plain).strip()

                tables.append({
                    "markdown": md_text,
                    "bbox": crop_box,
                    "caption": cap["text"],
                    "text": plain,
                })
                
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        return tables


class SectionDetector:
    SECTION_PATTERNS = [r'^\d+\.?\s+[A-Z]', r'^[IVX]+\.?\s+[A-Z]', r'^(Abstract|Introduction|Methods|Results|Discussion|Conclusion|References)']
    @classmethod
    def detect_section(cls, text: str) -> Optional[str]:
        text = text.strip()
        if len(text) > 100: return None
        for p in cls.SECTION_PATTERNS:
            if re.match(p, text, re.IGNORECASE): return text
        return None

@dataclass
class ExtractedEquation:
    text: str; latex: str; page_num: int; bbox: Tuple[float, float, float, float]; global_number: int; section: str = ""
@dataclass
class ExtractedTable:
    text: str; markdown: str; page_num: int; bbox: Tuple[float, float, float, float]; global_number: int; caption: str = ""; section: str = ""
@dataclass
class ExtractedFigure:
    caption: str; page_num: int; bbox: Tuple[float, float, float, float]; global_number: int; image_path: Optional[str] = None; section: str = ""

class PDFProcessorV2:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eq_det = StrictEquationDetector()
        self.tbl_det = TableDetector()
        self.sec_det = SectionDetector()
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        self.eq_counter, self.tbl_counter, self.fig_counter, self.current_section = 0, 0, 0, ""
        doc = fitz.open(pdf_path)
        plumber_doc = pdfplumber.open(pdf_path)
        equations, tables, figures, page_texts = [], [], [], []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            plumber_page = plumber_doc.pages[page_num]
            
            page_tables = self.tbl_det.extract_table_markdown(plumber_page, page)
            for pt in page_tables:
                self.tbl_counter += 1
                tbl = ExtractedTable(str(pt["text"]), pt["markdown"], page_num, pt.get("bbox", (0,0,0,0)), self.tbl_counter, pt.get("caption", ""), self.current_section)
                tables.append(tbl)
            
            blocks = page.get_text("blocks")
            page_text_parts = []
            
            for b in blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                text = text.strip()
                bbox = (x0, y0, x1, y1)
                
                if not text and block_type == 0: continue
                
                if block_type == 1:
                    self.fig_counter += 1
                    fig = ExtractedFigure(f"Image from page {page_num + 1}", page_num, bbox, self.fig_counter, self.current_section)
                    figures.append(fig)
                    page_text_parts.append(f"[Figure {self.fig_counter}]")
                    continue
                    
                if block_type == 0:
                    if re.match(r'^\s*(Figure|Fig\.)\s*\d+', text, re.IGNORECASE):
                        self.fig_counter += 1
                        caption = text.replace('\n', ' ')
                        fig = ExtractedFigure(caption, page_num, bbox, self.fig_counter, self.current_section)
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
                        latex = self.eq_det.extract_latex_from_text(text)
                        eq = ExtractedEquation(text.replace('\n', ' '), latex, page_num, bbox, self.eq_counter, self.current_section)
                        equations.append(eq)
                        page_text_parts.append(f"[Equation {self.eq_counter}]")
                        continue
                    
                    page_text_parts.append(text)
            
            page_texts.append("\n\n".join(page_text_parts))
        
        num_pages = len(doc)
        doc.close()
        plumber_doc.close()
        
        return {'equations': equations, 'tables': tables, 'figures': figures, 'page_texts': page_texts, 'num_pages': num_pages}

class EnhancedPDFProcessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.processor = PDFProcessorV2(config or {})
        
    def process_pdf(self, pdf_path: str) -> ProcessedDocument:
        raw = self.processor.process_pdf(pdf_path)
        
        equations = [ProcessedEquation(f"eq_{uuid.uuid4().hex[:8]}", e.global_number, e.text, e.latex, e.page_num, e.bbox, e.section) for e in raw.get('equations', [])]
        tables = [ProcessedTable(f"tb_{uuid.uuid4().hex[:8]}", t.global_number, t.caption, t.page_num, t.markdown, t.section, bbox=t.bbox) for t in raw.get('tables', [])]
        figures = [ProcessedFigure(f"fig_{uuid.uuid4().hex[:8]}", f.global_number, f.caption, f.page_num, f.image_path, f.bbox, f.section) for f in raw.get('figures', [])]
            
        return ProcessedDocument(
            doc_id=f"doc_{uuid.uuid4().hex[:8]}", filename=os.path.basename(pdf_path), num_pages=raw.get('num_pages', 0),
            page_texts=raw.get('page_texts', []), enriched_page_texts=raw.get('page_texts', []), sections=[], 
            equations=equations, tables=tables, figures=figures, title=os.path.basename(pdf_path)
        )
import base64
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import fitz
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from multimodel_rag.utils.equation_latex import to_latex

logger = logging.getLogger(__name__)


@dataclass
class EquationOCRResult:
    raw_text: str
    latex: str
    confidence: float = 0.75


class EquationOCRService:
    def __init__(
        self,
        groq_api_key: str,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ):
        self.llm = ChatGroq(model=model, groq_api_key=groq_api_key)

    def extract_from_image_bytes(self, image_bytes: bytes) -> Optional[EquationOCRResult]:
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            message = [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Extract only the mathematical equation(s) from this image. "
                                "Return plain text only, no explanation, no markdown, no extra words. "
                                "Preserve symbols like =, +, -, /, ^, _, Greek letters, integrals, sums."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ]
                )
            ]

            response = self.llm.invoke(message)
            raw = (response.content or "").strip()
            if not raw:
                return None

            return EquationOCRResult(
                raw_text=raw,
                latex=to_latex(raw),
                confidence=0.75,
            )

        except Exception as e:
            logger.warning("Equation OCR failed: %s", e)
            return None

    def extract_from_bbox(
        self,
        pdf_path: str,
        page_num: int,
        bbox: Tuple[float, float, float, float],
        pad: int = 10,
        zoom: float = 2.5,
    ) -> Optional[EquationOCRResult]:
        doc = None
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]

            x0, y0, x1, y1 = bbox
            rect = fitz.Rect(
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(page.rect.width, x1 + pad),
                min(page.rect.height, y1 + pad),
            )

            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
            image_bytes = pix.tobytes("png")
            return self.extract_from_image_bytes(image_bytes)

        except Exception as e:
            logger.warning("Failed to OCR bbox on page %s: %s", page_num, e)
            return None
        finally:
            if doc is not None:
                doc.close()
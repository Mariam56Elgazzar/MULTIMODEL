"""
response_formatter_v2.py - Production Response Formatter
========================================================
✅ Clean LaTeX rendering ($$...$$ blocks)
✅ Anti-repetition engine
✅ Prompt leakage prevention
✅ Display mode switching
✅ Structured metadata injection
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  LATEX RENDERER
# ═══════════════════════════════════════════════════════════════════════════

class LaTeXRenderer:
    """Render equations in proper LaTeX format"""
    
    @staticmethod
    def render_equation(latex: str, description: str = "") -> str:
        """
        Render equation in LaTeX block format.
        
        Args:
            latex: LaTeX equation string
            description: Brief description (optional)
            
        Returns:
            Formatted equation block
        """
        if not latex:
            return "[Equation not available]"
        
        # Clean LaTeX
        latex = latex.strip()
        
        # Remove existing delimiters
        latex = re.sub(r'^\$+|\$+$', '', latex)
        latex = re.sub(r'^\\begin\{equation\}|\\end\{equation\}$', '', latex)
        
        # Build output
        output = f"$$\n{latex}\n$$"
        
        if description:
            output += f"\n\n*{description}*"
        
        return output
    
    @staticmethod
    def extract_latex_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
        """Extract LaTeX from chunk metadata"""
        
        # Try common keys
        for key in ['latex', 'latex_equation', 'equation', 'formula']:
            if key in metadata and metadata[key]:
                return str(metadata[key])
        
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  ANTI-REPETITION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class AntiRepetitionEngine:
    """Remove duplicate sentences and repetitive content"""
    
    @staticmethod
    def remove_duplicate_lines(text: str) -> str:
        """Remove exact duplicate lines"""
        lines = text.split('\n')
        seen = set()
        unique_lines = []
        
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                seen.add(line_clean)
                unique_lines.append(line)
            elif not line_clean:  # Keep empty lines
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    @staticmethod
    def remove_duplicate_sentences(text: str) -> str:
        """Remove duplicate sentences"""
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            sentence_clean = sentence.strip().lower()
            
            # Ignore very short fragments
            if len(sentence_clean) < 10:
                unique_sentences.append(sentence)
                continue
            
            if sentence_clean not in seen:
                seen.add(sentence_clean)
                unique_sentences.append(sentence)
        
        return ' '.join(unique_sentences)
    
    @staticmethod
    def remove_repetitive_patterns(text: str) -> str:
        """Remove common repetitive patterns"""
        
        # Pattern 1: "Equation N is... Equation N is... Equation N is..."
        text = re.sub(
            r'((?:Equation|Table|Figure)\s+\d+\s+(?:is|are|shows|demonstrates)\s+[^.]+\.)\s*\1+',
            r'\1',
            text,
            flags=re.IGNORECASE
        )
        
        # Pattern 2: "The document contains... The document contains..."
        text = re.sub(
            r'(The document (?:contains|has|includes)[^.]+\.)\s*\1+',
            r'\1',
            text,
            flags=re.IGNORECASE
        )
        
        return text
    
    @classmethod
    def clean(cls, text: str) -> str:
        """Apply all anti-repetition techniques"""
        text = cls.remove_duplicate_lines(text)
        text = cls.remove_duplicate_sentences(text)
        text = cls.remove_repetitive_patterns(text)
        return text


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPT LEAKAGE CLEANER
# ═══════════════════════════════════════════════════════════════════════════

class PromptLeakageCleaner:
    """Remove internal prompt artifacts from responses"""
    
    LEAKAGE_PATTERNS = [
        r'Question:\s*',
        r'Answer:\s*',
        r'Context:\s*',
        r'Instructions?:\s*',
        r'System:\s*',
        r'Assistant:\s*',
        r'User:\s*',
        r'\[INST\].*?\[/INST\]',
        r'<s>.*?</s>',
        r'```instructions?.*?```',
    ]
    
    @classmethod
    def clean(cls, text: str) -> str:
        """Remove prompt leakage artifacts"""
        
        for pattern in cls.LEAKAGE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text


# ═══════════════════════════════════════════════════════════════════════════
#  METADATA EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class MetadataExtractor:
    """Extract and format metadata from chunks"""
    
    @staticmethod
    def extract_citations(chunks: List[Any]) -> List[str]:
        """Extract page citations from chunks"""
        citations = set()
        
        for chunk in chunks:
            page_num = chunk.metadata.get('page_number') or chunk.page_num
            if page_num is not None:
                citations.add(f"Page {page_num}")
        
        return sorted(list(citations))
    
    @staticmethod
    def extract_element_info(chunk: Any) -> Dict[str, Any]:
        """Extract element information from chunk"""
        
        info = {
            'type': chunk.chunk_type,
            'page': chunk.metadata.get('page_number') or chunk.page_num,
            'global_number': chunk.metadata.get('global_number'),
            'section': chunk.metadata.get('section', ''),
        }
        
        # Type-specific info
        if chunk.chunk_type == 'equation':
            info['latex'] = LaTeXRenderer.extract_latex_from_metadata(chunk.metadata)
            info['variables'] = chunk.metadata.get('variables', [])
        
        elif chunk.chunk_type == 'table':
            info['caption'] = chunk.metadata.get('caption', '')
            info['markdown'] = chunk.metadata.get('markdown', '')
        
        elif chunk.chunk_type == 'figure':
            info['caption'] = chunk.metadata.get('caption', '')
            info['image_path'] = chunk.image_path
        
        return info


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY MODES
# ═══════════════════════════════════════════════════════════════════════════

class DisplayMode:
    """Different display modes for responses"""
    
    @staticmethod
    def specific_element(chunk: Any) -> str:
        """Display mode for specific element request"""
        
        element_type = chunk.chunk_type
        info = MetadataExtractor.extract_element_info(chunk)
        
        output = []
        
        # Header
        if info['global_number']:
            output.append(f"### {element_type.title()} {info['global_number']}")
        else:
            output.append(f"### {element_type.title()}")
        
        # Content
        if element_type == 'equation':
            latex = info.get('latex')
            if latex:
                output.append(LaTeXRenderer.render_equation(latex))
            else:
                output.append("[LaTeX not available]")
            
            # Brief description from text
            desc = chunk.text[:200].strip()
            if desc:
                output.append(f"\n*{desc}*")
        
        elif element_type == 'table':
            markdown = info.get('markdown')
            if markdown:
                output.append(markdown)
            else:
                output.append(chunk.text)
            
            if info.get('caption'):
                output.append(f"\n*{info['caption']}*")
        
        elif element_type == 'figure':
            if info.get('caption'):
                output.append(info['caption'])
            else:
                output.append(chunk.text)
        
        else:  # text
            output.append(chunk.text)
        
        # Citation
        if info['page']:
            output.append(f"\n📄 *Source: Page {info['page']}*")
        
        return '\n\n'.join(output)
    
    @staticmethod
    def list_all(chunks: List[Any], element_type: str) -> str:
        """Display mode for listing all elements"""
        
        output = [f"### All {element_type.title()}s\n"]
        
        for chunk in chunks:
            info = MetadataExtractor.extract_element_info(chunk)
            
            if info['global_number']:
                # Brief description (first 100 chars)
                desc = chunk.text[:100].strip()
                if len(chunk.text) > 100:
                    desc += "..."
                
                output.append(f"**{element_type.title()} {info['global_number']}**: {desc}")
        
        return '\n\n'.join(output)
    
    @staticmethod
    def explanation(response_text: str, chunks: List[Any]) -> str:
        """Display mode for detailed explanation"""
        
        output = [response_text]
        
        # Add citations
        citations = MetadataExtractor.extract_citations(chunks)
        if citations:
            output.append(f"\n\n📚 **Sources**: {', '.join(citations)}")
        
        return '\n\n'.join(output)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN RESPONSE FORMATTER
# ═══════════════════════════════════════════════════════════════════════════

class ResponseFormatterV2:
    """
    Production-level response formatter with:
    - Clean LaTeX rendering
    - Anti-repetition
    - Prompt leakage prevention
    - Display modes
    - Metadata injection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.latex_renderer = LaTeXRenderer()
        self.anti_rep = AntiRepetitionEngine()
        self.leakage_cleaner = PromptLeakageCleaner()
        self.metadata_extractor = MetadataExtractor()
        self.display = DisplayMode()
        
        logger.info("✅ ResponseFormatterV2 initialized")
    
    def format_response(
        self,
        response_text: str,
        chunks: List[Any],
        query_intent: Any,
        include_metadata: bool = True
    ) -> str:
        """
        Format LLM response with proper structure.
        
        Args:
            response_text: Raw LLM output
            chunks: Retrieved chunks
            query_intent: QueryIntent object
            include_metadata: Whether to add citations
            
        Returns:
            Formatted response string
        """
        
        # 1. Clean prompt leakage
        response_text = self.leakage_cleaner.clean(response_text)
        
        # 2. Remove repetition
        response_text = self.anti_rep.clean(response_text)
        
        # 3. Apply display mode
        if query_intent.intent == "SPECIFIC_ELEMENT" and len(chunks) == 1:
            # Use structured display for single element
            formatted = self.display.specific_element(chunks[0])
        
        elif query_intent.intent == "LIST_ALL":
            # Use list display
            formatted = self.display.list_all(chunks, query_intent.element_type or "element")
        
        else:
            # Standard explanation mode
            formatted = response_text
            
            # Add metadata if requested
            if include_metadata and chunks:
                citations = self.metadata_extractor.extract_citations(chunks)
                if citations:
                    formatted += f"\n\n📚 **Sources**: {', '.join(citations)}"
        
        # 4. Final cleanup
        formatted = formatted.strip()
        
        # Remove excessive blank lines
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        
        return formatted
    
    def format_equation_response(
        self,
        equation_chunk: Any,
        explanation: str = ""
    ) -> str:
        """
        Format equation-specific response.
        
        Args:
            equation_chunk: Equation chunk
            explanation: Optional explanation text
            
        Returns:
            Formatted equation display
        """
        
        info = self.metadata_extractor.extract_element_info(equation_chunk)
        latex = info.get('latex')
        
        output = []
        
        # Render equation
        if latex:
            output.append(self.latex_renderer.render_equation(latex, explanation))
        else:
            output.append("[LaTeX not available]")
            if explanation:
                output.append(explanation)
        
        # Add metadata
        if info['global_number']:
            output.append(f"\n**Equation {info['global_number']}**")
        
        if info['page']:
            output.append(f"📄 *Page {info['page']}*")
        
        if info['section']:
            output.append(f"📂 *Section: {info['section']}*")
        
        return '\n\n'.join(output)
    
    def format_table_response(
        self,
        table_chunk: Any,
        caption: str = ""
    ) -> str:
        """
        Format table-specific response.
        
        Args:
            table_chunk: Table chunk
            caption: Optional caption text
            
        Returns:
            Formatted table display
        """
        
        info = self.metadata_extractor.extract_element_info(table_chunk)
        
        output = []
        
        # Header
        if info['global_number']:
            output.append(f"### Table {info['global_number']}")
        
        # Markdown table
        markdown = info.get('markdown')
        if markdown:
            output.append(markdown)
        else:
            output.append(table_chunk.text)
        
        # Caption
        if caption or info.get('caption'):
            output.append(f"\n*{caption or info['caption']}*")
        
        # Metadata
        if info['page']:
            output.append(f"\n📄 *Page {info['page']}*")
        
        return '\n\n'.join(output)


# ═══════════════════════════════════════════════════════════════════════════
#  USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test anti-repetition
    test_text = """
    Equation 3 is used for probability. Equation 3 is used for probability.
    The document contains 5 equations. The document contains 5 equations.
    
    This is a unique sentence.
    This is another unique sentence.
    This is a unique sentence.
    """
    
    engine = AntiRepetitionEngine()
    cleaned = engine.clean(test_text)
    
    print("Original:")
    print(test_text)
    print("\nCleaned:")
    print(cleaned)
    
    print("\n✅ response_formatter_v2.py ready")
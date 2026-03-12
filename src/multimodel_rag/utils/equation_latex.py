"""
equation_latex.py - LaTeX Normalizer for Equations (FIXED V17)
===============================================================
✅ Fixed character-by-character spacing detection
✅ Stronger prose rejection
✅ Better word boundary handling
✅ Operator requirement enforcement
"""

import re
from typing import Optional

# ─── Greek Letters Mapping ───────────────────────────────────────────────────
GREEK_MAP = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
    "Ψ": r"\Psi", "Ω": r"\Omega",
}

# ─── Math Symbols Mapping ────────────────────────────────────────────────────
MATH_SYMBOLS = {
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "√": r"\sqrt",
    "±": r"\pm", "∂": r"\partial", "∇": r"\nabla", "∞": r"\infty",
    "≈": r"\approx", "≠": r"\neq", "≤": r"\leq", "≥": r"\geq",
    "∝": r"\propto", "∈": r"\in", "∉": r"\notin",
    "⊂": r"\subset", "⊆": r"\subseteq", "∪": r"\cup", "∩": r"\cap",
    "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⊕": r"\oplus", "⊗": r"\otimes", "⊤": r"\top",
    "‖": r"\|", "⟨": r"\langle", "⟩": r"\rangle",
    "·": r"\cdot", "×": r"\times",
}

# ─── Prose Detection ─────────────────────────────────────────────────────────
COMMON_PROSE_WORDS = {
    'the', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those',
    'we', 'our', 'their', 'your', 'in', 'of', 'for', 'with', 'and', 'to',
    'a', 'an', 'as', 'by', 'from', 'on', 'at', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will',
    'would', 'should', 'may', 'might', 'must', 'shall',
}


def _count_prose_words(text: str) -> int:
    """Count common prose words in text"""
    words = re.findall(r'\b\w+\b', text.lower())
    return sum(1 for w in words if w in COMMON_PROSE_WORDS)


def _has_equation_structure(text: str) -> bool:
    """
    Check if text has actual equation structure (not just symbols)
    
    Requirements:
    - Has operators (=, ∝, →, etc.) OR
    - Has functions (exp, log, sum) with parentheses OR
    - Has fractions/subscripts/superscripts OR
    - Has probability notation p(x|y) with conditional
    """
    # Check for operators
    has_operator = any(op in text for op in ['=', '∝', '→', '←', '≡', '≈', '≤', '≥', '∫', '∑', '∏'])
    
    # Check for function calls
    has_function_call = bool(re.search(
        r'\b(exp|log|sin|cos|tan|max|min|argmax|argmin|softmax|sigmoid)\s*\(',
        text, re.IGNORECASE
    ))
    
    # Check for structural elements
    has_structure = bool(re.search(r'[_^{}\\]', text))
    
    # Check for probability notation with conditionals: p(z|x), q(y|z,x)
    has_probability = bool(re.search(r'\b[pqPQ][_\\]?[\w\\]+\([^)]*\|[^)]*\)', text))
    
    return has_operator or has_function_call or has_structure or has_probability


def normalize_math_text(raw: str) -> str:
    """
    Normalize raw text by cleaning up spacing and converting patterns.
    
    CRITICAL FIXES:
    1. Detect and merge character-spaced text ("M I P S" → "MIPS")
    2. Remove spaces around operators
    3. Fix subscript spacing
    
    Args:
        raw: Raw equation text
        
    Returns:
        Normalized text
    """
    if not raw:
        return ""
    
    s = raw.strip()
    
    # ✅ FIX 1: Detect character-by-character spacing
    words = s.split()
    
    # Check if excessive single-character words (PDF extraction bug)
    if len(words) > 5:
        single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
        
        if single_char_ratio > 0.5:
            # Definitely character-spaced, merge intelligently
            merged = []
            i = 0
            while i < len(words):
                word = words[i]
                
                # Collect consecutive alpha single chars
                if word.isalpha() and len(word) == 1:
                    chars = [word]
                    j = i + 1
                    while j < len(words) and words[j].isalpha() and len(words[j]) == 1:
                        chars.append(words[j])
                        j += 1
                    
                    # If collected 2+ chars, merge them
                    if len(chars) > 1:
                        merged_word = ''.join(chars)
                        # Only merge if it looks like a word (not random letters)
                        if len(merged_word) >= 2:
                            merged.append(merged_word)
                        else:
                            merged.extend(chars)
                        i = j
                    else:
                        merged.append(word)
                        i += 1
                else:
                    merged.append(word)
                    i += 1
            
            s = ' '.join(merged)
    
    # ✅ FIX 2: Remove spaces around operators
    s = re.sub(r'\s+([=+\-*/|∝→←≈≡≤≥])\s+', r'\1', s)
    
    # ✅ FIX 3: Remove spaces in parentheses
    s = re.sub(r'\s+\(', '(', s)
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    
    # ✅ FIX 4: Fix subscript spacing (p η → pη, p θ → pθ)
    s = re.sub(r'\b([a-zA-Z])\s+([αβγδεηθλμπστφψω])\b', r'\1\2', s)
    
    # Clean excessive whitespace
    s = re.sub(r'\s{2,}', ' ', s).strip()
    
    # Replace Greek letters
    for greek, latex in GREEK_MAP.items():
        s = s.replace(greek, latex)
    
    # Replace math symbols
    for symbol, latex in MATH_SYMBOLS.items():
        s = s.replace(symbol, latex)
    
    # Functions
    for func in ['exp', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'max', 'min']:
        s = re.sub(rf'\b{func}\s*\(', rf'\\{func}(', s, flags=re.IGNORECASE)
        s = re.sub(rf'\b{func}\b', rf'\\{func}', s, flags=re.IGNORECASE)
    
    # Special operators
    s = re.sub(r'\bargmax\b', r'\\operatorname{argmax}', s, flags=re.IGNORECASE)
    s = re.sub(r'\bargmin\b', r'\\operatorname{argmin}', s, flags=re.IGNORECASE)
    s = re.sub(r'\bsoftmax\b', r'\\operatorname{softmax}', s, flags=re.IGNORECASE)
    s = re.sub(r'\bsigmoid\b', r'\\operatorname{sigmoid}', s, flags=re.IGNORECASE)
    
    # Subscripts (common patterns)
    s = re.sub(r'\bp_η\b', r'p_\\eta', s)
    s = re.sub(r'\bp_θ\b', r'p_\\theta', s)
    s = re.sub(r'\bpη\b', r'p_\\eta', s)
    s = re.sub(r'\bpθ\b', r'p_\\theta', s)
    
    # Conditional probability |
    s = re.sub(r'\|(?=\s*[a-zA-Z\\_])', r' \\mid ', s)
    s = re.sub(r'\bsim\b', r'\\sim', s)
    
    # Final cleanup
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s


def to_latex(raw: str) -> str:
    """
    Convert raw equation text to clean LaTeX.
    
    Args:
        raw: Raw equation text
        
    Returns:
        LaTeX string ready for st.latex()
    """
    if not raw:
        return ""
    
    # First normalize
    s = normalize_math_text(raw)
    
    # Additional cleanup for display
    s = re.sub(r'\s*=\s*', ' = ', s)
    s = re.sub(r'\s*\\propto\s*', r' \\propto ', s)
    
    # Remove equation numbers like (1), (12), (3a) at end
    s = re.sub(r'\(\s*\d{1,3}[a-z]?\s*\)\s*$', '', s)
    
    # Clean up spaces before/after delimiters
    s = re.sub(r'\s*\{\s*', '{', s)
    s = re.sub(r'\s*}\s*', '}', s)
    s = re.sub(r'\s*\(\s*', '(', s)
    s = re.sub(r'\s*\)\s*', ')', s)
    
    return sanitize_latex(s.strip())


def sanitize_latex(latex_str: str) -> str:
    """
    D) Equation Canonicalization + Safe LaTeX:
    - Automatically cleans broken unicode 
    - Fixes unmatched brackets
    """
    if not latex_str:
        return ""
    
    # 1. Remove broken characters and smart quotes
    cleaned = latex_str.replace("", "").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    
    # 2. Balance brackets and braces
    open_braces = cleaned.count('{')
    close_braces = cleaned.count('}')
    
    if open_braces > close_braces:
        cleaned += "}" * (open_braces - close_braces)
    elif close_braces > open_braces:
        # Just strip trailing extra closing brackets
        for _ in range(close_braces - open_braces):
            if cleaned.endswith("}"):
                cleaned = cleaned[:-1]
                
    # 3. Drop unrenderable plain text math if still broken
    # E.g. raw text like "pη(z|x) ∝ exp d(z)⊤q(x)" missing proper braces
    if "\\" not in cleaned and len(cleaned) > 20 and not any(op in cleaned for op in ["=", "∝", "\\"]):
        # Fallback: simple math formatting bypassing latex constraints
        return f"\\text{{{cleaned}}}"
        
    return cleaned


def looks_like_math(text: str, strict: bool = True) -> bool:
    """
    Check if text looks like a mathematical equation (STRICTER VERSION).
    
    Args:
        text: Text to check
        strict: If True, require equation structure (not just symbols)
        
    Returns:
        True if it looks like math
    """
    if not text or len(text.strip()) < 3:
        return False
    
    text = text.strip()
    
    # ❌ REJECT: Starts with caption keywords
    if re.match(r'^\s*(Figure|Fig|Table|Appendix|Algorithm|Listing)\s+\d', text, re.IGNORECASE):
        return False
    
    # ❌ REJECT: Too much prose (>40% common words)
    words = re.findall(r'\b\w+\b', text)
    if len(words) > 5:
        prose_count = _count_prose_words(text)
        prose_ratio = prose_count / len(words)
        if prose_ratio > 0.4:
            return False
    
    # ❌ REJECT: Very long (likely paragraph)
    if len(text) > 500 or len(words) > 50:
        return False
    
    # Check for math indicators
    math_signals = [
        r'\\exp', r'\\log', r'\\sum', r'\\prod', r'\\int',
        r'\\alpha', r'\\beta', r'\\gamma', r'\\theta', r'\\eta',
        r'\\propto', r'\\approx', r'\\leq', r'\\geq',
        r'[=∑∏∫∝∈]',
        r'\bexp\b', r'\blog\b', r'\bsoftmax\b', r'\bargmax\b',
        r'[αβγδεηθλμνπρστφψω]',  # Greek letters
    ]
    
    has_math_symbol = any(re.search(pattern, text) for pattern in math_signals)
    
    if not has_math_symbol:
        return False
    
    # ✅ REQUIRE: Equation structure (if strict mode)
    if strict and not _has_equation_structure(text):
        return False
    
    return True


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Valid equations
        ("p η ( z | x ) p θ ( y | x , z )", True),
        ("∑ i = 1 n x i", True),
        ("exp ( - x )", True),
        ("p_θ(y|x,z) = ∫ p(z|x) dz", True),
        
        # Invalid (prose)
        ("Generator pθ", False),
        ("MIPS pθ", False),
        ("Answer Generation Retriever pη", False),
        ("The model uses a generator pθ to produce outputs", False),
        
        # Edge cases
        ("M I P S p θ", False),  # Just acronym + symbol
        ("top - k selection", False),  # No real equation
    ]
    
    print("Testing equation validation:")
    print("=" * 70)
    for text, expected in test_cases:
        result = looks_like_math(text, strict=True)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{text[:50]}...' → {result} (expected {expected})")
    
    print("\n" + "=" * 70)
    print("Testing LaTeX conversion:")
    print("=" * 70)
    valid_eqs = [text for text, exp in test_cases if exp]
    for eq in valid_eqs:
        latex = to_latex(eq)
        print(f"\nOriginal: {eq}")
        print(f"LaTeX:    {latex}")
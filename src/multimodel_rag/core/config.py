"""
Enhanced Configuration V3 - إعدادات محسّنة شاملة
✅ دعم كامل V3
✅ إعدادات آمنة
✅ معايير جودة عالية
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

# ==================== API KEYS ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ==================== GROQ MODELS ====================
GROQ_MODELS = {
    'text': 'llama-3.3-70b-versatile',
    'vision': 'llama-3.2-11b-vision-preview',
    'reasoning': 'llama-3.3-70b-versatile',
    'fast': 'llama-3.1-8b-instant',
    'chat': 'llama-3.3-70b-versatile'
}

# ==================== PDF PROCESSOR CONFIG ====================
PDF_PROCESSOR_CONFIG = {
    'extract_equations': True,
    'extract_tables': True,
    'extract_images': True,
    'equation_confidence_threshold': 0.45,
    'table_confidence_threshold': 0.65,
    'preserve_formatting': True,
}

# ==================== VECTOR STORE CONFIG ====================
VECTOR_STORE_CONFIG = {
    'backend': 'faiss',
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'dimension': 384,
    'cache_enabled': True,
    'cache_dir': './cache',
    'top_k': 7,
    'similarity_threshold': 0.3,
}

# ==================== RESPONSE FORMATTER CONFIG ====================
RESPONSE_FORMATTER_CONFIG = {
    'enable_hallucination_detection': True,
    'safe_mode': True,
    'max_response_length': 5000,
    'include_metadata': True,
    'include_page_references': True,
}

# ==================== CHUNKING CONFIG ====================
CHUNKING_CONFIG = {
    'chunk_size': 1500,
    'chunk_overlap': 300,
    'separators': ["\n\n", "\n", ". ", " ", ""],
    'min_chunk_size': 100,
    'smart_chunking': True,
    'preserve_equations': True,
    'preserve_tables': True
}

# ==================== EMBEDDING CONFIG ====================
EMBEDDING_CONFIG = {
    'model': 'sentence-transformers/all-MiniLM-L6-v2',
    'dimension': 384,
    'batch_size': 32,
    'cache_enabled': True,
}

# ==================== WEB SEARCH CONFIG ====================
WEB_SEARCH_CONFIG = {
    'enable_web_search': True,
    'search_providers': {
        'wikipedia': True,
        'arxiv': True,
        'general': True
    },
    'max_search_results': 5,
    'search_timeout': 10,
    'cache_results': True,
}

# ==================== CHAT CONFIG ====================
CHAT_CONFIG = {
    'max_history_length': 20,
    'history_window': 10,
    'save_history': True,
    'history_file': 'chat_history.json',
}

# ==================== RATE LIMITER CONFIG ====================
RATE_LIMITER_CONFIG = {
    'requests_per_minute': 30,
    'tokens_per_minute': 14400,
    'tokens_per_day': 600_000,
    'tokens_per_month': 15_000_000,
    'retry_attempts': 3,
    'retry_delay': 2.0,
    'safety_margin_percent': 10.0,
    'use_exponential_backoff': True
}

# ==================== LOGGING CONFIG ====================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'rag_system_v3.log',
    'save_logs': True
}

# ==================== SYSTEM PROMPTS ====================
SYSTEM_PROMPTS = {
    'equation': """You are a mathematics expert. When explaining equations:
1. Show equation in clear format
2. Explain each variable and symbol
3. Describe relationships
4. Provide examples
5. Cite source (page number)""",

    'table': """You are a data analysis expert. When analyzing tables:
1. Identify headers and structure
2. Extract key values
3. Identify patterns
4. Make observations
5. Cite table number and page""",

    'figure': """You are an expert in analyzing images. When describing figures:
1. Identify the type
2. Describe key elements
3. Explain data shown
4. Extract numerical values
5. Cite source clearly""",

    'general': """You are an intelligent AI assistant with deep document understanding.

Capabilities:
✅ Deep document analysis
✅ Cross-reference information
✅ Understand context
✅ Remember history
✅ Cite sources
✅ Format professionally

Rules:
- Remember all context
- Always cite sources
- Format equations, tables beautifully
- Connect information
- Never hallucinate about missing elements"""
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration"""
    return {
        'groq_api_key': GROQ_API_KEY,
        'groq_models': GROQ_MODELS,
        'pdf_processor': PDF_PROCESSOR_CONFIG,
        'vector_store': VECTOR_STORE_CONFIG,
        'response_formatter': RESPONSE_FORMATTER_CONFIG,
        'chunking': CHUNKING_CONFIG,
        'embedding': EMBEDDING_CONFIG,
        'web_search': WEB_SEARCH_CONFIG,
        'chat': CHAT_CONFIG,
        'rate_limiter': RATE_LIMITER_CONFIG,
        'logging': LOGGING_CONFIG,
        'prompts': SYSTEM_PROMPTS
    }

def validate_config() -> bool:
    """Validate configuration"""
    issues = []
    
    if not GROQ_API_KEY or GROQ_API_KEY == "your-api-key":
        issues.append("❌ GROQ_API_KEY not set")
    else:
        print("✅ GROQ_API_KEY configured")
    
    try:
        import faiss
        print("✅ FAISS available")
    except ImportError:
        issues.append("⚠️ FAISS not installed")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers available")
    except ImportError:
        issues.append("❌ sentence-transformers not installed")
    
    try:
        import fitz
        print("✅ PyMuPDF available")
    except ImportError:
        issues.append("⚠️ PyMuPDF not installed")
    
    if issues:
        print("\n⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  {issue}")
    
    return len([i for i in issues if i.startswith("❌")]) == 0

if __name__ == "__main__":
    print("✅ Validating configuration...")
    if validate_config():
        print("\n✅ All configurations valid!")
    else:
        print("\n❌ Fix issues above")
"""
chat_history_v2.py - Production Chat History Manager
====================================================
✅ Smart context limiting (last 3 Q/A only)
✅ Auto-reset on new PDF upload
✅ Failed answer filtering
✅ Reference tracking per turn
✅ Memory-efficient storage
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChatTurn:
    """Single conversation turn (Q + A)"""
    turn_id: int
    timestamp: str
    user_query: str
    assistant_response: str
    
    # Optional fields with defaults
    query_intent: Optional[Dict[str, Any]] = None
    response_chunks: List[str] = field(default_factory=list)  # chunk_ids used
    response_citations: List[str] = field(default_factory=list)  # page references
    
    # Quality metrics
    retrieval_score: float = 0.0
    response_valid: bool = True
    error_message: Optional[str] = None
    
    # Metadata
    document_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatTurn':
        return cls(**data)
    
    def to_message_format(self) -> List[Dict[str, str]]:
        """Convert to LLM message format"""
        return [
            {"role": "user", "content": self.user_query},
            {"role": "assistant", "content": self.assistant_response}
        ]


# ═══════════════════════════════════════════════════════════════════════════
#  CHAT HISTORY MANAGER V2
# ═══════════════════════════════════════════════════════════════════════════

class ChatHistoryManagerV2:
    """
    Production chat history with smart context management.
    
    Features:
    - Limit context to last N turns
    - Auto-reset on document change
    - Filter failed responses
    - Track references per turn
    - Export capabilities
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        max_context_turns: int = 3,  # Only last 3 Q/A pairs
        auto_save: bool = True
    ):
        self.config = config
        self.max_context_turns = max_context_turns
        self.auto_save = auto_save
        
        self.turns: List[ChatTurn] = []
        self.current_document_id: Optional[str] = None
        self.session_id = self._generate_session_id()
        
        self.history_file = config.get('history_file', 'chat_history_v2.json')
        
        # Load existing history if available
        if auto_save and os.path.exists(self.history_file):
            self._load_history()
        
        logger.info(f"✅ ChatHistoryManagerV2 initialized (max_context={max_context_turns})")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ─────────────────────────────────────────────────────────────────────────
    #  TURN MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_turn(
        self,
        user_query: str,
        assistant_response: str,
        query_intent: Optional[Dict[str, Any]] = None,
        response_chunks: List[str] = None,
        response_citations: List[str] = None,
        retrieval_score: float = 0.0,
        response_valid: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Add a conversation turn.
        
        Args:
            user_query: User's question
            assistant_response: Assistant's answer
            query_intent: Parsed query intent (optional)
            response_chunks: List of chunk IDs used
            response_citations: List of page references
            retrieval_score: Retrieval quality score
            response_valid: Whether response was successful
            error_message: Error message if failed
        """
        
        # Create turn
        turn = ChatTurn(
            turn_id=len(self.turns) + 1,
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            assistant_response=assistant_response,
            query_intent=query_intent,
            response_chunks=response_chunks or [],
            response_citations=response_citations or [],
            retrieval_score=retrieval_score,
            response_valid=response_valid,
            error_message=error_message,
            document_id=self.current_document_id
        )
        
        self.turns.append(turn)
        
        logger.info(f"Added turn #{turn.turn_id} (valid={response_valid}, score={retrieval_score:.3f})")
        
        # Auto-save
        if self.auto_save:
            self._save_history()
    
    def get_context_for_llm(
        self,
        include_failed: bool = False,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation context for LLM.
        
        Args:
            include_failed: Whether to include failed responses
            max_turns: Maximum turns to include (default: self.max_context_turns)
            
        Returns:
            List of message dicts [{"role": "user"|"assistant", "content": "..."}]
        """
        
        if max_turns is None:
            max_turns = self.max_context_turns
        
        # Filter turns
        valid_turns = self.turns
        if not include_failed:
            valid_turns = [t for t in self.turns if t.response_valid]
        
        # Get recent turns
        recent_turns = valid_turns[-max_turns:] if valid_turns else []
        
        # Convert to message format
        messages = []
        for turn in recent_turns:
            messages.extend(turn.to_message_format())
        
        logger.debug(f"Context: {len(messages)} messages from {len(recent_turns)} turns")
        
        return messages
    
    def get_recent_turns(self, count: int = 5) -> List[ChatTurn]:
        """Get N most recent turns"""
        return self.turns[-count:] if self.turns else []
    
    def get_all_turns(self) -> List[ChatTurn]:
        """Get all turns"""
        return self.turns.copy()
    
    # ─────────────────────────────────────────────────────────────────────────
    #  DOCUMENT MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_new_document(self, document_id: str):
        """
        Reset history when new document is loaded.
        
        Args:
            document_id: Unique identifier for the document
        """
        
        logger.info(f"📄 New document loaded: {document_id}")
        
        # Save current history before reset
        if self.turns and self.auto_save:
            self._save_history()
        
        # Reset
        self.turns = []
        self.current_document_id = document_id
        self.session_id = self._generate_session_id()
        
        logger.info("✅ Chat history reset for new document")
    
    def clear_history(self):
        """Clear all history"""
        self.turns = []
        logger.info("Chat history cleared")
    
    # ─────────────────────────────────────────────────────────────────────────
    #  PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────
    
    def _save_history(self):
        """Save history to file"""
        try:
            data = {
                'session_id': self.session_id,
                'document_id': self.current_document_id,
                'timestamp': datetime.now().isoformat(),
                'turns': [turn.to_dict() for turn in self.turns]
            }
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved {len(self.turns)} turns to {self.history_file}")
        
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def _load_history(self):
        """Load history from file"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.session_id = data.get('session_id', self._generate_session_id())
            self.current_document_id = data.get('document_id')
            
            self.turns = [
                ChatTurn.from_dict(turn_data)
                for turn_data in data.get('turns', [])
            ]
            
            logger.info(f"Loaded {len(self.turns)} turns from {self.history_file}")
        
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self.turns = []
    
    # ─────────────────────────────────────────────────────────────────────────
    #  STATISTICS & EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        
        if not self.turns:
            return {
                'total_turns': 0,
                'valid_turns': 0,
                'failed_turns': 0,
                'avg_retrieval_score': 0.0
            }
        
        valid_turns = [t for t in self.turns if t.response_valid]
        failed_turns = [t for t in self.turns if not t.response_valid]
        
        avg_score = sum(t.retrieval_score for t in self.turns) / len(self.turns)
        
        return {
            'session_id': self.session_id,
            'document_id': self.current_document_id,
            'total_turns': len(self.turns),
            'valid_turns': len(valid_turns),
            'failed_turns': len(failed_turns),
            'avg_retrieval_score': avg_score,
            'latest_turn': self.turns[-1].timestamp if self.turns else None
        }
    
    def export_markdown(self, output_path: str):
        """Export conversation to markdown"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Chat History - {self.session_id}\n\n")
            
            if self.current_document_id:
                f.write(f"**Document**: {self.current_document_id}\n\n")
            
            f.write("---\n\n")
            
            for turn in self.turns:
                # Turn header
                f.write(f"## Turn {turn.turn_id}\n")
                f.write(f"*{turn.timestamp}*\n\n")
                
                # User query
                f.write("### 👤 User\n")
                f.write(f"{turn.user_query}\n\n")
                
                # Assistant response
                f.write("### 🤖 Assistant\n")
                f.write(f"{turn.assistant_response}\n\n")
                
                # Metadata
                if turn.response_citations:
                    f.write(f"📚 **Citations**: {', '.join(turn.response_citations)}\n")
                
                f.write(f"📊 **Score**: {turn.retrieval_score:.3f}\n")
                
                if turn.error_message:
                    f.write(f"⚠️ **Error**: {turn.error_message}\n")
                
                f.write("\n---\n\n")
        
        logger.info(f"Exported history to {output_path}")
    
    def export_json(self, output_path: str):
        """Export conversation to JSON"""
        
        data = {
            'session_id': self.session_id,
            'document_id': self.current_document_id,
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'turns': [turn.to_dict() for turn in self.turns]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported history to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create manager
    config = {'history_file': 'test_history.json'}
    manager = ChatHistoryManagerV2(config, max_context_turns=3)
    
    # Simulate conversation
    manager.on_new_document("test_doc_001")
    
    manager.add_turn(
        user_query="What is equation 3?",
        assistant_response="Equation 3 is: $$p(y|x,z) = ...$$",
        retrieval_score=0.85,
        response_valid=True
    )
    
    manager.add_turn(
        user_query="Show me table 1",
        assistant_response="Table 1 shows...",
        retrieval_score=0.75,
        response_valid=True
    )
    
    manager.add_turn(
        user_query="What is equation 999?",
        assistant_response="Equation not found",
        retrieval_score=0.0,
        response_valid=False,
        error_message="Element not found"
    )
    
    # Get context (should exclude failed turn by default)
    context = manager.get_context_for_llm()
    print(f"\nContext messages: {len(context)}")
    
    # Statistics
    stats = manager.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export
    manager.export_markdown("test_history.md")
    
    print("\n✅ chat_history_v2.py ready")
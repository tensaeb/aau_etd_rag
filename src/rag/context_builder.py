import nltk
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.logger import setup_logger

logger = setup_logger("context_builder")

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ContextBuilder:
    """Handles loading, caching, and formatting of text chunks for context."""
    
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = chunk_dir
        self._cache = {} # Cache for loaded chunk files

    def load_chunk(self, source: str, chunk_id: int) -> str:
        """Loads a specific chunk, utilizing a cache for file contents."""
        if source not in self._cache:
            file_path = self.chunk_dir / f"{source}.txt"
            if not file_path.exists():
                logger.error(f"Chunk file not found: {file_path}")
                return f"[Error: Chunk file not found: {source}]"
            
            try:
                content = file_path.read_text(encoding="utf-8")
                self._cache[source] = content.split("\n\n---\n\n")
            except Exception as e:
                logger.error(f"Error reading chunk file {file_path}: {e}")
                return f"[Error loading file: {source}]"

        chunks = self._cache[source]
        if chunk_id >= len(chunks):
            return f"[Error: Chunk ID {chunk_id} out of range for {source}]"
        
        return chunks[chunk_id]

    def truncate_text(self, text: str, max_chars: int) -> str:
        """Truncates text at sentence boundaries using NLTK."""
        if len(text) <= max_chars:
            return text
        
        sentences = nltk.sent_tokenize(text)
        current_text = ""
        for sent in sentences:
            if len(current_text) + len(sent) + 1 <= max_chars:
                current_text += (sent + " ")
            else:
                break
        
        return current_text.strip() + "..." if current_text else text[:max_chars] + "..."

    def build(self, items: List[Dict[str, Any]], max_context_chars: int) -> str:
        """Builds the final context string from retrieved items."""
        max_chunk_chars = max_context_chars // len(items) if items else max_context_chars
        
        blocks = []
        total_chars = 0
        
        for item in items:
            text = self.load_chunk(item["source"], item["chunk_id"])
            if len(text) > max_chunk_chars:
                text = self.truncate_text(text, max_chunk_chars)
            
            block = f"[{item['source']} | chunk {item['chunk_id']}]\n{text}"
            
            if total_chars + len(block) > max_context_chars and blocks:
                break
            
            blocks.append(block)
            total_chars += len(block) + 2
            
        return "\n\n".join(blocks)

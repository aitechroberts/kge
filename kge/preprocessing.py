"""
Text preprocessing and chunking utilities.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextProcessor:
    """Text preprocessing and chunking utilities."""
    
    def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 200):
        """Initialize text processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Compile regex patterns for efficiency
        self._sentence_pattern = re.compile(r'[.!?]+\s+')
        self._paragraph_pattern = re.compile(r'\n\s*\n')
        self._whitespace_pattern = re.compile(r'\s+')
        
        logger.info(f"TextProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = self._whitespace_pattern.sub(' ', text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, preserve_sentences: bool = True) -> List[TextChunk]:
        """Split text into chunks with optional sentence preservation.
        
        Args:
            text: Input text to chunk
            preserve_sentences: Whether to avoid breaking sentences
            
        Returns:
            List of TextChunk objects
        """
        if not text:
            return []
        
        # Clean the text first
        text = self.clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [TextChunk(
                text=text,
                start_pos=0,
                end_pos=len(text),
                chunk_id=0
            )]
        
        chunks = []
        start_pos = 0
        chunk_id = 0
        
        while start_pos < len(text):
            # Calculate end position
            end_pos = min(start_pos + self.chunk_size, len(text))
            
            # If we're not at the end and want to preserve sentences
            if end_pos < len(text) and preserve_sentences:
                # Look for sentence boundary within the last 200 characters
                search_start = max(end_pos - 200, start_pos)
                sentence_matches = list(self._sentence_pattern.finditer(text, search_start, end_pos))
                
                if sentence_matches:
                    # Use the last sentence boundary
                    last_match = sentence_matches[-1]
                    end_pos = last_match.end()
            
            # Extract chunk text
            chunk_text = text[start_pos:end_pos].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    chunk_id=chunk_id,
                    metadata={
                        "length": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    }
                ))
                chunk_id += 1
            
            # Move to next chunk with overlap
            if end_pos >= len(text):
                break
            
            start_pos = max(end_pos - self.chunk_overlap, start_pos + 1)
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[TextChunk]:
        """Split text into chunks based on paragraph boundaries.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of TextChunk objects
        """
        text = self.clean_text(text)
        paragraphs = self._paragraph_pattern.split(text)
        
        chunks = []
        current_chunk = ""
        start_pos = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                # Save current chunk
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    start_pos=start_pos,
                    end_pos=start_pos + len(current_chunk),
                    chunk_id=chunk_id,
                    metadata={
                        "length": len(current_chunk),
                        "paragraph_count": current_chunk.count('\n\n') + 1
                    }
                ))
                chunk_id += 1
                start_pos += len(current_chunk)
                current_chunk = ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        
        # Add final chunk if not empty
        if current_chunk.strip():
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                start_pos=start_pos,
                end_pos=start_pos + len(current_chunk),
                chunk_id=chunk_id,
                metadata={
                    "length": len(current_chunk),
                    "paragraph_count": current_chunk.count('\n\n') + 1
                }
            ))
        
        logger.debug(f"Split text into {len(chunks)} paragraph-based chunks")
        return chunks
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing text metadata
        """
        if not text:
            return {}
        
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(self._sentence_pattern.findall(text))
        paragraph_count = len(self._paragraph_pattern.split(text))
        
        # Language detection (simple heuristic)
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text) if text else 0
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "ascii_ratio": ascii_ratio,
            "avg_words_per_sentence": word_count / sentence_count if sentence_count > 0 else 0,
            "avg_chars_per_word": char_count / word_count if word_count > 0 else 0
        }
    
    def get_optimal_chunk_size(self, text: str, target_chunks: int = 10) -> int:
        """Calculate optimal chunk size for given text and target number of chunks.
        
        Args:
            text: Input text
            target_chunks: Desired number of chunks
            
        Returns:
            Recommended chunk size
        """
        if not text:
            return self.chunk_size
        
        text_length = len(text)
        optimal_size = text_length // target_chunks
        
        # Ensure it's within reasonable bounds
        min_size = 500
        max_size = 5000
        
        return max(min_size, min(max_size, optimal_size))
    
    def validate_chunks(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Validate chunk quality and return statistics.
        
        Args:
            chunks: List of text chunks to validate
            
        Returns:
            Validation statistics
        """
        if not chunks:
            return {"valid": False, "error": "No chunks provided"}
        
        total_chars = sum(len(chunk.text) for chunk in chunks)
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        
        stats = {
            "valid": True,
            "chunk_count": len(chunks),
            "total_chars": total_chars,
            "avg_chunk_size": total_chars / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "size_variance": sum((size - total_chars/len(chunks))**2 for size in chunk_sizes) / len(chunks)
        }
        
        # Check for issues
        issues = []
        if stats["min_chunk_size"] < 100:
            issues.append("Some chunks are very small (< 100 chars)")
        if stats["max_chunk_size"] > self.chunk_size * 1.2:
            issues.append("Some chunks exceed target size by >20%")
        if stats["size_variance"] > (self.chunk_size * 0.5) ** 2:
            issues.append("High variance in chunk sizes")
        
        stats["issues"] = issues
        stats["quality_score"] = max(0, 1.0 - len(issues) * 0.2)
        
        return stats
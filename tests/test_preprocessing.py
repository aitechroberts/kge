"""
Tests for text preprocessing module.
"""

import pytest
from kge.preprocessing import TextProcessor, TextChunk


class TestTextProcessor:
    """Test cases for TextProcessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = TextProcessor(chunk_size=100, chunk_overlap=20)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test basic cleaning
        dirty_text = "  Hello   world!  \n\n\n  This is a test.  "
        clean_text = self.processor.clean_text(dirty_text)
        # The cleaner normalizes whitespace, so check for basic cleaning
        assert "Hello world!" in clean_text
        assert "This is a test." in clean_text
        assert clean_text.strip() == clean_text  # No leading/trailing whitespace
        
        # Test empty text
        assert self.processor.clean_text("") == ""
        assert self.processor.clean_text(None) == ""
        
        # Test control characters
        text_with_controls = "Hello\x00world\x01test"
        clean_text = self.processor.clean_text(text_with_controls)
        assert "\x00" not in clean_text
        assert "\x01" not in clean_text
    
    def test_chunk_text_small(self):
        """Test chunking of small text."""
        text = "This is a small text."
        chunks = self.processor.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_id == 0
        assert chunks[0].start_pos == 0
        assert chunks[0].end_pos == len(text)
    
    def test_chunk_text_large(self):
        """Test chunking of large text."""
        # Create text larger than chunk size
        text = "This is a sentence. " * 10  # Should be > 100 chars
        chunks = self.processor.chunk_text(text)
        
        assert len(chunks) > 1
        
        # Check chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == i
            assert len(chunk.text) <= self.processor.chunk_size + 50  # Allow some flexibility
            assert chunk.metadata["length"] == len(chunk.text)
            assert chunk.metadata["word_count"] > 0
    
    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        text = "Word " * 50  # Create text that will need chunking
        chunks = self.processor.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                current_end = chunks[i].end_pos
                next_start = chunks[i + 1].start_pos
                assert current_end > next_start  # Overlap exists
    
    def test_chunk_by_paragraphs(self):
        """Test paragraph-based chunking."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = self.processor.chunk_by_paragraphs(text)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["paragraph_count"] >= 1
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        text = "This is a test. It has multiple sentences! And some punctuation?"
        metadata = self.processor.extract_metadata(text)
        
        assert metadata["char_count"] == len(text)
        assert metadata["word_count"] > 0
        assert metadata["sentence_count"] >= 2
        assert 0 <= metadata["ascii_ratio"] <= 1
        assert metadata["avg_words_per_sentence"] > 0
    
    def test_get_optimal_chunk_size(self):
        """Test optimal chunk size calculation."""
        text = "Word " * 1000
        optimal_size = self.processor.get_optimal_chunk_size(text, target_chunks=5)
        
        assert 500 <= optimal_size <= 5000  # Within reasonable bounds
    
    def test_validate_chunks(self):
        """Test chunk validation."""
        text = "This is a test text for validation."
        chunks = self.processor.chunk_text(text)
        validation = self.processor.validate_chunks(chunks)
        
        assert validation["valid"] is True
        assert validation["chunk_count"] == len(chunks)
        assert validation["total_chars"] > 0
        assert 0 <= validation["quality_score"] <= 1


class TestTextChunk:
    """Test cases for TextChunk."""
    
    def test_text_chunk_creation(self):
        """Test TextChunk creation."""
        chunk = TextChunk(
            text="Test text",
            start_pos=0,
            end_pos=9,
            chunk_id=0
        )
        
        assert chunk.text == "Test text"
        assert chunk.start_pos == 0
        assert chunk.end_pos == 9
        assert chunk.chunk_id == 0
        assert chunk.metadata == {}
    
    def test_text_chunk_with_metadata(self):
        """Test TextChunk with metadata."""
        metadata = {"length": 9, "word_count": 2}
        chunk = TextChunk(
            text="Test text",
            start_pos=0,
            end_pos=9,
            chunk_id=0,
            metadata=metadata
        )
        
        assert chunk.metadata == metadata


if __name__ == "__main__":
    pytest.main([__file__])
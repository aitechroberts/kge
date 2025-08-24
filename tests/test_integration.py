"""
Integration tests for the KGE system.
"""

import pytest
from unittest.mock import Mock, patch
from kge.core import KGEPipeline, KGEConfig
from kge.extractors import Entity, Relationship


class TestKGEPipelineIntegration:
    """Integration tests for KGEPipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create a test configuration with memory backend
        self.config = KGEConfig(
            model_path="test/model",
            storage_backend="memory",
            enable_caching=False,  # Disable caching for simpler tests
            chunk_size=200,
            chunk_overlap=50
        )
    
    @patch('kge.extractors.LLMExtractor')
    def test_pipeline_initialization(self, mock_extractor):
        """Test pipeline initialization."""
        # Mock the extractor to avoid loading actual model
        mock_extractor.return_value = Mock()
        
        pipeline = KGEPipeline(self.config)
        
        assert pipeline.config == self.config
        assert pipeline.text_processor is not None
        assert pipeline.extractor is not None
        assert pipeline.graph_builder is not None
        assert pipeline.storage is not None
    
    @patch('kge.extractors.LLMExtractor')
    def test_process_text_simple(self, mock_extractor):
        """Test processing simple text."""
        # Mock extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        # Mock extraction result
        mock_entities = [
            Entity(name="Apple Inc", type="Company", confidence=0.9),
            Entity(name="Tim Cook", type="Person", confidence=0.8)
        ]
        mock_relationships = [
            Relationship(source="Tim Cook", relation="WORKS_FOR", target="Apple Inc", confidence=0.9)
        ]
        
        mock_extractor_instance.extract.return_value = {
            "entities": mock_entities,
            "relationships": mock_relationships
        }
        
        pipeline = KGEPipeline(self.config)
        
        # Process text
        text = "Tim Cook works for Apple Inc. Apple Inc is a technology company."
        result = pipeline.process_text(text, document_id="test_doc")
        
        # Verify results
        assert result["document_id"] == "test_doc"
        assert result["chunks_processed"] == 1  # Short text, single chunk
        assert result["entities_extracted"] == 2
        assert result["relationships_extracted"] == 1
        assert "graph_data" in result
        assert "storage_result" in result
    
    @patch('kge.extractors.LLMExtractor')
    def test_process_text_chunking(self, mock_extractor):
        """Test processing text that requires chunking."""
        # Mock extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        # Mock extraction result for each chunk
        mock_extractor_instance.extract.return_value = {
            "entities": [Entity(name="Test Entity", type="Company", confidence=0.8)],
            "relationships": []
        }
        
        pipeline = KGEPipeline(self.config)
        
        # Create long text that will be chunked
        long_text = "This is a test sentence. " * 20  # Should exceed chunk_size
        result = pipeline.process_text(long_text, document_id="long_doc")
        
        # Should have multiple chunks
        assert result["chunks_processed"] > 1
        assert result["document_id"] == "long_doc"
    
    @patch('kge.extractors.LLMExtractor')
    def test_process_batch(self, mock_extractor):
        """Test batch processing."""
        # Mock extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        mock_extractor_instance.extract.return_value = {
            "entities": [Entity(name="Test Entity", type="Company", confidence=0.8)],
            "relationships": []
        }
        
        pipeline = KGEPipeline(self.config)
        
        # Process batch
        texts = ["Text 1", "Text 2", "Text 3"]
        document_ids = ["doc1", "doc2", "doc3"]
        
        results = pipeline.process_batch(texts, document_ids)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["document_id"] == document_ids[i]
    
    @patch('kge.extractors.LLMExtractor')
    def test_process_batch_with_errors(self, mock_extractor):
        """Test batch processing with some errors."""
        # Mock extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        # Mock extraction to raise error for second text
        def mock_extract(text):
            if "error" in text.lower():
                raise Exception("Test error")
            return {
                "entities": [Entity(name="Test Entity", type="Company", confidence=0.8)],
                "relationships": []
            }
        
        mock_extractor_instance.extract.side_effect = mock_extract
        
        pipeline = KGEPipeline(self.config)
        
        # Process batch with one error
        texts = ["Good text", "Error text", "Another good text"]
        results = pipeline.process_batch(texts)
        
        assert len(results) == 3
        assert "error" not in results[0]
        assert "error" in results[1]
        assert results[1]["success"] is False
        assert "error" not in results[2]
    
    @patch('kge.extractors.LLMExtractor')
    def test_get_statistics(self, mock_extractor):
        """Test getting pipeline statistics."""
        mock_extractor.return_value = Mock()
        
        pipeline = KGEPipeline(self.config)
        stats = pipeline.get_statistics()
        
        assert "config" in stats
        assert "storage_stats" in stats
        assert stats["config"]["storage_backend"] == "memory"
    
    @patch('kge.extractors.LLMExtractor')
    def test_cleanup(self, mock_extractor):
        """Test pipeline cleanup."""
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        pipeline = KGEPipeline(self.config)
        
        # Should not raise exception
        pipeline.cleanup()
    
    @patch('kge.extractors.LLMExtractor')
    def test_context_manager(self, mock_extractor):
        """Test pipeline as context manager."""
        mock_extractor.return_value = Mock()
        
        with KGEPipeline(self.config) as pipeline:
            assert pipeline is not None
        
        # Cleanup should have been called automatically


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @patch('kge.extractors.LLMExtractor')
    def test_complete_workflow(self, mock_extractor):
        """Test complete workflow from text to storage."""
        # Mock extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        # Mock realistic extraction result
        mock_entities = [
            Entity(name="Apple Inc", type="Company", confidence=0.9),
            Entity(name="iPhone", type="Product", confidence=0.8),
            Entity(name="Cupertino", type="Location", confidence=0.7)
        ]
        mock_relationships = [
            Relationship(source="Apple Inc", relation="DEVELOPS", target="iPhone", confidence=0.9),
            Relationship(source="Apple Inc", relation="LOCATED_IN", target="Cupertino", confidence=0.8)
        ]
        
        mock_extractor_instance.extract.return_value = {
            "entities": mock_entities,
            "relationships": mock_relationships
        }
        
        # Create pipeline
        config = KGEConfig(
            model_path="test/model",
            storage_backend="memory",
            enable_entity_resolution=True,
            min_confidence=0.6
        )
        
        pipeline = KGEPipeline(config)
        
        # Process text
        text = """
        Apple Inc is a technology company based in Cupertino, California.
        The company develops the iPhone, which is one of their most popular products.
        Apple Inc has been innovating in the smartphone market for many years.
        """
        
        result = pipeline.process_text(text, document_id="apple_doc")
        
        # Verify complete workflow
        assert result["success"] is not False  # No explicit success field, but no error
        assert result["entities_extracted"] == 3
        assert result["relationships_extracted"] == 2
        assert result["entities_final"] <= 3  # May be reduced by entity resolution
        assert result["relationships_final"] <= 2
        
        # Verify data was stored
        stored_nodes = pipeline.storage.query_nodes()
        stored_edges = pipeline.storage.query_edges()
        
        assert len(stored_nodes) > 0
        assert len(stored_edges) > 0
        
        # Verify statistics
        stats = pipeline.storage.get_statistics()
        assert stats["node_count"] > 0
        assert stats["edge_count"] > 0
    
    @patch('kge.extractors.LLMExtractor')
    def test_entity_resolution_workflow(self, mock_extractor):
        """Test workflow with entity resolution."""
        # Mock extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        # Mock entities with similar names
        mock_entities = [
            Entity(name="Apple Inc", type="Company", confidence=0.9),
            Entity(name="Apple Corporation", type="Company", confidence=0.8),
            Entity(name="Tim Cook", type="Person", confidence=0.9)
        ]
        mock_relationships = [
            Relationship(source="Tim Cook", relation="WORKS_FOR", target="Apple Inc", confidence=0.9),
            Relationship(source="Tim Cook", relation="MANAGES", target="Apple Corporation", confidence=0.8)
        ]
        
        mock_extractor_instance.extract.return_value = {
            "entities": mock_entities,
            "relationships": mock_relationships
        }
        
        # Create pipeline with entity resolution enabled
        config = KGEConfig(
            model_path="test/model",
            storage_backend="memory",
            enable_entity_resolution=True,
            entity_similarity_threshold=0.8
        )
        
        pipeline = KGEPipeline(config)
        
        # Process text
        text = "Tim Cook works for Apple Inc. He also manages Apple Corporation."
        result = pipeline.process_text(text)
        
        # Entity resolution should merge similar entities
        assert result["entities_extracted"] == 3
        assert result["entities_final"] == 2  # Apple Inc and Apple Corporation merged
        
        # Both relationships should still exist
        assert result["relationships_final"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
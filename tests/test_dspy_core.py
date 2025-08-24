"""
Tests for DSPy-based core pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from kge.dspy_core import (
    DSPyKGEConfig, DSPyKGEPipeline, 
    create_pipeline, create_vllm_pipeline, create_test_pipeline
)
from kge.dspy_config import DSPyConfig
from kge.dspy_extractors import ExtractionExample


class TestDSPyKGEConfig:
    """Test cases for DSPy KGE configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DSPyKGEConfig()
        
        assert isinstance(config.dspy_config, DSPyConfig)
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100
        assert len(config.entity_types) > 0
        assert config.enable_validation is True
        assert config.min_confidence == 0.5
        assert config.storage_backend == "memory"
        assert config.enable_caching is True
        assert config.batch_size == 10
        assert config.max_workers == 4
    
    def test_custom_config(self):
        """Test custom configuration values."""
        dspy_config = DSPyConfig(model_name="custom-model")
        
        config = DSPyKGEConfig(
            dspy_config=dspy_config,
            chunk_size=500,
            entity_types=["Person", "Organization"],
            storage_backend="neo4j",
            enable_optimization=True
        )
        
        assert config.dspy_config.model_name == "custom-model"
        assert config.chunk_size == 500
        assert config.entity_types == ["Person", "Organization"]
        assert config.storage_backend == "neo4j"
        assert config.enable_optimization is True


class TestDSPyKGEPipeline:
    """Test cases for DSPy KGE pipeline."""
    
    @patch('kge.dspy_core.setup_dspy_environment')
    @patch('kge.dspy_core.TextProcessor')
    @patch('kge.dspy_core.DSPyKGExtractor')
    @patch('kge.dspy_core.GraphBuilder')
    @patch('kge.dspy_core.GraphStorage')
    def test_pipeline_initialization(self, mock_storage, mock_graph_builder, 
                                   mock_extractor, mock_text_processor, mock_dspy_setup):
        """Test pipeline initialization."""
        config = DSPyKGEConfig()
        
        # Mock the DSPy manager
        mock_manager = Mock()
        mock_dspy_setup.return_value = mock_manager
        
        pipeline = DSPyKGEPipeline(config)
        
        assert pipeline.config == config
        assert hasattr(pipeline, 'performance_monitor')
        assert hasattr(pipeline, 'dspy_manager')
        assert hasattr(pipeline, 'text_processor')
        assert hasattr(pipeline, 'extractor')
        assert hasattr(pipeline, 'graph_builder')
        assert hasattr(pipeline, 'storage')
    
    @patch('kge.dspy_core.setup_dspy_environment')
    @patch('kge.dspy_core.TextProcessor')
    @patch('kge.dspy_core.DSPyKGExtractor')
    @patch('kge.dspy_core.GraphBuilder')
    @patch('kge.dspy_core.GraphStorage')
    def test_process_text(self, mock_storage, mock_graph_builder, 
                         mock_extractor, mock_text_processor, mock_dspy_setup):
        """Test text processing."""
        config = DSPyKGEConfig()
        
        # Mock components
        mock_manager = Mock()
        mock_dspy_setup.return_value = mock_manager
        
        mock_chunks = [Mock(text="chunk1"), Mock(text="chunk2")]
        mock_text_processor.return_value.process_text.return_value = mock_chunks
        
        mock_extraction_result = {
            "entities": [Mock()],
            "relationships": [Mock()],
            "quality_score": 0.8
        }
        mock_extractor.return_value.return_value = mock_extraction_result
        
        mock_graph_data = {
            "nodes": [Mock()],
            "edges": [Mock()],
            "metadata": {}
        }
        mock_graph_builder.return_value.build_graph.return_value = mock_graph_data
        
        mock_storage_result = {"success": True}
        mock_storage.return_value.store_graph.return_value = mock_storage_result
        
        pipeline = DSPyKGEPipeline(config)
        
        # Test processing
        result = pipeline.process_text("Test text", document_id="test_doc")
        
        assert result["document_id"] == "test_doc"
        assert result["chunks_processed"] == 2
        assert "entities_extracted" in result
        assert "entities_final" in result
        assert "relationships_extracted" in result
        assert "relationships_final" in result
        assert "average_quality_score" in result
        assert "graph_data" in result
        assert "storage_result" in result
        assert "processing_time" in result
    
    @patch('kge.dspy_core.setup_dspy_environment')
    @patch('kge.dspy_core.TextProcessor')
    @patch('kge.dspy_core.DSPyKGExtractor')
    @patch('kge.dspy_core.GraphBuilder')
    @patch('kge.dspy_core.GraphStorage')
    def test_process_batch(self, mock_storage, mock_graph_builder, 
                          mock_extractor, mock_text_processor, mock_dspy_setup):
        """Test batch processing."""
        config = DSPyKGEConfig()
        
        # Mock components
        mock_manager = Mock()
        mock_dspy_setup.return_value = mock_manager
        
        pipeline = DSPyKGEPipeline(config)
        
        # Mock process_text method
        pipeline.process_text = Mock(return_value={"entities_final": 1, "relationships_final": 1})
        
        texts = ["text1", "text2", "text3"]
        results = pipeline.process_batch(texts)
        
        assert len(results) == 3
        assert pipeline.process_text.call_count == 3
    
    @patch('kge.dspy_core.setup_dspy_environment')
    @patch('kge.dspy_core.TextProcessor')
    @patch('kge.dspy_core.DSPyKGExtractor')
    @patch('kge.dspy_core.GraphBuilder')
    @patch('kge.dspy_core.GraphStorage')
    def test_process_file(self, mock_storage, mock_graph_builder, 
                         mock_extractor, mock_text_processor, mock_dspy_setup):
        """Test file processing."""
        config = DSPyKGEConfig()
        
        # Mock components
        mock_manager = Mock()
        mock_dspy_setup.return_value = mock_manager
        
        pipeline = DSPyKGEPipeline(config)
        
        # Mock process_text method
        pipeline.process_text = Mock(return_value={"entities_final": 1})
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data="file content")):
            with patch('pathlib.Path.exists', return_value=True):
                result = pipeline.process_file("test.txt")
        
        assert result["entities_final"] == 1
        pipeline.process_text.assert_called_once_with("file content", document_id="test.txt")
    
    @patch('kge.dspy_core.setup_dspy_environment')
    @patch('kge.dspy_core.DSPyOptimizer')
    def test_optimize_with_examples(self, mock_optimizer_class, mock_dspy_setup):
        """Test optimization with examples."""
        config = DSPyKGEConfig(enable_optimization=True)
        
        # Mock components
        mock_manager = Mock()
        mock_dspy_setup.return_value = mock_manager
        
        mock_optimizer = Mock()
        mock_optimizer.optimize_with_bootstrap.return_value = Mock()
        mock_optimizer.evaluate.return_value = {"average_score": 0.8, "success_rate": 0.9}
        mock_optimizer_class.return_value = mock_optimizer
        
        pipeline = DSPyKGEPipeline(config)
        
        # Create test examples
        examples = [
            ExtractionExample(
                text="Test text",
                entities=[{"name": "Test", "type": "Organization"}],
                relationships=[]
            )
        ]
        
        result = pipeline.optimize_with_examples(examples)
        
        assert result["optimization_successful"] is True
        assert "evaluation_results" in result
        assert result["evaluation_results"]["average_score"] == 0.8
    
    @patch('kge.dspy_core.setup_dspy_environment')
    def test_optimize_without_optimizer(self, mock_dspy_setup):
        """Test optimization when optimizer is not enabled."""
        config = DSPyKGEConfig(enable_optimization=False)
        
        # Mock components
        mock_manager = Mock()
        mock_dspy_setup.return_value = mock_manager
        
        pipeline = DSPyKGEPipeline(config)
        
        examples = []
        
        with pytest.raises(ValueError, match="Optimization not enabled"):
            pipeline.optimize_with_examples(examples)
    
    @patch('kge.dspy_core.setup_dspy_environment')
    @patch('kge.dspy_core.GraphStorage')
    def test_query_graph(self, mock_storage, mock_dspy_setup):
        """Test graph querying."""
        config = DSPyKGEConfig()
        
        # Mock components
        mock_manager = Mock()
        mock_dspy_setup.return_value = mock_manager
        
        mock_query_result = {"nodes": [{"name": "Apple"}]}
        mock_storage.return_value.query_nodes.return_value = mock_query_result
        
        pipeline = DSPyKGEPipeline(config)
        
        result = pipeline.query_graph("Apple", limit=50)
        
        assert result == mock_query_result
        mock_storage.return_value.query_nodes.assert_called_once_with(
            filters={"name": "Apple"}, limit=50
        )
    
    @patch('kge.dspy_core.setup_dspy_environment')
    @patch('kge.dspy_core.GraphStorage')
    def test_get_statistics(self, mock_storage, mock_dspy_setup):
        """Test getting pipeline statistics."""
        config = DSPyKGEConfig()
        
        # Mock components
        mock_manager = Mock()
        mock_manager.get_model_info.return_value = {"model_name": "test-model"}
        mock_dspy_setup.return_value = mock_manager
        
        mock_storage.return_value.get_statistics.return_value = {"nodes": 10}
        
        pipeline = DSPyKGEPipeline(config)
        
        stats = pipeline.get_statistics()
        
        assert "storage_statistics" in stats
        assert "performance_statistics" in stats
        assert "model_information" in stats
        assert "configuration" in stats
        
        assert stats["model_information"]["model_name"] == "test-model"
        assert stats["storage_statistics"]["nodes"] == 10
    
    @patch('kge.dspy_core.setup_dspy_environment')
    @patch('kge.dspy_core.GraphStorage')
    @patch('kge.dspy_core.DataExporter')
    def test_export_graph(self, mock_exporter_class, mock_storage, mock_dspy_setup):
        """Test graph export functionality."""
        config = DSPyKGEConfig()
        
        # Mock components
        mock_manager = Mock()
        mock_dspy_setup.return_value = mock_manager
        
        mock_storage.return_value.query_nodes.return_value = {"nodes": []}
        mock_storage.return_value.query_edges.return_value = {"edges": []}
        
        mock_exporter = Mock()
        mock_exporter.to_json.return_value = '{"test": "json"}'
        mock_exporter_class.return_value = mock_exporter
        
        pipeline = DSPyKGEPipeline(config)
        
        result = pipeline.export_graph(format="json")
        
        assert result == '{"test": "json"}'
        mock_exporter.to_json.assert_called_once()
    
    @patch('kge.dspy_core.setup_dspy_environment')
    def test_context_manager(self, mock_dspy_setup):
        """Test pipeline as context manager."""
        config = DSPyKGEConfig()
        
        # Mock components
        mock_manager = Mock()
        mock_dspy_setup.return_value = mock_manager
        
        with DSPyKGEPipeline(config) as pipeline:
            assert isinstance(pipeline, DSPyKGEPipeline)
        
        # Cleanup should be called automatically


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @patch('kge.dspy_core.DSPyKGEPipeline')
    def test_create_pipeline(self, mock_pipeline_class):
        """Test create_pipeline convenience function."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        result = create_pipeline(
            model_type="openai",
            model_name="gpt-4",
            api_key="test-key",
            storage_backend="neo4j",
            enable_optimization=True
        )
        
        assert result == mock_pipeline
        
        # Check that the pipeline was created with correct config
        call_args = mock_pipeline_class.call_args[0][0]  # First positional argument (config)
        assert call_args.dspy_config.model_type == "openai"
        assert call_args.dspy_config.model_name == "gpt-4"
        assert call_args.dspy_config.api_key == "test-key"
        assert call_args.storage_backend == "neo4j"
        assert call_args.enable_optimization is True
    
    @patch('kge.dspy_core.DSPyKGEPipeline')
    @patch('kge.dspy_core.get_vllm_config')
    def test_create_vllm_pipeline(self, mock_get_vllm_config, mock_pipeline_class):
        """Test create_vllm_pipeline convenience function."""
        mock_vllm_config = Mock()
        mock_get_vllm_config.return_value = mock_vllm_config
        
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        result = create_vllm_pipeline(
            model_path="test/model",
            host="localhost",
            port=8000,
            storage_backend="kuzu"
        )
        
        assert result == mock_pipeline
        mock_get_vllm_config.assert_called_once_with("test/model", "localhost", 8000)
    
    @patch('kge.dspy_core.DSPyKGEPipeline')
    def test_create_test_pipeline(self, mock_pipeline_class):
        """Test create_test_pipeline convenience function."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        result = create_test_pipeline()
        
        assert result == mock_pipeline
        
        # Check that the pipeline was created with test config
        call_args = mock_pipeline_class.call_args[0][0]  # First positional argument (config)
        assert call_args.dspy_config.model_name == "mock-model"
        assert call_args.dspy_config.model_type == "mock"
        assert call_args.storage_backend == "memory"
        assert call_args.enable_optimization is False


# Helper function for mocking file operations
def mock_open(read_data=""):
    """Create a mock for file operations."""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open(read_data=read_data)
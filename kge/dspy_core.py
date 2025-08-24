"""
DSPy-powered core pipeline for knowledge graph extraction.

This module provides the main DSPyKGEPipeline class that orchestrates the entire
knowledge graph extraction process using DSPy for optimized LLM interactions.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from .preprocessing import TextProcessor, TextChunk
from .graph import GraphBuilder
from .storage import GraphStorage
from .optimization import PerformanceOptimizer
from .utils import PerformanceMonitor, setup_logging
from .dspy_extractors import DSPyKGExtractor, DSPyOptimizer, ExtractionExample
from .dspy_config import DSPyConfig, DSPyModelManager, setup_dspy_environment

logger = logging.getLogger(__name__)


@dataclass
class DSPyKGEConfig:
    """Configuration for DSPy-powered KGE pipeline."""
    
    # DSPy Model Configuration
    dspy_config: DSPyConfig = field(default_factory=lambda: DSPyConfig())
    
    # Text Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    
    # Entity and Relationship Extraction
    entity_types: List[str] = field(default_factory=lambda: [
        "Person", "Organization", "Location", "Event", "Product", 
        "Technology", "Concept", "Date", "Money", "Quantity"
    ])
    enable_validation: bool = True
    min_confidence: float = 0.5
    
    # Graph Construction
    enable_entity_resolution: bool = True
    entity_similarity_threshold: float = 0.85
    relationship_confidence_threshold: float = 0.6
    
    # Storage
    storage_backend: str = "memory"  # memory, neo4j, kuzu
    storage_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance and Optimization
    enable_caching: bool = True
    cache_dir: Optional[str] = None
    batch_size: int = 10
    max_workers: int = 4
    
    # DSPy Optimization
    enable_optimization: bool = False
    optimization_examples: int = 50
    max_bootstrapped_demos: int = 8
    max_labeled_demos: int = 4
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None


class DSPyKGEPipeline:
    """DSPy-powered knowledge graph extraction pipeline."""
    
    def __init__(self, config: DSPyKGEConfig):
        """Initialize the DSPy KGE pipeline."""
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        
        # Setup logging
        setup_logging(level=config.log_level)
        
        # Initialize components
        self._setup_components()
        
        logger.info("DSPy KGE Pipeline initialized successfully")
    
    def _setup_components(self):
        """Setup all pipeline components."""
        # Setup DSPy environment
        self.dspy_manager = setup_dspy_environment(self.config.dspy_config)
        
        # Initialize text processor
        self.text_processor = TextProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Initialize DSPy extractor
        self.extractor = DSPyKGExtractor(
            entity_types=self.config.entity_types,
            enable_validation=self.config.enable_validation
        )
        
        # Initialize graph builder
        self.graph_builder = GraphBuilder(
            enable_entity_resolution=self.config.enable_entity_resolution,
            entity_similarity_threshold=self.config.entity_similarity_threshold,
            min_confidence=self.config.min_confidence
        )
        
        # Initialize storage
        self.storage = GraphStorage(
            backend=self.config.storage_backend,
            config=self.config.storage_config
        )
        
        # Initialize optimizer if enabled
        if self.config.enable_optimization:
            self.optimizer = DSPyOptimizer(self.extractor)
        else:
            self.optimizer = None
        
        # Initialize performance optimizer
        self.perf_optimizer = PerformanceOptimizer(
            enable_caching=self.config.enable_caching,
            cache_dir=self.config.cache_dir
        )
    
    def process_text(self, text: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a single text document through the complete pipeline."""
        start_time = time.time()
        self.performance_monitor.start_operation("process_text")
        
        try:
            # Preprocess text
            chunks = self.text_processor.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks from text")
            
            # Extract information from chunks
            all_entities = []
            all_relationships = []
            total_quality_score = 0.0
            
            for chunk in chunks:
                chunk_result = self.extractor(chunk.text)
                all_entities.extend(chunk_result["entities"])
                all_relationships.extend(chunk_result["relationships"])
                total_quality_score += chunk_result["quality_score"]
            
            avg_quality_score = total_quality_score / len(chunks) if chunks else 0.0
            
            # Build knowledge graph
            graph_data = self.graph_builder.build_graph(
                all_entities, 
                all_relationships,
                document_id=document_id
            )
            
            # Store graph
            storage_result = self.storage.store_graph(graph_data)
            
            # Prepare result
            result = {
                "document_id": document_id,
                "chunks_processed": len(chunks),
                "entities_extracted": len(all_entities),
                "entities_final": len(graph_data["nodes"]),
                "relationships_extracted": len(all_relationships),
                "relationships_final": len(graph_data["edges"]),
                "average_quality_score": avg_quality_score,
                "graph_data": graph_data,
                "storage_result": storage_result,
                "processing_time": time.time() - start_time
            }
            
            self.performance_monitor.end_operation("process_text", success=True)
            logger.info(f"Successfully processed text: {result['entities_final']} entities, {result['relationships_final']} relationships")
            
            return result
            
        except Exception as e:
            self.performance_monitor.end_operation("process_text", success=False, error=str(e))
            logger.error(f"Error processing text: {e}")
            raise
    
    def process_batch(self, texts: List[str], document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process multiple texts in batch."""
        self.performance_monitor.start_operation("process_batch")
        
        try:
            if document_ids is None:
                document_ids = [f"doc_{i}" for i in range(len(texts))]
            
            results = []
            for i, text in enumerate(texts):
                doc_id = document_ids[i] if i < len(document_ids) else f"doc_{i}"
                result = self.process_text(text, doc_id)
                results.append(result)
            
            self.performance_monitor.end_operation("process_batch", success=True)
            logger.info(f"Successfully processed batch of {len(texts)} texts")
            
            return results
            
        except Exception as e:
            self.performance_monitor.end_operation("process_batch", success=False, error=str(e))
            logger.error(f"Error processing batch: {e}")
            raise
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a text file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        
        return self.process_text(text, document_id=str(file_path))
    
    def optimize_with_examples(self, examples: List[ExtractionExample]) -> Dict[str, Any]:
        """Optimize the DSPy extractor using training examples."""
        if not self.optimizer:
            raise ValueError("Optimization not enabled. Set enable_optimization=True in config.")
        
        self.performance_monitor.start_operation("optimize_extractor")
        
        try:
            # Split examples into training and validation
            split_idx = int(len(examples) * 0.8)
            train_examples = examples[:split_idx]
            val_examples = examples[split_idx:]
            
            # Add examples to optimizer
            for example in train_examples:
                self.optimizer.add_training_example(example)
            
            for example in val_examples:
                self.optimizer.add_validation_example(example)
            
            # Optimize the extractor
            optimized_extractor = self.optimizer.optimize_with_bootstrap(
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos
            )
            
            # Evaluate the optimized extractor
            evaluation_results = self.optimizer.evaluate()
            
            # Update the pipeline's extractor
            self.extractor = optimized_extractor
            
            self.performance_monitor.end_operation("optimize_extractor", success=True)
            
            result = {
                "training_examples": len(train_examples),
                "validation_examples": len(val_examples),
                "evaluation_results": evaluation_results,
                "optimization_successful": True
            }
            
            logger.info(f"Successfully optimized extractor with {len(examples)} examples")
            return result
            
        except Exception as e:
            self.performance_monitor.end_operation("optimize_extractor", success=False, error=str(e))
            logger.error(f"Error optimizing extractor: {e}")
            raise
    
    def query_graph(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """Query the stored knowledge graph."""
        return self.storage.query_nodes(filters={"name": query}, limit=limit)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        storage_stats = self.storage.get_statistics()
        performance_stats = self.performance_monitor.get_summary()
        model_info = self.dspy_manager.get_model_info()
        
        return {
            "storage_statistics": storage_stats,
            "performance_statistics": performance_stats,
            "model_information": model_info,
            "configuration": {
                "entity_types": self.config.entity_types,
                "enable_validation": self.config.enable_validation,
                "enable_optimization": self.config.enable_optimization,
                "storage_backend": self.config.storage_backend
            }
        }
    
    def export_graph(self, format: str = "json", include_metadata: bool = True) -> str:
        """Export the knowledge graph in specified format."""
        # Get all nodes and edges
        nodes = self.storage.query_nodes(limit=10000)
        edges = self.storage.query_edges(limit=10000)
        
        graph_data = {
            "nodes": nodes.get("nodes", []),
            "edges": edges.get("edges", [])
        }
        
        if include_metadata:
            graph_data["metadata"] = self.get_statistics()
        
        # Use existing export utilities
        from .utils import DataExporter
        exporter = DataExporter()
        
        if format.lower() == "json":
            return exporter.to_json(graph_data, pretty=True)
        elif format.lower() == "cypher":
            return exporter.to_cypher(graph_data)
        elif format.lower() == "graphml":
            return exporter.to_graphml(graph_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.storage, 'cleanup'):
            self.storage.cleanup()
        
        logger.info("Pipeline cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Convenience functions for quick setup
def create_pipeline(
    model_type: str = "openai",
    model_name: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    storage_backend: str = "memory",
    enable_optimization: bool = False
) -> DSPyKGEPipeline:
    """Create a DSPy KGE pipeline with common configurations."""
    
    dspy_config = DSPyConfig(
        model_name=model_name,
        model_type=model_type,
        api_key=api_key
    )
    
    config = DSPyKGEConfig(
        dspy_config=dspy_config,
        storage_backend=storage_backend,
        enable_optimization=enable_optimization
    )
    
    return DSPyKGEPipeline(config)


def create_vllm_pipeline(
    model_path: str,
    host: str = "localhost",
    port: int = 8000,
    storage_backend: str = "memory"
) -> DSPyKGEPipeline:
    """Create a DSPy KGE pipeline with vLLM backend."""
    
    from .dspy_config import get_vllm_config
    
    dspy_config = get_vllm_config(model_path, host, port)
    
    config = DSPyKGEConfig(
        dspy_config=dspy_config,
        storage_backend=storage_backend,
        enable_optimization=True  # Enable optimization for vLLM
    )
    
    return DSPyKGEPipeline(config)


def create_test_pipeline() -> DSPyKGEPipeline:
    """Create a DSPy KGE pipeline for testing."""
    
    from .dspy_config import configure_for_testing
    
    # This will use a mock model
    dspy_config = DSPyConfig(
        model_name="mock-model",
        model_type="mock",
        enable_caching=False
    )
    
    config = DSPyKGEConfig(
        dspy_config=dspy_config,
        storage_backend="memory",
        enable_optimization=False,
        log_level="DEBUG"
    )
    
    return DSPyKGEPipeline(config)
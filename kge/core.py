"""
Core pipeline for knowledge graph extraction.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from .preprocessing import TextProcessor
from .extractors import LLMExtractor
from .graph import GraphBuilder
from .storage import GraphStorage
from .utils import setup_logging, PerformanceMonitor
from .optimization import PerformanceOptimizer

logger = logging.getLogger(__name__)


@dataclass
class KGEConfig:
    """Configuration for KGE pipeline."""
    
    # Model configuration
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    quantization: str = "awq_marlin"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85
    
    # Processing configuration
    chunk_size: int = 2500
    chunk_overlap: int = 200
    batch_size: int = 10
    
    # Extraction configuration
    max_tokens: int = 384
    temperature: float = 0.0
    confidence_threshold: float = 0.7
    
    # Graph configuration
    entity_similarity_threshold: float = 0.85
    enable_entity_resolution: bool = True
    
    # Storage configuration
    storage_backend: str = "neo4j"  # neo4j, kuzu, or memory
    storage_config: Dict[str, Any] = None
    
    # Performance configuration
    enable_caching: bool = True
    enable_monitoring: bool = True
    
    def __post_init__(self):
        if self.storage_config is None:
            self.storage_config = {}


class KGEPipeline:
    """Main pipeline for knowledge graph extraction."""
    
    def __init__(self, config: Optional[KGEConfig] = None):
        """Initialize the KGE pipeline.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or KGEConfig()
        self.monitor = PerformanceMonitor() if self.config.enable_monitoring else None
        
        # Initialize components
        self._setup_logging()
        self._initialize_components()
        
        logger.info("KGE Pipeline initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        setup_logging()
    
    def _initialize_components(self):
        """Initialize pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # Text processor
        self.text_processor = TextProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Information extractor
        self.extractor = LLMExtractor(
            model_path=self.config.model_path,
            quantization=self.config.quantization,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        # Graph builder
        self.graph_builder = GraphBuilder(
            entity_similarity_threshold=self.config.entity_similarity_threshold,
            enable_entity_resolution=self.config.enable_entity_resolution
        )
        
        # Storage layer
        self.storage = GraphStorage(
            backend=self.config.storage_backend,
            config=self.config.storage_config
        )
        
        # Performance optimizer
        if self.config.enable_caching:
            from pathlib import Path
            self.optimizer = PerformanceOptimizer(
                enable_caching=self.config.enable_caching,
                cache_dir=Path("./cache") if self.config.enable_caching else None
            )
        else:
            self.optimizer = None
        
        logger.info("All components initialized successfully")
    
    def process_text(self, text: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a single text document.
        
        Args:
            text: Input text to process
            document_id: Optional document identifier
            
        Returns:
            Dictionary containing extracted entities, relationships, and metadata
        """
        if self.monitor:
            self.monitor.start_operation("process_text")
        
        try:
            # Preprocess text
            chunks = self.text_processor.chunk_text(text)
            logger.info(f"Text split into {len(chunks)} chunks")
            
            # Extract information from chunks
            all_entities = []
            all_relationships = []
            
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                
                extraction_result = self.extractor.extract(chunk)
                all_entities.extend(extraction_result.get("entities", []))
                all_relationships.extend(extraction_result.get("relationships", []))
            
            # Build knowledge graph
            graph_data = self.graph_builder.build_graph(
                entities=all_entities,
                relationships=all_relationships,
                document_id=document_id
            )
            
            # Store in graph database
            storage_result = self.storage.store_graph(graph_data)
            
            result = {
                "document_id": document_id,
                "chunks_processed": len(chunks),
                "entities_extracted": len(all_entities),
                "relationships_extracted": len(all_relationships),
                "entities_final": len(graph_data["entities"]),
                "relationships_final": len(graph_data["relationships"]),
                "storage_result": storage_result,
                "graph_data": graph_data
            }
            
            if self.monitor:
                self.monitor.end_operation("process_text")
                result["performance"] = self.monitor.get_stats()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            if self.monitor:
                self.monitor.end_operation("process_text", error=True)
            raise
    
    def process_batch(self, texts: List[str], document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process a batch of text documents.
        
        Args:
            texts: List of input texts
            document_ids: Optional list of document identifiers
            
        Returns:
            List of processing results
        """
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(texts))]
        
        if len(texts) != len(document_ids):
            raise ValueError("Number of texts and document_ids must match")
        
        logger.info(f"Processing batch of {len(texts)} documents")
        
        results = []
        for text, doc_id in zip(texts, document_ids):
            try:
                result = self.process_text(text, doc_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {str(e)}")
                results.append({
                    "document_id": doc_id,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Processing result
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.process_text(text, document_id=file_path.stem)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            "config": self.config.__dict__,
            "storage_stats": self.storage.get_statistics(),
        }
        
        if self.monitor:
            stats["performance"] = self.monitor.get_summary()
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up pipeline resources...")
        
        if hasattr(self.extractor, 'cleanup'):
            self.extractor.cleanup()
        
        if hasattr(self.storage, 'cleanup'):
            self.storage.cleanup()
        
        logger.info("Cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
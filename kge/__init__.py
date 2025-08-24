"""
Knowledge Graph Extraction (KGE) System

An optimized system for creating knowledge graphs from text through
information extraction using large language models with DSPy optimization.
"""

__version__ = "0.2.0"
__author__ = "KGE Team"

# DSPy components (preferred)
try:
    from .dspy_core import DSPyKGEPipeline, DSPyKGEConfig, create_pipeline, create_vllm_pipeline, create_test_pipeline
    from .dspy_extractors import DSPyKGExtractor, DSPyOptimizer, ExtractionExample
    from .dspy_config import DSPyConfig, DSPyModelManager, setup_dspy_environment
    _has_dspy = True
except ImportError:
    _has_dspy = False

# Original components (fallback)
try:
    from .core import KGEPipeline as OriginalKGEPipeline, KGEConfig
    _has_core = True
except ImportError:
    _has_core = False

try:
    from .extractors import LLMExtractor
    _has_extractors = True
except ImportError:
    _has_extractors = False

# Always available components
from .entities import Entity, Relationship
from .preprocessing import TextProcessor, TextChunk
from .graph import GraphBuilder, GraphNode, GraphEdge, EntityResolver
from .storage import GraphStorage
from .utils import PerformanceMonitor, DataExporter, setup_logging

# Optional components
try:
    from .optimization import PerformanceOptimizer
    _has_optimization = True
except ImportError:
    _has_optimization = False

try:
    from .api import create_app
    _has_api = True
except ImportError:
    _has_api = False

# Export main components
__all__ = [
    # Core classes
    "Entity", "Relationship",
    "TextProcessor", "TextChunk", 
    "GraphBuilder", "GraphNode", "GraphEdge", "EntityResolver",
    "GraphStorage",
    "PerformanceMonitor", "DataExporter", "setup_logging",
]

# Add DSPy components (preferred)
if _has_dspy:
    __all__.extend([
        "DSPyKGEPipeline", "DSPyKGEConfig", "create_pipeline", "create_vllm_pipeline", "create_test_pipeline",
        "DSPyKGExtractor", "DSPyOptimizer", "ExtractionExample",
        "DSPyConfig", "DSPyModelManager", "setup_dspy_environment"
    ])
    # Use DSPy pipeline as the default
    KGEPipeline = DSPyKGEPipeline
    KGEConfig = DSPyKGEConfig
    __all__.extend(["KGEPipeline", "KGEConfig"])

# Add original components
if _has_core:
    __all__.extend(["OriginalKGEPipeline"])

if _has_extractors:
    __all__.append("LLMExtractor")

if _has_optimization:
    __all__.append("PerformanceOptimizer")

if _has_api:
    __all__.append("create_app")
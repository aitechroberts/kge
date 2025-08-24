"""
Knowledge Graph Extraction (KGE) System

An optimized system for creating knowledge graphs from text through
information extraction using large language models.
"""

__version__ = "0.1.0"
__author__ = "KGE Team"

# Import core components conditionally to handle missing dependencies
try:
    from .core import KGEPipeline
    _has_core = True
except ImportError:
    _has_core = False

try:
    from .extractors import LLMExtractor
    _has_extractors = True
except ImportError:
    _has_extractors = False

from .graph import GraphBuilder
from .storage import GraphStorage

__all__ = ["GraphBuilder", "GraphStorage"]

if _has_core:
    __all__.append("KGEPipeline")

if _has_extractors:
    __all__.append("LLMExtractor")
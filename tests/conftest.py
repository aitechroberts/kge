"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

try:
    from kge.core import KGEConfig
    _has_core = True
except ImportError:
    _has_core = False
    # Create a mock config for testing
    class KGEConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

from kge.entities import Entity, Relationship
from kge.graph import GraphNode, GraphEdge


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample KGE configuration for testing."""
    return KGEConfig(
        model_path="test/model",
        storage_backend="memory",
        chunk_size=500,
        chunk_overlap=100,
        batch_size=5,
        max_tokens=200,
        temperature=0.0,
        confidence_threshold=0.7,
        entity_similarity_threshold=0.85,
        enable_entity_resolution=True,
        enable_caching=False
    )


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(name="Apple Inc", type="Company", confidence=0.9, description="Technology company"),
        Entity(name="Tim Cook", type="Person", confidence=0.8, description="CEO of Apple"),
        Entity(name="iPhone", type="Product", confidence=0.85, description="Smartphone"),
        Entity(name="Cupertino", type="Location", confidence=0.7, description="City in California")
    ]


@pytest.fixture
def sample_relationships():
    """Create sample relationships for testing."""
    return [
        Relationship(source="Tim Cook", relation="WORKS_FOR", target="Apple Inc", confidence=0.9),
        Relationship(source="Apple Inc", relation="DEVELOPS", target="iPhone", confidence=0.85),
        Relationship(source="Apple Inc", relation="LOCATED_IN", target="Cupertino", confidence=0.8)
    ]


@pytest.fixture
def sample_graph_nodes():
    """Create sample graph nodes for testing."""
    node1 = GraphNode(
        entity_id="ent:123",
        name="Apple Inc",
        type="Company",
        confidence=0.9,
        description="Technology company"
    )
    node1.aliases.add("apple corp")
    node1.source_documents.add("doc1")
    
    node2 = GraphNode(
        entity_id="ent:456",
        name="Tim Cook",
        type="Person",
        confidence=0.8,
        description="CEO"
    )
    node2.source_documents.add("doc1")
    
    return [node1, node2]


@pytest.fixture
def sample_graph_edges():
    """Create sample graph edges for testing."""
    edge = GraphEdge(
        source_id="ent:456",
        target_id="ent:123",
        relation="WORKS_FOR",
        confidence=0.9
    )
    edge.source_documents.add("doc1")
    
    return [edge]


@pytest.fixture
def sample_graph_data(sample_graph_nodes, sample_graph_edges):
    """Create sample graph data for testing."""
    return {
        "nodes": sample_graph_nodes,
        "edges": sample_graph_edges,
        "statistics": {
            "node_count": len(sample_graph_nodes),
            "edge_count": len(sample_graph_edges),
            "node_types": {"Company": 1, "Person": 1},
            "relation_types": {"WORKS_FOR": 1}
        },
        "metadata": {
            "document_id": "test_doc",
            "original_entities": 2,
            "original_relationships": 1
        }
    }


@pytest.fixture
def mock_llm_extractor():
    """Create a mock LLM extractor for testing."""
    mock_extractor = Mock()
    
    # Default extraction result
    mock_extractor.extract.return_value = {
        "entities": [
            Entity(name="Test Company", type="Company", confidence=0.8),
            Entity(name="Test Person", type="Person", confidence=0.7)
        ],
        "relationships": [
            Relationship(source="Test Person", relation="WORKS_FOR", target="Test Company", confidence=0.8)
        ],
        "metadata": {"generation_time": 0.1}
    }
    
    # Batch extraction
    def mock_extract_batch(texts):
        return [mock_extractor.extract.return_value for _ in texts]
    
    mock_extractor.extract_batch = mock_extract_batch
    
    # Supported types
    mock_extractor.get_supported_types.return_value = {
        "entity_types": ["Company", "Person", "Location", "Product"],
        "relation_types": ["WORKS_FOR", "LOCATED_IN", "DEVELOPS", "OWNS"]
    }
    
    return mock_extractor


@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "Apple Inc is a technology company based in Cupertino.",
        "Tim Cook is the CEO of Apple Inc.",
        "The iPhone is Apple's flagship product.",
        "Microsoft Corporation competes with Apple in various markets.",
        "Google develops Android, which competes with iOS."
    ]


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise in tests


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require actual model"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers to tests based on their names/paths
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        if "slow" in item.nodeid or "test_complete_workflow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_gpu)
        
        if "model" in item.nodeid.lower() and "mock" not in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_model)
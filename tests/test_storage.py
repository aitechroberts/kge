"""
Tests for storage module.
"""

import pytest
import tempfile
from pathlib import Path
from kge.storage import GraphStorage, MemoryBackend, GraphStorageBackend
from kge.graph import GraphNode, GraphEdge


class TestMemoryBackend:
    """Test cases for MemoryBackend."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.backend = MemoryBackend()
        self.backend.connect({})
    
    def test_connect(self):
        """Test backend connection."""
        assert self.backend.connected is True
    
    def test_store_empty_graph(self):
        """Test storing empty graph."""
        graph_data = {"nodes": [], "edges": []}
        result = self.backend.store_graph(graph_data)
        
        assert result["success"] is True
        assert result["nodes_created"] == 0
        assert result["edges_created"] == 0
        assert result["backend"] == "memory"
    
    def test_store_simple_graph(self):
        """Test storing simple graph."""
        # Create test nodes
        node1 = GraphNode(
            entity_id="ent:123",
            name="Apple Inc",
            type="Company",
            confidence=0.9
        )
        
        node2 = GraphNode(
            entity_id="ent:456",
            name="Tim Cook",
            type="Person",
            confidence=0.8
        )
        
        # Create test edge
        edge = GraphEdge(
            source_id="ent:456",
            target_id="ent:123",
            relation="WORKS_FOR",
            confidence=0.9
        )
        
        graph_data = {"nodes": [node1, node2], "edges": [edge]}
        result = self.backend.store_graph(graph_data)
        
        assert result["success"] is True
        assert result["nodes_created"] == 2
        assert result["edges_created"] == 1
    
    def test_query_nodes_all(self):
        """Test querying all nodes."""
        # Store test data first
        node = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company")
        graph_data = {"nodes": [node], "edges": []}
        self.backend.store_graph(graph_data)
        
        # Query all nodes
        nodes = self.backend.query_nodes()
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Apple Inc"
    
    def test_query_nodes_filtered(self):
        """Test querying nodes with filters."""
        # Store test data
        node1 = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company", confidence=0.9)
        node2 = GraphNode(entity_id="ent:456", name="Tim Cook", type="Person", confidence=0.8)
        graph_data = {"nodes": [node1, node2], "edges": []}
        self.backend.store_graph(graph_data)
        
        # Query by type
        companies = self.backend.query_nodes({"type": "Company"})
        assert len(companies) == 1
        assert companies[0]["name"] == "Apple Inc"
        
        # Query by confidence
        high_conf = self.backend.query_nodes({"min_confidence": 0.85})
        assert len(high_conf) == 1
        assert high_conf[0]["name"] == "Apple Inc"
    
    def test_query_edges_all(self):
        """Test querying all edges."""
        # Store test data
        node1 = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company")
        node2 = GraphNode(entity_id="ent:456", name="Tim Cook", type="Person")
        edge = GraphEdge(source_id="ent:456", target_id="ent:123", relation="WORKS_FOR")
        
        graph_data = {"nodes": [node1, node2], "edges": [edge]}
        self.backend.store_graph(graph_data)
        
        # Query all edges
        edges = self.backend.query_edges()
        assert len(edges) == 1
        assert edges[0]["relation"] == "WORKS_FOR"
    
    def test_query_edges_filtered(self):
        """Test querying edges with filters."""
        # Store test data
        node1 = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company")
        node2 = GraphNode(entity_id="ent:456", name="Tim Cook", type="Person")
        edge1 = GraphEdge(source_id="ent:456", target_id="ent:123", relation="WORKS_FOR", confidence=0.9)
        edge2 = GraphEdge(source_id="ent:456", target_id="ent:123", relation="MANAGES", confidence=0.7)
        
        graph_data = {"nodes": [node1, node2], "edges": [edge1, edge2]}
        self.backend.store_graph(graph_data)
        
        # Query by relation
        works_for = self.backend.query_edges({"relation": "WORKS_FOR"})
        assert len(works_for) == 1
        assert works_for[0]["relation"] == "WORKS_FOR"
        
        # Query by confidence
        high_conf = self.backend.query_edges({"min_confidence": 0.8})
        assert len(high_conf) == 1
        assert high_conf[0]["relation"] == "WORKS_FOR"
    
    def test_get_statistics(self):
        """Test getting statistics."""
        # Store test data
        node1 = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company")
        node2 = GraphNode(entity_id="ent:456", name="Tim Cook", type="Person")
        edge = GraphEdge(source_id="ent:456", target_id="ent:123", relation="WORKS_FOR")
        
        graph_data = {"nodes": [node1, node2], "edges": [edge]}
        self.backend.store_graph(graph_data)
        
        stats = self.backend.get_statistics()
        
        assert stats["connected"] is True
        assert stats["backend"] == "memory"
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert "node_types" in stats
        assert stats["node_types"]["Company"] == 1
        assert stats["node_types"]["Person"] == 1
    
    def test_cleanup(self):
        """Test cleanup."""
        # Store some data
        node = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company")
        graph_data = {"nodes": [node], "edges": []}
        self.backend.store_graph(graph_data)
        
        # Cleanup
        self.backend.cleanup()
        
        assert self.backend.connected is False
        assert len(self.backend.nodes) == 0
        assert len(self.backend.edges) == 0


class TestGraphStorage:
    """Test cases for GraphStorage."""
    
    def test_memory_backend_initialization(self):
        """Test initialization with memory backend."""
        storage = GraphStorage(backend="memory")
        
        assert storage.backend_name == "memory"
        assert isinstance(storage.backend, MemoryBackend)
        assert storage.backend.connected is True
    
    def test_unsupported_backend(self):
        """Test initialization with unsupported backend."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            GraphStorage(backend="unsupported")
    
    def test_store_and_query_integration(self):
        """Test integration of store and query operations."""
        storage = GraphStorage(backend="memory")
        
        # Create test data
        node = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company")
        edge = GraphEdge(source_id="ent:123", target_id="ent:123", relation="SELF_REF")
        
        graph_data = {"nodes": [node], "edges": [edge]}
        
        # Store data
        result = storage.store_graph(graph_data)
        assert result["success"] is True
        
        # Query nodes
        nodes = storage.query_nodes()
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Apple Inc"
        
        # Query edges
        edges = storage.query_edges()
        assert len(edges) == 1
        assert edges[0]["relation"] == "SELF_REF"
        
        # Get statistics
        stats = storage.get_statistics()
        assert stats["node_count"] == 1
        assert stats["edge_count"] == 1
    
    def test_cleanup(self):
        """Test storage cleanup."""
        storage = GraphStorage(backend="memory")
        
        # Store some data
        node = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company")
        graph_data = {"nodes": [node], "edges": []}
        storage.store_graph(graph_data)
        
        # Cleanup
        storage.cleanup()
        
        # Backend should be cleaned up
        assert storage.backend.connected is False


# Skip Neo4j and Kuzu tests if dependencies not available
@pytest.mark.skipif(True, reason="Neo4j not available in test environment")
class TestNeo4jBackend:
    """Test cases for Neo4jBackend (skipped if Neo4j not available)."""
    
    def test_neo4j_placeholder(self):
        """Placeholder test for Neo4j backend."""
        # This would contain actual Neo4j tests if the database was available
        pass


@pytest.mark.skipif(True, reason="Kuzu not available in test environment")
class TestKuzuBackend:
    """Test cases for KuzuBackend (skipped if Kuzu not available)."""
    
    def test_kuzu_placeholder(self):
        """Placeholder test for Kuzu backend."""
        # This would contain actual Kuzu tests if the database was available
        pass


if __name__ == "__main__":
    pytest.main([__file__])
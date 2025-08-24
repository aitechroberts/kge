"""
Tests for graph construction module.
"""

import pytest
from kge.graph import GraphBuilder, GraphNode, GraphEdge, EntityResolver
from kge.entities import Entity, Relationship


class TestEntityResolver:
    """Test cases for EntityResolver."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = EntityResolver(similarity_threshold=0.85)
    
    def test_calculate_similarity_exact(self):
        """Test exact name similarity."""
        similarity = self.resolver.calculate_similarity("Apple Inc", "Apple Inc")
        assert similarity == 1.0
    
    def test_calculate_similarity_different(self):
        """Test different name similarity."""
        similarity = self.resolver.calculate_similarity("Apple Inc", "Microsoft Corp")
        assert similarity < 0.5
    
    def test_calculate_similarity_abbreviation(self):
        """Test abbreviation matching."""
        similarity = self.resolver.calculate_similarity("IBM", "International Business Machines")
        assert similarity > 0.8
    
    def test_calculate_similarity_partial(self):
        """Test partial matching."""
        similarity = self.resolver.calculate_similarity("Apple Inc", "Apple Corporation")
        assert similarity > 0.6
    
    def test_resolve_entities_empty(self):
        """Test entity resolution with empty list."""
        result = self.resolver.resolve_entities([])
        assert result == {}
    
    def test_resolve_entities_single(self):
        """Test entity resolution with single entity."""
        entity = Entity(name="Apple Inc", type="Company")
        result = self.resolver.resolve_entities([entity])
        
        assert len(result) == 1
        assert entity.entity_id in result
        assert result[entity.entity_id] == [entity]
    
    def test_resolve_entities_similar(self):
        """Test entity resolution with similar entities."""
        # Use a lower threshold resolver for this test
        resolver = EntityResolver(similarity_threshold=0.6)
        
        entity1 = Entity(name="Apple Inc", type="Company", confidence=0.9)
        entity2 = Entity(name="Apple Corporation", type="Company", confidence=0.8)
        
        result = resolver.resolve_entities([entity1, entity2])
        
        # Should be grouped together with lower threshold
        assert len(result) == 1
        group = list(result.values())[0]
        assert len(group) == 2
        assert entity1 in group
        assert entity2 in group
    
    def test_resolve_entities_different_types(self):
        """Test entity resolution with different types."""
        entity1 = Entity(name="Apple", type="Company")
        entity2 = Entity(name="Apple", type="Product")
        
        result = self.resolver.resolve_entities([entity1, entity2])
        
        # Should be in separate groups due to different types
        assert len(result) == 2


class TestGraphNode:
    """Test cases for GraphNode."""
    
    def test_graph_node_creation(self):
        """Test GraphNode creation."""
        node = GraphNode(
            entity_id="ent:123",
            name="Apple Inc",
            type="Company",
            confidence=0.9
        )
        
        assert node.entity_id == "ent:123"
        assert node.name == "Apple Inc"
        assert node.type == "Company"
        assert node.confidence == 0.9
        assert len(node.aliases) == 0
        assert len(node.source_documents) == 0
    
    def test_add_alias(self):
        """Test adding aliases."""
        node = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company")
        node.add_alias("Apple Corporation")
        
        assert "apple corporation" in node.aliases
    
    def test_merge_with(self):
        """Test merging nodes."""
        node1 = GraphNode(
            entity_id="ent:123",
            name="Apple Inc",
            type="Company",
            confidence=0.8,
            description="Tech company"
        )
        
        node2 = GraphNode(
            entity_id="ent:456",
            name="Apple Corporation",
            type="Company",
            confidence=0.9,
            description="Technology corporation"
        )
        
        node1.merge_with(node2)
        
        assert node1.confidence == 0.9  # Takes maximum
        assert "apple corporation" in node1.aliases
        assert node1.description == "Technology corporation"  # Takes longer description


class TestGraphEdge:
    """Test cases for GraphEdge."""
    
    def test_graph_edge_creation(self):
        """Test GraphEdge creation."""
        edge = GraphEdge(
            source_id="ent:123",
            target_id="ent:456",
            relation="OWNS",
            confidence=0.8
        )
        
        assert edge.source_id == "ent:123"
        assert edge.target_id == "ent:456"
        assert edge.relation == "OWNS"
        assert edge.confidence == 0.8
    
    def test_get_edge_id(self):
        """Test edge ID generation."""
        edge = GraphEdge(
            source_id="ent:123",
            target_id="ent:456",
            relation="OWNS"
        )
        
        edge_id = edge.get_edge_id()
        assert edge_id.startswith("edge:")
        assert len(edge_id) > 5


class TestGraphBuilder:
    """Test cases for GraphBuilder."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = GraphBuilder(
            entity_similarity_threshold=0.85,
            enable_entity_resolution=True,
            min_confidence=0.5
        )
    
    def test_build_graph_empty(self):
        """Test building graph with empty inputs."""
        result = self.builder.build_graph([], [])
        
        assert len(result["nodes"]) == 0
        assert len(result["edges"]) == 0
        assert "statistics" in result
        assert "metadata" in result
    
    def test_build_graph_simple(self):
        """Test building simple graph."""
        entities = [
            Entity(name="Apple Inc", type="Company", confidence=0.9),
            Entity(name="Tim Cook", type="Person", confidence=0.8)
        ]
        
        relationships = [
            Relationship(source="Tim Cook", relation="WORKS_FOR", target="Apple Inc", confidence=0.9)
        ]
        
        result = self.builder.build_graph(entities, relationships, document_id="doc1")
        
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        
        # Check metadata
        assert result["metadata"]["document_id"] == "doc1"
        assert result["metadata"]["original_entities"] == 2
        assert result["metadata"]["original_relationships"] == 1
    
    def test_build_graph_with_resolution(self):
        """Test building graph with entity resolution."""
        # Create builder with lower threshold for this test
        builder = GraphBuilder(
            entity_similarity_threshold=0.6,  # Lower threshold to merge similar entities
            enable_entity_resolution=True,
            min_confidence=0.5
        )
        
        entities = [
            Entity(name="Apple Inc", type="Company", confidence=0.9),
            Entity(name="Apple Corporation", type="Company", confidence=0.8),
            Entity(name="Tim Cook", type="Person", confidence=0.8)
        ]
        
        relationships = [
            Relationship(source="Tim Cook", relation="WORKS_FOR", target="Apple Inc", confidence=0.9),
            Relationship(source="Tim Cook", relation="MANAGES", target="Apple Corporation", confidence=0.8)
        ]
        
        result = builder.build_graph(entities, relationships)
        
        # Should resolve Apple Inc and Apple Corporation to same entity
        assert len(result["nodes"]) == 2  # Apple (merged) + Tim Cook
        assert len(result["edges"]) == 2  # Both relationships should exist
    
    def test_build_graph_confidence_filtering(self):
        """Test confidence-based filtering."""
        entities = [
            Entity(name="Apple Inc", type="Company", confidence=0.9),
            Entity(name="Low Confidence Entity", type="Company", confidence=0.3)
        ]
        
        relationships = [
            Relationship(source="Apple Inc", relation="OWNS", target="Low Confidence Entity", confidence=0.2)
        ]
        
        result = self.builder.build_graph(entities, relationships)
        
        # Low confidence entity and relationship should be filtered out
        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 0
    
    def test_merge_graphs(self):
        """Test merging multiple graphs."""
        # Create first graph
        entities1 = [Entity(name="Apple Inc", type="Company")]
        relationships1 = []
        graph1 = self.builder.build_graph(entities1, relationships1, "doc1")
        
        # Create second graph
        entities2 = [Entity(name="Tim Cook", type="Person")]
        relationships2 = []
        graph2 = self.builder.build_graph(entities2, relationships2, "doc2")
        
        # Merge graphs
        merged = self.builder.merge_graphs([graph1, graph2])
        
        assert len(merged["nodes"]) == 2
        assert merged["metadata"]["merged_from"] == 2
    
    def test_optimize_graph(self):
        """Test graph optimization."""
        entities = [
            Entity(name="Apple Inc", type="Company", confidence=0.9),
            Entity(name="Low Confidence", type="Company", confidence=0.3)
        ]
        
        relationships = []
        
        graph = self.builder.build_graph(entities, relationships)
        optimized = self.builder.optimize_graph(graph)
        
        # Low confidence entity should be removed
        assert len(optimized["nodes"]) < len(graph["nodes"])
        assert optimized["metadata"]["optimized"] is True


if __name__ == "__main__":
    pytest.main([__file__])
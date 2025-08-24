"""
Tests for utility functions.
"""

import pytest
import time
from kge.utils import (
    PerformanceMonitor, ConfigValidator, DataExporter, 
    validate_graph_data, setup_logging
)
from kge.graph import GraphNode, GraphEdge


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = PerformanceMonitor(max_history=100)
    
    def test_start_end_operation(self):
        """Test starting and ending operations."""
        op_id = self.monitor.start_operation("test_op")
        assert op_id.startswith("test_op_")
        assert "test_op" in [op.name for op in self.monitor.current_operations.values()]
        
        time.sleep(0.01)  # Small delay
        self.monitor.end_operation("test_op", success=True)
        
        assert len(self.monitor.current_operations) == 0
        assert len(self.monitor.operations) == 1
        
        completed_op = self.monitor.operations[0]
        assert completed_op.name == "test_op"
        assert completed_op.success is True
        assert completed_op.duration > 0
    
    def test_operation_with_error(self):
        """Test operation with error."""
        self.monitor.start_operation("error_op")
        self.monitor.end_operation("error_op", success=False, error="Test error")
        
        completed_op = self.monitor.operations[0]
        assert completed_op.success is False
        assert completed_op.error == "Test error"
    
    def test_get_stats_empty(self):
        """Test getting stats with no operations."""
        stats = self.monitor.get_stats()
        assert stats == {}
    
    def test_get_stats_with_operations(self):
        """Test getting stats with operations."""
        # Add some operations
        for i in range(5):
            self.monitor.start_operation("test_op")
            time.sleep(0.001)
            self.monitor.end_operation("test_op", success=i < 4)  # 4 success, 1 failure
        
        stats = self.monitor.get_stats()
        
        assert stats["total_operations"] == 5
        assert stats["recent_operations"] == 5
        assert stats["success_rate"] == 0.8  # 4/5
        assert "test_op" in stats["operations_by_type"]
        
        op_stats = stats["operations_by_type"]["test_op"]
        assert op_stats["count"] == 5
        assert op_stats["success_rate"] == 0.8
        assert op_stats["avg_duration"] > 0
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        # Add operations
        self.monitor.start_operation("op1")
        time.sleep(0.001)
        self.monitor.end_operation("op1", success=True)
        
        self.monitor.start_operation("op2")
        time.sleep(0.001)
        self.monitor.end_operation("op2", success=False)
        
        summary = self.monitor.get_summary()
        
        assert summary["total_operations"] == 2
        assert summary["successful_operations"] == 1
        assert summary["failed_operations"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["total_time"] > 0
        assert summary["average_duration"] > 0
    
    def test_reset(self):
        """Test resetting monitor."""
        self.monitor.start_operation("test_op")
        self.monitor.end_operation("test_op")
        
        assert len(self.monitor.operations) == 1
        
        self.monitor.reset()
        
        assert len(self.monitor.operations) == 0
        assert len(self.monitor.current_operations) == 0
        assert len(self.monitor.stats) == 0


class TestConfigValidator:
    """Test cases for ConfigValidator."""
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            "model_path": "test/model",
            "chunk_size": 1000,
            "temperature": 0.5,
            "storage_backend": "memory"
        }
        
        errors = ConfigValidator.validate_kge_config(config)
        assert len(errors) == 0
    
    def test_validate_missing_required(self):
        """Test validation with missing required fields."""
        config = {
            "chunk_size": 1000
        }
        
        errors = ConfigValidator.validate_kge_config(config)
        assert any("Missing required field: model_path" in error for error in errors)
    
    def test_validate_invalid_numeric(self):
        """Test validation with invalid numeric values."""
        config = {
            "model_path": "test/model",
            "chunk_size": -100,  # Invalid
            "temperature": 5.0,  # Invalid
            "confidence_threshold": 1.5  # Invalid
        }
        
        errors = ConfigValidator.validate_kge_config(config)
        assert len(errors) >= 3  # At least 3 validation errors
    
    def test_validate_invalid_backend(self):
        """Test validation with invalid backend."""
        config = {
            "model_path": "test/model",
            "storage_backend": "invalid_backend"
        }
        
        errors = ConfigValidator.validate_kge_config(config)
        assert any("storage_backend must be one of" in error for error in errors)


class TestDataExporter:
    """Test cases for DataExporter."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.node1 = GraphNode(
            entity_id="ent:123",
            name="Apple Inc",
            type="Company",
            confidence=0.9
        )
        self.node1.aliases.add("apple corp")
        self.node1.source_documents.add("doc1")
        
        self.node2 = GraphNode(
            entity_id="ent:456",
            name="Tim Cook",
            type="Person",
            confidence=0.8
        )
        
        self.edge = GraphEdge(
            source_id="ent:456",
            target_id="ent:123",
            relation="WORKS_FOR",
            confidence=0.9
        )
        self.edge.source_documents.add("doc1")
        
        self.graph_data = {
            "nodes": [self.node1, self.node2],
            "edges": [self.edge],
            "statistics": {"node_count": 2, "edge_count": 1},
            "metadata": {"test": True}
        }
    
    def test_to_json_pretty(self):
        """Test JSON export with pretty formatting."""
        json_str = DataExporter.to_json(self.graph_data, pretty=True)
        
        assert '"name": "Apple Inc"' in json_str
        assert '"relation": "WORKS_FOR"' in json_str
        assert '"node_count": 2' in json_str
        
        # Check pretty formatting
        assert "\n" in json_str
        assert "  " in json_str  # Indentation
    
    def test_to_json_compact(self):
        """Test JSON export with compact formatting."""
        json_str = DataExporter.to_json(self.graph_data, pretty=False)
        
        assert '"name":"Apple Inc"' in json_str or '"name": "Apple Inc"' in json_str
        # Compact format should have less whitespace
        pretty_str = DataExporter.to_json(self.graph_data, pretty=True)
        assert len(json_str) < len(pretty_str)
    
    def test_to_cypher(self):
        """Test Cypher export."""
        cypher_str = DataExporter.to_cypher(self.graph_data)
        
        assert "CREATE" in cypher_str
        assert "Apple Inc" in cypher_str
        assert "Tim Cook" in cypher_str
        assert "WORKS_FOR" in cypher_str
        assert "MATCH" in cypher_str
    
    def test_to_graphml(self):
        """Test GraphML export."""
        graphml_str = DataExporter.to_graphml(self.graph_data)
        
        assert '<?xml version="1.0"' in graphml_str
        assert "<graphml" in graphml_str
        assert "<node" in graphml_str
        assert "<edge" in graphml_str
        assert "Apple Inc" in graphml_str
        assert "WORKS_FOR" in graphml_str


class TestValidateGraphData:
    """Test cases for validate_graph_data function."""
    
    def test_validate_valid_graph(self):
        """Test validation of valid graph data."""
        node = GraphNode(entity_id="ent:123", name="Apple Inc", type="Company")
        edge = GraphEdge(source_id="ent:123", target_id="ent:123", relation="SELF_REF")
        
        graph_data = {
            "nodes": [node],
            "edges": [edge]
        }
        
        errors = validate_graph_data(graph_data)
        assert len(errors) == 0
    
    def test_validate_invalid_structure(self):
        """Test validation of invalid structure."""
        errors = validate_graph_data("not a dict")
        assert "Graph data must be a dictionary" in errors
    
    def test_validate_missing_keys(self):
        """Test validation with missing keys."""
        graph_data = {"nodes": []}
        
        errors = validate_graph_data(graph_data)
        assert any("Missing required key: edges" in error for error in errors)
    
    def test_validate_invalid_nodes(self):
        """Test validation with invalid nodes."""
        graph_data = {
            "nodes": "not a list",
            "edges": []
        }
        
        errors = validate_graph_data(graph_data)
        assert any("Nodes must be a list" in error for error in errors)
    
    def test_validate_duplicate_node_ids(self):
        """Test validation with duplicate node IDs."""
        node1 = GraphNode(entity_id="ent:123", name="Node 1", type="Test")
        node2 = GraphNode(entity_id="ent:123", name="Node 2", type="Test")
        
        graph_data = {
            "nodes": [node1, node2],
            "edges": []
        }
        
        errors = validate_graph_data(graph_data)
        assert any("Duplicate node ID" in error for error in errors)
    
    def test_validate_invalid_edge_references(self):
        """Test validation with invalid edge references."""
        node = GraphNode(entity_id="ent:123", name="Node", type="Test")
        edge = GraphEdge(source_id="ent:123", target_id="ent:999", relation="TEST")
        
        graph_data = {
            "nodes": [node],
            "edges": [edge]
        }
        
        errors = validate_graph_data(graph_data)
        assert any("references unknown target node" in error for error in errors)


class TestSetupLogging:
    """Test cases for setup_logging function."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        # This is hard to test without affecting global state
        # Just ensure it doesn't raise an exception
        setup_logging()
    
    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        setup_logging(level="DEBUG")
    
    def test_setup_logging_custom_format(self):
        """Test logging setup with custom format."""
        setup_logging(format_string="%(levelname)s: %(message)s")


if __name__ == "__main__":
    pytest.main([__file__])
"""
Utility functions and classes for the KGE system.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import sys


def setup_logging(level: str = "INFO", format_string: Optional[str] = None):
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


@dataclass
class OperationStats:
    """Statistics for a single operation."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True, error: Optional[str] = None):
        """Mark operation as finished."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize performance monitor.
        
        Args:
            max_history: Maximum number of operations to keep in history
        """
        self.max_history = max_history
        self.operations = deque(maxlen=max_history)
        self.current_operations = {}
        self.stats = defaultdict(list)
    
    def start_operation(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an operation.
        
        Args:
            name: Operation name
            metadata: Optional metadata
            
        Returns:
            Operation ID
        """
        op_id = f"{name}_{time.time()}"
        operation = OperationStats(
            name=name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self.current_operations[op_id] = operation
        return op_id
    
    def end_operation(self, name: str, success: bool = True, error: Optional[str] = None):
        """End tracking an operation.
        
        Args:
            name: Operation name
            success: Whether operation succeeded
            error: Error message if failed
        """
        # Find the most recent operation with this name
        op_id = None
        for oid, op in self.current_operations.items():
            if op.name == name:
                op_id = oid
                break
        
        if op_id:
            operation = self.current_operations.pop(op_id)
            operation.finish(success=success, error=error)
            
            self.operations.append(operation)
            self.stats[name].append(operation.duration)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics.
        
        Returns:
            Dictionary containing performance stats
        """
        if not self.operations:
            return {}
        
        recent_ops = list(self.operations)[-100:]  # Last 100 operations
        
        stats = {
            "total_operations": len(self.operations),
            "recent_operations": len(recent_ops),
            "success_rate": sum(1 for op in recent_ops if op.success) / len(recent_ops),
            "operations_by_type": {},
            "average_durations": {}
        }
        
        # Group by operation type
        by_type = defaultdict(list)
        for op in recent_ops:
            by_type[op.name].append(op)
        
        for op_name, ops in by_type.items():
            durations = [op.duration for op in ops if op.duration is not None]
            
            stats["operations_by_type"][op_name] = {
                "count": len(ops),
                "success_rate": sum(1 for op in ops if op.success) / len(ops),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0
            }
        
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Summary statistics
        """
        if not self.operations:
            return {"message": "No operations recorded"}
        
        all_ops = list(self.operations)
        successful_ops = [op for op in all_ops if op.success]
        failed_ops = [op for op in all_ops if not op.success]
        
        durations = [op.duration for op in all_ops if op.duration is not None]
        
        summary = {
            "total_operations": len(all_ops),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(all_ops) if all_ops else 0,
            "total_time": sum(durations) if durations else 0,
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0
        }
        
        return summary
    
    def reset(self):
        """Reset all statistics."""
        self.operations.clear()
        self.current_operations.clear()
        self.stats.clear()


class ConfigValidator:
    """Validate configuration objects."""
    
    @staticmethod
    def validate_kge_config(config: Dict[str, Any]) -> List[str]:
        """Validate KGE configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Required fields
        required_fields = ["model_path"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Numeric validations
        numeric_fields = {
            "chunk_size": (100, 10000),
            "chunk_overlap": (0, 1000),
            "batch_size": (1, 100),
            "max_tokens": (50, 2048),
            "temperature": (0.0, 2.0),
            "confidence_threshold": (0.0, 1.0),
            "entity_similarity_threshold": (0.0, 1.0),
            "gpu_memory_utilization": (0.1, 1.0)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{field} must be numeric")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{field} must be between {min_val} and {max_val}")
        
        # String validations
        string_fields = ["storage_backend", "quantization"]
        for field in string_fields:
            if field in config and not isinstance(config[field], str):
                errors.append(f"{field} must be a string")
        
        # Backend-specific validations
        if "storage_backend" in config:
            valid_backends = ["neo4j", "kuzu", "memory"]
            if config["storage_backend"] not in valid_backends:
                errors.append(f"storage_backend must be one of: {valid_backends}")
        
        return errors


class DataExporter:
    """Export graph data to various formats."""
    
    @staticmethod
    def to_json(graph_data: Dict[str, Any], pretty: bool = True) -> str:
        """Export graph data to JSON.
        
        Args:
            graph_data: Graph data dictionary
            pretty: Whether to format JSON prettily
            
        Returns:
            JSON string
        """
        # Convert objects to dictionaries
        export_data = {
            "nodes": [],
            "edges": [],
            "statistics": graph_data.get("statistics", {}),
            "metadata": graph_data.get("metadata", {})
        }
        
        # Convert nodes
        for node in graph_data.get("nodes", []):
            if hasattr(node, '__dict__'):
                node_dict = node.__dict__.copy()
                # Convert sets to lists for JSON serialization
                if "aliases" in node_dict:
                    node_dict["aliases"] = list(node_dict["aliases"])
                if "source_documents" in node_dict:
                    node_dict["source_documents"] = list(node_dict["source_documents"])
                export_data["nodes"].append(node_dict)
            else:
                export_data["nodes"].append(node)
        
        # Convert edges
        for edge in graph_data.get("edges", []):
            if hasattr(edge, '__dict__'):
                edge_dict = edge.__dict__.copy()
                # Convert sets to lists for JSON serialization
                if "source_documents" in edge_dict:
                    edge_dict["source_documents"] = list(edge_dict["source_documents"])
                export_data["edges"].append(edge_dict)
            else:
                export_data["edges"].append(edge)
        
        if pretty:
            return json.dumps(export_data, indent=2, default=str)
        else:
            return json.dumps(export_data, default=str)
    
    @staticmethod
    def to_cypher(graph_data: Dict[str, Any]) -> str:
        """Export graph data to Cypher statements.
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            Cypher statements as string
        """
        statements = []
        
        # Create nodes
        for node in graph_data.get("nodes", []):
            if hasattr(node, '__dict__'):
                node_dict = node.__dict__
            else:
                node_dict = node
            
            properties = []
            for key, value in node_dict.items():
                if key == "entity_id":
                    continue
                if isinstance(value, str):
                    properties.append(f"{key}: '{value.replace("'", "\\'")}'")
                elif isinstance(value, (int, float)):
                    properties.append(f"{key}: {value}")
                elif isinstance(value, (list, set)):
                    list_str = str(list(value)).replace("'", '"')
                    properties.append(f"{key}: {list_str}")
            
            props_str = ", ".join(properties)
            statement = f"CREATE (n{node_dict['entity_id'].replace(':', '_')}:Entity {{entity_id: '{node_dict['entity_id']}', {props_str}}});"
            statements.append(statement)
        
        # Create relationships
        for edge in graph_data.get("edges", []):
            if hasattr(edge, '__dict__'):
                edge_dict = edge.__dict__
            else:
                edge_dict = edge
            
            source_var = f"n{edge_dict['source_id'].replace(':', '_')}"
            target_var = f"n{edge_dict['target_id'].replace(':', '_')}"
            
            properties = []
            for key, value in edge_dict.items():
                if key in ["source_id", "target_id", "relation"]:
                    continue
                if isinstance(value, str):
                    properties.append(f"{key}: '{value.replace("'", "\\'")}'")
                elif isinstance(value, (int, float)):
                    properties.append(f"{key}: {value}")
            
            props_str = ", ".join(properties)
            rel_props = f" {{{props_str}}}" if props_str else ""
            
            statement = f"MATCH ({source_var}), ({target_var}) CREATE ({source_var})-[:{edge_dict['relation']}{rel_props}]->({target_var});"
            statements.append(statement)
        
        return "\n".join(statements)
    
    @staticmethod
    def to_graphml(graph_data: Dict[str, Any]) -> str:
        """Export graph data to GraphML format.
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            GraphML XML string
        """
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
            '         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
            '',
            '  <!-- Node attributes -->',
            '  <key id="name" for="node" attr.name="name" attr.type="string"/>',
            '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>',
            '',
            '  <!-- Edge attributes -->',
            '  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>',
            '  <key id="edge_confidence" for="edge" attr.name="confidence" attr.type="double"/>',
            '',
            '  <graph id="KnowledgeGraph" edgedefault="directed">'
        ]
        
        # Add nodes
        for node in graph_data.get("nodes", []):
            if hasattr(node, '__dict__'):
                node_dict = node.__dict__
            else:
                node_dict = node
            
            lines.append(f'    <node id="{node_dict["entity_id"]}">')
            lines.append(f'      <data key="name">{node_dict.get("name", "")}</data>')
            lines.append(f'      <data key="type">{node_dict.get("type", "")}</data>')
            lines.append(f'      <data key="confidence">{node_dict.get("confidence", 1.0)}</data>')
            lines.append('    </node>')
        
        # Add edges
        edge_id = 0
        for edge in graph_data.get("edges", []):
            if hasattr(edge, '__dict__'):
                edge_dict = edge.__dict__
            else:
                edge_dict = edge
            
            lines.append(f'    <edge id="e{edge_id}" source="{edge_dict["source_id"]}" target="{edge_dict["target_id"]}">')
            lines.append(f'      <data key="relation">{edge_dict.get("relation", "")}</data>')
            lines.append(f'      <data key="edge_confidence">{edge_dict.get("confidence", 1.0)}</data>')
            lines.append('    </edge>')
            edge_id += 1
        
        lines.extend([
            '  </graph>',
            '</graphml>'
        ])
        
        return '\n'.join(lines)


def validate_graph_data(graph_data: Dict[str, Any]) -> List[str]:
    """Validate graph data structure.
    
    Args:
        graph_data: Graph data to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if not isinstance(graph_data, dict):
        errors.append("Graph data must be a dictionary")
        return errors
    
    # Check required keys
    required_keys = ["nodes", "edges"]
    for key in required_keys:
        if key not in graph_data:
            errors.append(f"Missing required key: {key}")
    
    # Validate nodes
    nodes = graph_data.get("nodes", [])
    if not isinstance(nodes, list):
        errors.append("Nodes must be a list")
    else:
        node_ids = set()
        for i, node in enumerate(nodes):
            if hasattr(node, 'entity_id'):
                node_id = node.entity_id
            elif isinstance(node, dict) and 'entity_id' in node:
                node_id = node['entity_id']
            else:
                errors.append(f"Node {i} missing entity_id")
                continue
            
            if node_id in node_ids:
                errors.append(f"Duplicate node ID: {node_id}")
            node_ids.add(node_id)
    
    # Validate edges
    edges = graph_data.get("edges", [])
    if not isinstance(edges, list):
        errors.append("Edges must be a list")
    else:
        for i, edge in enumerate(edges):
            if hasattr(edge, 'source_id') and hasattr(edge, 'target_id'):
                source_id = edge.source_id
                target_id = edge.target_id
            elif isinstance(edge, dict):
                source_id = edge.get('source_id')
                target_id = edge.get('target_id')
            else:
                errors.append(f"Edge {i} missing source_id or target_id")
                continue
            
            if source_id not in node_ids:
                errors.append(f"Edge {i} references unknown source node: {source_id}")
            if target_id not in node_ids:
                errors.append(f"Edge {i} references unknown target node: {target_id}")
    
    return errors
"""
Graph database integration and storage layer.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import asdict
import json

from .graph import GraphNode, GraphEdge

logger = logging.getLogger(__name__)


class GraphStorageBackend(ABC):
    """Abstract base class for graph storage backends."""
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to the storage backend."""
        pass
    
    @abstractmethod
    def store_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store graph data."""
        pass
    
    @abstractmethod
    def query_nodes(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query nodes with optional filters."""
        pass
    
    @abstractmethod
    def query_edges(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query edges with optional filters."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources."""
        pass


class Neo4jBackend(GraphStorageBackend):
    """Neo4j graph database backend."""
    
    def __init__(self):
        self.driver = None
        self.connected = False
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to Neo4j database."""
        try:
            from neo4j import GraphDatabase
            
            uri = config.get("uri", "bolt://localhost:7687")
            username = config.get("username", "neo4j")
            password = config.get("password", "password")
            
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            self.connected = True
            logger.info(f"Connected to Neo4j at {uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.connected = False
            return False
    
    def store_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store graph data in Neo4j."""
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")
        
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        nodes_created = 0
        edges_created = 0
        
        try:
            with self.driver.session() as session:
                # Create nodes
                for node in nodes:
                    node_data = {
                        "entity_id": node.entity_id,
                        "name": node.name,
                        "type": node.type,
                        "description": node.description or "",
                        "confidence": node.confidence,
                        "aliases": list(node.aliases),
                        "properties": node.properties,
                        "source_documents": list(node.source_documents)
                    }
                    
                    query = """
                    MERGE (n:Entity {entity_id: $entity_id})
                    SET n.name = $name,
                        n.type = $type,
                        n.description = $description,
                        n.confidence = $confidence,
                        n.aliases = $aliases,
                        n.properties = $properties,
                        n.source_documents = $source_documents
                    """
                    
                    session.run(query, **node_data)
                    nodes_created += 1
                
                # Create edges
                for edge in edges:
                    edge_data = {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "relation": edge.relation,
                        "confidence": edge.confidence,
                        "properties": edge.properties,
                        "source_documents": list(edge.source_documents)
                    }
                    
                    query = """
                    MATCH (source:Entity {entity_id: $source_id})
                    MATCH (target:Entity {entity_id: $target_id})
                    MERGE (source)-[r:RELATION {type: $relation}]->(target)
                    SET r.confidence = $confidence,
                        r.properties = $properties,
                        r.source_documents = $source_documents
                    """
                    
                    session.run(query, **edge_data)
                    edges_created += 1
            
            result = {
                "success": True,
                "nodes_created": nodes_created,
                "edges_created": edges_created,
                "backend": "neo4j"
            }
            
            logger.info(f"Stored graph in Neo4j: {nodes_created} nodes, {edges_created} edges")
            return result
            
        except Exception as e:
            logger.error(f"Error storing graph in Neo4j: {e}")
            return {"success": False, "error": str(e), "backend": "neo4j"}
    
    def query_nodes(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query nodes from Neo4j."""
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")
        
        try:
            with self.driver.session() as session:
                query = "MATCH (n:Entity) RETURN n"
                
                if filters:
                    conditions = []
                    params = {}
                    
                    if "type" in filters:
                        conditions.append("n.type = $type")
                        params["type"] = filters["type"]
                    
                    if "min_confidence" in filters:
                        conditions.append("n.confidence >= $min_confidence")
                        params["min_confidence"] = filters["min_confidence"]
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                    
                    result = session.run(query, **params)
                else:
                    result = session.run(query)
                
                nodes = []
                for record in result:
                    node_data = dict(record["n"])
                    nodes.append(node_data)
                
                return nodes
                
        except Exception as e:
            logger.error(f"Error querying nodes from Neo4j: {e}")
            return []
    
    def query_edges(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query edges from Neo4j."""
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")
        
        try:
            with self.driver.session() as session:
                query = "MATCH (source)-[r:RELATION]->(target) RETURN source.entity_id as source_id, target.entity_id as target_id, r"
                
                if filters:
                    conditions = []
                    params = {}
                    
                    if "relation" in filters:
                        conditions.append("r.type = $relation")
                        params["relation"] = filters["relation"]
                    
                    if "min_confidence" in filters:
                        conditions.append("r.confidence >= $min_confidence")
                        params["min_confidence"] = filters["min_confidence"]
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                    
                    result = session.run(query, **params)
                else:
                    result = session.run(query)
                
                edges = []
                for record in result:
                    edge_data = {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        **dict(record["r"])
                    }
                    edges.append(edge_data)
                
                return edges
                
        except Exception as e:
            logger.error(f"Error querying edges from Neo4j: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Neo4j statistics."""
        if not self.connected:
            return {"connected": False}
        
        try:
            with self.driver.session() as session:
                # Count nodes
                node_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
                node_count = node_result.single()["count"]
                
                # Count edges
                edge_result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count")
                edge_count = edge_result.single()["count"]
                
                # Node types
                type_result = session.run("MATCH (n:Entity) RETURN n.type as type, count(n) as count")
                node_types = {record["type"]: record["count"] for record in type_result}
                
                return {
                    "connected": True,
                    "backend": "neo4j",
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "node_types": node_types
                }
                
        except Exception as e:
            logger.error(f"Error getting Neo4j statistics: {e}")
            return {"connected": False, "error": str(e)}
    
    def cleanup(self):
        """Cleanup Neo4j resources."""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Neo4j connection closed")


class KuzuBackend(GraphStorageBackend):
    """Kuzu graph database backend."""
    
    def __init__(self):
        self.db = None
        self.conn = None
        self.connected = False
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to Kuzu database."""
        try:
            import kuzu
            
            db_path = config.get("db_path", "./kuzu_db")
            
            self.db = kuzu.Database(db_path)
            self.conn = kuzu.Connection(self.db)
            
            # Create schema if not exists
            self._create_schema()
            
            self.connected = True
            logger.info(f"Connected to Kuzu at {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kuzu: {e}")
            self.connected = False
            return False
    
    def _create_schema(self):
        """Create Kuzu schema."""
        try:
            # Create Entity table
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    entity_id STRING,
                    name STRING,
                    type STRING,
                    description STRING,
                    confidence DOUBLE,
                    aliases STRING[],
                    properties STRING,
                    source_documents STRING[],
                    PRIMARY KEY (entity_id)
                )
            """)
            
            # Create Relation table
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATION(
                    FROM Entity TO Entity,
                    relation_type STRING,
                    confidence DOUBLE,
                    properties STRING,
                    source_documents STRING[]
                )
            """)
            
            logger.debug("Kuzu schema created")
            
        except Exception as e:
            logger.warning(f"Schema creation warning (may already exist): {e}")
    
    def store_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store graph data in Kuzu."""
        if not self.connected:
            raise RuntimeError("Not connected to Kuzu")
        
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        nodes_created = 0
        edges_created = 0
        
        try:
            # Store nodes
            for node in nodes:
                query = """
                MERGE (n:Entity {entity_id: $entity_id})
                SET n.name = $name,
                    n.type = $type,
                    n.description = $description,
                    n.confidence = $confidence,
                    n.aliases = $aliases,
                    n.properties = $properties,
                    n.source_documents = $source_documents
                """
                
                params = {
                    "entity_id": node.entity_id,
                    "name": node.name,
                    "type": node.type,
                    "description": node.description or "",
                    "confidence": node.confidence,
                    "aliases": list(node.aliases),
                    "properties": json.dumps(node.properties),
                    "source_documents": list(node.source_documents)
                }
                
                self.conn.execute(query, params)
                nodes_created += 1
            
            # Store edges
            for edge in edges:
                query = """
                MATCH (source:Entity {entity_id: $source_id})
                MATCH (target:Entity {entity_id: $target_id})
                CREATE (source)-[r:RELATION {
                    relation_type: $relation_type,
                    confidence: $confidence,
                    properties: $properties,
                    source_documents: $source_documents
                }]->(target)
                """
                
                params = {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation,
                    "confidence": edge.confidence,
                    "properties": json.dumps(edge.properties),
                    "source_documents": list(edge.source_documents)
                }
                
                self.conn.execute(query, params)
                edges_created += 1
            
            result = {
                "success": True,
                "nodes_created": nodes_created,
                "edges_created": edges_created,
                "backend": "kuzu"
            }
            
            logger.info(f"Stored graph in Kuzu: {nodes_created} nodes, {edges_created} edges")
            return result
            
        except Exception as e:
            logger.error(f"Error storing graph in Kuzu: {e}")
            return {"success": False, "error": str(e), "backend": "kuzu"}
    
    def query_nodes(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query nodes from Kuzu."""
        if not self.connected:
            raise RuntimeError("Not connected to Kuzu")
        
        try:
            query = "MATCH (n:Entity) RETURN n.*"
            
            if filters:
                conditions = []
                if "type" in filters:
                    conditions.append(f"n.type = '{filters['type']}'")
                if "min_confidence" in filters:
                    conditions.append(f"n.confidence >= {filters['min_confidence']}")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            result = self.conn.execute(query)
            nodes = []
            
            while result.hasNext():
                record = result.getNext()
                nodes.append(record)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error querying nodes from Kuzu: {e}")
            return []
    
    def query_edges(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query edges from Kuzu."""
        if not self.connected:
            raise RuntimeError("Not connected to Kuzu")
        
        try:
            query = "MATCH (source)-[r:RELATION]->(target) RETURN source.entity_id, target.entity_id, r.*"
            
            if filters:
                conditions = []
                if "relation" in filters:
                    conditions.append(f"r.relation_type = '{filters['relation']}'")
                if "min_confidence" in filters:
                    conditions.append(f"r.confidence >= {filters['min_confidence']}")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            result = self.conn.execute(query)
            edges = []
            
            while result.hasNext():
                record = result.getNext()
                edges.append(record)
            
            return edges
            
        except Exception as e:
            logger.error(f"Error querying edges from Kuzu: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Kuzu statistics."""
        if not self.connected:
            return {"connected": False}
        
        try:
            # Count nodes
            node_result = self.conn.execute("MATCH (n:Entity) RETURN count(*)")
            node_count = node_result.getNext()[0] if node_result.hasNext() else 0
            
            # Count edges
            edge_result = self.conn.execute("MATCH ()-[r:RELATION]->() RETURN count(*)")
            edge_count = edge_result.getNext()[0] if edge_result.hasNext() else 0
            
            return {
                "connected": True,
                "backend": "kuzu",
                "node_count": node_count,
                "edge_count": edge_count
            }
            
        except Exception as e:
            logger.error(f"Error getting Kuzu statistics: {e}")
            return {"connected": False, "error": str(e)}
    
    def cleanup(self):
        """Cleanup Kuzu resources."""
        if self.conn:
            self.conn.close()
        if self.db:
            self.db.close()
        self.connected = False
        logger.info("Kuzu connection closed")


class MemoryBackend(GraphStorageBackend):
    """In-memory graph storage backend for testing and development."""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.connected = False
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to memory backend (always succeeds)."""
        self.connected = True
        logger.info("Connected to memory backend")
        return True
    
    def store_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store graph data in memory."""
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        nodes_created = 0
        edges_created = 0
        
        # Store nodes
        for node in nodes:
            self.nodes[node.entity_id] = {
                "entity_id": node.entity_id,
                "name": node.name,
                "type": node.type,
                "description": node.description,
                "confidence": node.confidence,
                "aliases": list(node.aliases),
                "properties": node.properties,
                "source_documents": list(node.source_documents)
            }
            nodes_created += 1
        
        # Store edges
        for edge in edges:
            edge_data = {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation,
                "confidence": edge.confidence,
                "properties": edge.properties,
                "source_documents": list(edge.source_documents)
            }
            self.edges.append(edge_data)
            edges_created += 1
        
        result = {
            "success": True,
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "backend": "memory"
        }
        
        logger.info(f"Stored graph in memory: {nodes_created} nodes, {edges_created} edges")
        return result
    
    def query_nodes(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query nodes from memory."""
        nodes = list(self.nodes.values())
        
        if filters:
            if "type" in filters:
                nodes = [n for n in nodes if n["type"] == filters["type"]]
            if "min_confidence" in filters:
                nodes = [n for n in nodes if n["confidence"] >= filters["min_confidence"]]
        
        return nodes
    
    def query_edges(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query edges from memory."""
        edges = self.edges.copy()
        
        if filters:
            if "relation" in filters:
                edges = [e for e in edges if e["relation"] == filters["relation"]]
            if "min_confidence" in filters:
                edges = [e for e in edges if e["confidence"] >= filters["min_confidence"]]
        
        return edges
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory backend statistics."""
        node_types = {}
        for node in self.nodes.values():
            node_type = node["type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            "connected": True,
            "backend": "memory",
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": node_types
        }
    
    def cleanup(self):
        """Cleanup memory backend."""
        self.nodes.clear()
        self.edges.clear()
        self.connected = False
        logger.info("Memory backend cleared")


class GraphStorage:
    """Main graph storage interface."""
    
    def __init__(self, backend: str = "memory", config: Optional[Dict[str, Any]] = None):
        """Initialize graph storage.
        
        Args:
            backend: Storage backend type ("neo4j", "kuzu", or "memory")
            config: Backend-specific configuration
        """
        self.backend_name = backend
        self.config = config or {}
        
        # Initialize backend
        if backend == "neo4j":
            self.backend = Neo4jBackend()
        elif backend == "kuzu":
            self.backend = KuzuBackend()
        elif backend == "memory":
            self.backend = MemoryBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        # Connect to backend
        if not self.backend.connect(self.config):
            raise RuntimeError(f"Failed to connect to {backend} backend")
        
        logger.info(f"GraphStorage initialized with {backend} backend")
    
    def store_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store graph data."""
        return self.backend.store_graph(graph_data)
    
    def query_nodes(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query nodes with optional filters."""
        return self.backend.query_nodes(filters)
    
    def query_edges(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query edges with optional filters."""
        return self.backend.query_edges(filters)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.backend.get_statistics()
    
    def cleanup(self):
        """Cleanup storage resources."""
        self.backend.cleanup()
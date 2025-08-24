"""
Knowledge graph construction and optimization.
"""

import logging
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import difflib

from .entities import Entity, Relationship

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    entity_id: str
    name: str
    type: str
    description: Optional[str] = None
    confidence: float = 1.0
    aliases: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    source_documents: Set[str] = field(default_factory=set)
    
    def add_alias(self, alias: str):
        """Add an alias for this entity."""
        self.aliases.add(alias.strip().lower())
    
    def merge_with(self, other: 'GraphNode'):
        """Merge another node into this one."""
        # Combine aliases
        self.aliases.update(other.aliases)
        self.aliases.add(other.name.lower())
        
        # Update confidence (take maximum)
        self.confidence = max(self.confidence, other.confidence)
        
        # Merge properties
        self.properties.update(other.properties)
        
        # Combine source documents
        self.source_documents.update(other.source_documents)
        
        # Update description if other has a better one
        if other.description and (not self.description or len(other.description) > len(self.description)):
            self.description = other.description


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation: str
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source_documents: Set[str] = field(default_factory=set)
    
    def get_edge_id(self) -> str:
        """Generate unique edge ID."""
        key = f"{self.source_id}|{self.relation}|{self.target_id}"
        return "edge:" + hashlib.sha1(key.encode()).hexdigest()[:12]


class EntityResolver:
    """Handles entity resolution and deduplication."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize entity resolver.
        
        Args:
            similarity_threshold: Minimum similarity for entity matching
        """
        self.similarity_threshold = similarity_threshold
        
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two entity names.
        
        Args:
            name1: First entity name
            name2: Second entity name
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize names
        name1 = name1.strip().lower()
        name2 = name2.strip().lower()
        
        if name1 == name2:
            return 1.0
        
        # Use sequence matcher for similarity
        similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
        
        # Boost similarity for common patterns
        if self._is_abbreviation_match(name1, name2):
            similarity = max(similarity, 0.9)
        
        if self._is_partial_match(name1, name2):
            similarity = max(similarity, 0.8)
        
        return similarity
    
    def _is_abbreviation_match(self, name1: str, name2: str) -> bool:
        """Check if one name is an abbreviation of another."""
        short, long = (name1, name2) if len(name1) < len(name2) else (name2, name1)
        
        if len(short) < 2:
            return False
        
        # Check if short name matches initials of long name
        words = long.split()
        if len(words) >= len(short):
            initials = ''.join(word[0] for word in words if word)
            return short.replace('.', '').replace(' ', '') == initials
        
        return False
    
    def _is_partial_match(self, name1: str, name2: str) -> bool:
        """Check if names have significant word overlap."""
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Jaccard similarity
        return len(intersection) / len(union) > 0.6
    
    def resolve_entities(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """Group entities by similarity.
        
        Args:
            entities: List of entities to resolve
            
        Returns:
            Dictionary mapping canonical entity ID to list of similar entities
        """
        if not entities:
            return {}
        
        # Group entities by type first
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.type].append(entity)
        
        resolved_groups = {}
        
        for entity_type, type_entities in entities_by_type.items():
            # Find similar entities within the same type
            processed = set()
            
            for i, entity in enumerate(type_entities):
                if i in processed:
                    continue
                
                # Start a new group with this entity
                group = [entity]
                processed.add(i)
                
                # Find similar entities
                for j, other_entity in enumerate(type_entities[i+1:], i+1):
                    if j in processed:
                        continue
                    
                    similarity = self.calculate_similarity(entity.name, other_entity.name)
                    if similarity >= self.similarity_threshold:
                        group.append(other_entity)
                        processed.add(j)
                
                # Use the entity with highest confidence as canonical
                canonical_entity = max(group, key=lambda e: e.confidence)
                resolved_groups[canonical_entity.entity_id] = group
        
        logger.debug(f"Resolved {len(entities)} entities into {len(resolved_groups)} groups")
        return resolved_groups


class GraphBuilder:
    """Builds and optimizes knowledge graphs from extracted information."""
    
    def __init__(self, 
                 entity_similarity_threshold: float = 0.85,
                 enable_entity_resolution: bool = True,
                 min_confidence: float = 0.5):
        """Initialize graph builder.
        
        Args:
            entity_similarity_threshold: Threshold for entity similarity matching
            enable_entity_resolution: Whether to perform entity resolution
            min_confidence: Minimum confidence threshold for including entities/relationships
        """
        self.entity_similarity_threshold = entity_similarity_threshold
        self.enable_entity_resolution = enable_entity_resolution
        self.min_confidence = min_confidence
        
        self.entity_resolver = EntityResolver(entity_similarity_threshold)
        
        logger.info(f"GraphBuilder initialized with similarity_threshold={entity_similarity_threshold}")
    
    def build_graph(self, 
                   entities: List[Entity], 
                   relationships: List[Relationship],
                   document_id: Optional[str] = None) -> Dict[str, Any]:
        """Build knowledge graph from extracted entities and relationships.
        
        Args:
            entities: List of extracted entities
            relationships: List of extracted relationships
            document_id: Optional document identifier
            
        Returns:
            Dictionary containing graph data
        """
        logger.info(f"Building graph from {len(entities)} entities and {len(relationships)} relationships")
        
        # Filter by confidence
        entities = [e for e in entities if e.confidence >= self.min_confidence]
        relationships = [r for r in relationships if r.confidence >= self.min_confidence]
        
        # Resolve entities if enabled
        if self.enable_entity_resolution:
            entity_groups = self.entity_resolver.resolve_entities(entities)
        else:
            entity_groups = {e.entity_id: [e] for e in entities}
        
        # Create graph nodes
        nodes = {}
        entity_name_to_id = {}
        
        for canonical_id, entity_group in entity_groups.items():
            canonical_entity = max(entity_group, key=lambda e: e.confidence)
            
            node = GraphNode(
                entity_id=canonical_id,
                name=canonical_entity.name,
                type=canonical_entity.type,
                description=canonical_entity.description,
                confidence=canonical_entity.confidence
            )
            
            # Add document source
            if document_id:
                node.source_documents.add(document_id)
            
            # Merge information from all entities in the group
            for entity in entity_group:
                if entity != canonical_entity:
                    node.merge_with(GraphNode(
                        entity_id=entity.entity_id,
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        confidence=entity.confidence
                    ))
                
                # Map all entity names to this canonical ID
                entity_name_to_id[entity.name.lower()] = canonical_id
                for alias in node.aliases:
                    entity_name_to_id[alias] = canonical_id
            
            nodes[canonical_id] = node
        
        # Create graph edges
        edges = []
        edge_dedup = set()
        
        for relationship in relationships:
            # Map relationship entity names to canonical IDs
            source_id = entity_name_to_id.get(relationship.source.lower())
            target_id = entity_name_to_id.get(relationship.target.lower())
            
            if not source_id or not target_id:
                logger.debug(f"Skipping relationship with unmapped entities: {relationship.source} -> {relationship.target}")
                continue
            
            if source_id == target_id:
                logger.debug(f"Skipping self-relationship: {relationship.source}")
                continue
            
            # Create edge
            edge = GraphEdge(
                source_id=source_id,
                target_id=target_id,
                relation=relationship.relation,
                confidence=relationship.confidence
            )
            
            if document_id:
                edge.source_documents.add(document_id)
            
            # Deduplicate edges
            edge_key = (source_id, relationship.relation, target_id)
            if edge_key not in edge_dedup:
                edges.append(edge)
                edge_dedup.add(edge_key)
            else:
                # Update confidence of existing edge
                for existing_edge in edges:
                    if (existing_edge.source_id == source_id and 
                        existing_edge.target_id == target_id and 
                        existing_edge.relation == relationship.relation):
                        existing_edge.confidence = max(existing_edge.confidence, relationship.confidence)
                        if document_id:
                            existing_edge.source_documents.add(document_id)
                        break
        
        # Calculate graph statistics
        stats = self._calculate_graph_stats(nodes, edges)
        
        graph_data = {
            "nodes": list(nodes.values()),
            "edges": edges,
            "statistics": stats,
            "metadata": {
                "document_id": document_id,
                "original_entities": len(entities),
                "original_relationships": len(relationships),
                "resolved_entities": len(nodes),
                "final_relationships": len(edges),
                "entity_resolution_enabled": self.enable_entity_resolution
            }
        }
        
        logger.info(f"Graph built: {len(nodes)} nodes, {len(edges)} edges")
        return graph_data
    
    def _calculate_graph_stats(self, nodes: Dict[str, GraphNode], edges: List[GraphEdge]) -> Dict[str, Any]:
        """Calculate graph statistics.
        
        Args:
            nodes: Dictionary of graph nodes
            edges: List of graph edges
            
        Returns:
            Dictionary containing graph statistics
        """
        if not nodes:
            return {}
        
        # Node statistics
        node_types = Counter(node.type for node in nodes.values())
        confidence_scores = [node.confidence for node in nodes.values()]
        
        # Edge statistics
        relation_types = Counter(edge.relation for edge in edges)
        edge_confidences = [edge.confidence for edge in edges]
        
        # Degree statistics
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        
        for edge in edges:
            out_degrees[edge.source_id] += 1
            in_degrees[edge.target_id] += 1
        
        degrees = [in_degrees[node_id] + out_degrees[node_id] for node_id in nodes.keys()]
        
        stats = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": dict(node_types),
            "relation_types": dict(relation_types),
            "avg_node_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "avg_edge_confidence": sum(edge_confidences) / len(edge_confidences) if edge_confidences else 0,
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "density": len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
        }
        
        return stats
    
    def merge_graphs(self, graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple graphs into one.
        
        Args:
            graphs: List of graph dictionaries to merge
            
        Returns:
            Merged graph dictionary
        """
        if not graphs:
            return {"nodes": [], "edges": [], "statistics": {}, "metadata": {}}
        
        if len(graphs) == 1:
            return graphs[0]
        
        logger.info(f"Merging {len(graphs)} graphs")
        
        # Collect all entities and relationships
        all_entities = []
        all_relationships = []
        all_document_ids = set()
        
        for graph in graphs:
            # Convert nodes back to entities
            for node in graph.get("nodes", []):
                entity = Entity(
                    name=node.name,
                    type=node.type,
                    description=node.description,
                    confidence=node.confidence,
                    entity_id=node.entity_id
                )
                all_entities.append(entity)
            
            # Convert edges back to relationships
            for edge in graph.get("edges", []):
                # Find source and target names
                source_name = None
                target_name = None
                
                for node in graph.get("nodes", []):
                    if node.entity_id == edge.source_id:
                        source_name = node.name
                    if node.entity_id == edge.target_id:
                        target_name = node.name
                
                if source_name and target_name:
                    relationship = Relationship(
                        source=source_name,
                        relation=edge.relation,
                        target=target_name,
                        confidence=edge.confidence
                    )
                    all_relationships.append(relationship)
            
            # Collect document IDs
            if graph.get("metadata", {}).get("document_id"):
                all_document_ids.add(graph["metadata"]["document_id"])
        
        # Build merged graph
        merged_graph = self.build_graph(
            entities=all_entities,
            relationships=all_relationships,
            document_id=None  # Multiple documents
        )
        
        # Update metadata
        merged_graph["metadata"]["merged_from"] = len(graphs)
        merged_graph["metadata"]["source_documents"] = list(all_document_ids)
        
        logger.info(f"Merged graph: {len(merged_graph['nodes'])} nodes, {len(merged_graph['edges'])} edges")
        return merged_graph
    
    def optimize_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize graph by removing low-confidence nodes and edges.
        
        Args:
            graph_data: Graph data to optimize
            
        Returns:
            Optimized graph data
        """
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Remove low-confidence nodes
        high_conf_nodes = [node for node in nodes if node.confidence >= self.min_confidence]
        high_conf_node_ids = {node.entity_id for node in high_conf_nodes}
        
        # Remove edges connected to removed nodes or with low confidence
        high_conf_edges = [
            edge for edge in edges
            if (edge.confidence >= self.min_confidence and
                edge.source_id in high_conf_node_ids and
                edge.target_id in high_conf_node_ids)
        ]
        
        # Remove isolated nodes (nodes with no edges)
        connected_node_ids = set()
        for edge in high_conf_edges:
            connected_node_ids.add(edge.source_id)
            connected_node_ids.add(edge.target_id)
        
        final_nodes = [node for node in high_conf_nodes if node.entity_id in connected_node_ids]
        
        # Recalculate statistics
        stats = self._calculate_graph_stats(
            {node.entity_id: node for node in final_nodes},
            high_conf_edges
        )
        
        optimized_graph = {
            "nodes": final_nodes,
            "edges": high_conf_edges,
            "statistics": stats,
            "metadata": {
                **graph_data.get("metadata", {}),
                "optimized": True,
                "nodes_removed": len(nodes) - len(final_nodes),
                "edges_removed": len(edges) - len(high_conf_edges)
            }
        }
        
        logger.info(f"Graph optimized: {len(final_nodes)} nodes, {len(high_conf_edges)} edges")
        return optimized_graph
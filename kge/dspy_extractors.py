"""
DSPy-based information extraction modules for knowledge graph creation.

This module implements DSPy signatures and modules for optimized entity and relationship
extraction from text, replacing direct LLM calls with DSPy's optimization framework.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import dspy
from dspy import Signature, Module, InputField, OutputField
from dspy.teleprompt import BootstrapFewShot

from .entities import Entity, Relationship
from .preprocessing import TextChunk

logger = logging.getLogger(__name__)


# DSPy Signatures for Information Extraction
class ExtractEntities(Signature):
    """Extract named entities from text with their types and confidence scores."""
    
    text = InputField(desc="Input text to extract entities from")
    entity_types = InputField(desc="Comma-separated list of entity types to extract (e.g., 'Person, Organization, Location')")
    
    entities = OutputField(desc="JSON list of entities with name, type, description, and confidence (0.0-1.0)")


class ExtractRelationships(Signature):
    """Extract relationships between entities in text."""
    
    text = InputField(desc="Input text containing entities and their relationships")
    entities = InputField(desc="JSON list of entities found in the text")
    
    relationships = OutputField(desc="JSON list of relationships with source, relation, target, description, and confidence (0.0-1.0)")


class ValidateExtraction(Signature):
    """Validate and improve extracted entities and relationships."""
    
    text = InputField(desc="Original input text")
    entities = InputField(desc="JSON list of extracted entities")
    relationships = InputField(desc="JSON list of extracted relationships")
    
    validated_entities = OutputField(desc="JSON list of validated and improved entities")
    validated_relationships = OutputField(desc="JSON list of validated and improved relationships")
    quality_score = OutputField(desc="Overall quality score (0.0-1.0) for the extraction")


# DSPy Modules
class EntityExtractor(Module):
    """DSPy module for entity extraction with optimization capabilities."""
    
    def __init__(self, entity_types: List[str] = None):
        super().__init__()
        self.entity_types = entity_types or [
            "Person", "Organization", "Location", "Event", "Product", 
            "Technology", "Concept", "Date", "Money", "Quantity"
        ]
        self.extract = dspy.ChainOfThought(ExtractEntities)
    
    def forward(self, text: str) -> List[Entity]:
        """Extract entities from text using DSPy optimization."""
        entity_types_str = ", ".join(self.entity_types)
        
        try:
            result = self.extract(text=text, entity_types=entity_types_str)
            entities_data = json.loads(result.entities)
            
            entities = []
            for entity_data in entities_data:
                entity = Entity(
                    name=entity_data.get("name", ""),
                    type=entity_data.get("type", "Unknown"),
                    description=entity_data.get("description"),
                    confidence=float(entity_data.get("confidence", 0.5)),
                    source_text=text[:200] + "..." if len(text) > 200 else text
                )
                entities.append(entity)
            
            return entities
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse entity extraction result: {e}")
            return []


class RelationshipExtractor(Module):
    """DSPy module for relationship extraction with optimization capabilities."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractRelationships)
    
    def forward(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from text given entities using DSPy optimization."""
        entities_json = json.dumps([{
            "name": e.name,
            "type": e.type,
            "description": e.description
        } for e in entities])
        
        try:
            result = self.extract(text=text, entities=entities_json)
            relationships_data = json.loads(result.relationships)
            
            relationships = []
            for rel_data in relationships_data:
                relationship = Relationship(
                    source=rel_data.get("source", ""),
                    relation=rel_data.get("relation", "RELATED_TO"),
                    target=rel_data.get("target", ""),
                    description=rel_data.get("description"),
                    confidence=float(rel_data.get("confidence", 0.5)),
                    source_text=text[:200] + "..." if len(text) > 200 else text
                )
                relationships.append(relationship)
            
            return relationships
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse relationship extraction result: {e}")
            return []


class ExtractionValidator(Module):
    """DSPy module for validating and improving extractions."""
    
    def __init__(self):
        super().__init__()
        self.validate = dspy.ChainOfThought(ValidateExtraction)
    
    def forward(self, text: str, entities: List[Entity], relationships: List[Relationship]) -> Tuple[List[Entity], List[Relationship], float]:
        """Validate and improve extracted entities and relationships."""
        entities_json = json.dumps([{
            "name": e.name,
            "type": e.type,
            "description": e.description,
            "confidence": e.confidence
        } for e in entities])
        
        relationships_json = json.dumps([{
            "source": r.source,
            "relation": r.relation,
            "target": r.target,
            "description": r.description,
            "confidence": r.confidence
        } for r in relationships])
        
        try:
            result = self.validate(
                text=text,
                entities=entities_json,
                relationships=relationships_json
            )
            
            # Parse validated entities
            validated_entities_data = json.loads(result.validated_entities)
            validated_entities = []
            for entity_data in validated_entities_data:
                entity = Entity(
                    name=entity_data.get("name", ""),
                    type=entity_data.get("type", "Unknown"),
                    description=entity_data.get("description"),
                    confidence=float(entity_data.get("confidence", 0.5)),
                    source_text=text[:200] + "..." if len(text) > 200 else text
                )
                validated_entities.append(entity)
            
            # Parse validated relationships
            validated_relationships_data = json.loads(result.validated_relationships)
            validated_relationships = []
            for rel_data in validated_relationships_data:
                relationship = Relationship(
                    source=rel_data.get("source", ""),
                    relation=rel_data.get("relation", "RELATED_TO"),
                    target=rel_data.get("target", ""),
                    description=rel_data.get("description"),
                    confidence=float(rel_data.get("confidence", 0.5)),
                    source_text=text[:200] + "..." if len(text) > 200 else text
                )
                validated_relationships.append(relationship)
            
            quality_score = float(result.quality_score)
            
            return validated_entities, validated_relationships, quality_score
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse validation result: {e}")
            return entities, relationships, 0.5


class DSPyKGExtractor(Module):
    """Complete DSPy-based knowledge graph extraction pipeline."""
    
    def __init__(self, entity_types: List[str] = None, enable_validation: bool = True):
        super().__init__()
        self.entity_extractor = EntityExtractor(entity_types)
        self.relationship_extractor = RelationshipExtractor()
        self.enable_validation = enable_validation
        
        if enable_validation:
            self.validator = ExtractionValidator()
    
    def forward(self, text: str) -> Dict[str, Any]:
        """Complete extraction pipeline using DSPy modules."""
        # Extract entities
        entities = self.entity_extractor(text)
        
        # Extract relationships
        relationships = self.relationship_extractor(text, entities)
        
        # Validate if enabled
        quality_score = 0.7  # Default quality score
        if self.enable_validation and (entities or relationships):
            entities, relationships, quality_score = self.validator(text, entities, relationships)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "quality_score": quality_score,
            "source_text": text
        }


@dataclass
class ExtractionExample:
    """Training example for DSPy optimization."""
    text: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    quality_score: float = 1.0


class DSPyOptimizer:
    """Optimizer for DSPy-based extraction modules."""
    
    def __init__(self, extractor: DSPyKGExtractor):
        self.extractor = extractor
        self.training_examples = []
        self.validation_examples = []
    
    def add_training_example(self, example: ExtractionExample):
        """Add a training example for optimization."""
        self.training_examples.append(example)
    
    def add_validation_example(self, example: ExtractionExample):
        """Add a validation example for evaluation."""
        self.validation_examples.append(example)
    
    def create_dspy_examples(self, examples: List[ExtractionExample]) -> List[dspy.Example]:
        """Convert training examples to DSPy format."""
        dspy_examples = []
        
        for example in examples:
            # Create example for entity extraction
            entity_types_str = ", ".join(self.extractor.entity_extractor.entity_types)
            entities_json = json.dumps(example.entities)
            
            entity_example = dspy.Example(
                text=example.text,
                entity_types=entity_types_str,
                entities=entities_json
            ).with_inputs("text", "entity_types")
            
            dspy_examples.append(entity_example)
            
            # Create example for relationship extraction
            if example.relationships:
                relationships_json = json.dumps(example.relationships)
                entities_input = json.dumps([{
                    "name": e["name"],
                    "type": e["type"],
                    "description": e.get("description")
                } for e in example.entities])
                
                rel_example = dspy.Example(
                    text=example.text,
                    entities=entities_input,
                    relationships=relationships_json
                ).with_inputs("text", "entities")
                
                dspy_examples.append(rel_example)
        
        return dspy_examples
    
    def optimize_with_bootstrap(self, max_bootstrapped_demos: int = 8, max_labeled_demos: int = 4) -> DSPyKGExtractor:
        """Optimize the extractor using BootstrapFewShot."""
        if not self.training_examples:
            logger.warning("No training examples provided for optimization")
            return self.extractor
        
        # Convert examples to DSPy format
        train_examples = self.create_dspy_examples(self.training_examples)
        
        # Define metric for optimization
        def extraction_metric(example, pred, trace=None):
            """Metric for evaluating extraction quality."""
            try:
                # Simple metric based on JSON parsing success and content presence
                if hasattr(pred, 'entities'):
                    entities = json.loads(pred.entities)
                    if entities and len(entities) > 0:
                        return 1.0
                return 0.0
            except:
                return 0.0
        
        # Optimize entity extractor
        entity_optimizer = BootstrapFewShot(
            metric=extraction_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos
        )
        
        optimized_entity_extractor = entity_optimizer.compile(
            self.extractor.entity_extractor.extract,
            trainset=train_examples[:len(train_examples)//2]  # Use half for entity training
        )
        
        # Update the extractor
        self.extractor.entity_extractor.extract = optimized_entity_extractor
        
        logger.info(f"Optimized DSPy extractor with {len(train_examples)} examples")
        return self.extractor
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the extractor on validation examples."""
        if not self.validation_examples:
            logger.warning("No validation examples provided for evaluation")
            return {}
        
        total_score = 0.0
        successful_extractions = 0
        
        for example in self.validation_examples:
            try:
                result = self.extractor(example.text)
                
                # Simple evaluation metrics
                entity_score = min(len(result["entities"]) / max(len(example.entities), 1), 1.0)
                relationship_score = min(len(result["relationships"]) / max(len(example.relationships), 1), 1.0)
                quality_score = result["quality_score"]
                
                combined_score = (entity_score + relationship_score + quality_score) / 3
                total_score += combined_score
                successful_extractions += 1
                
            except Exception as e:
                logger.warning(f"Evaluation failed for example: {e}")
        
        if successful_extractions == 0:
            return {"average_score": 0.0, "success_rate": 0.0}
        
        return {
            "average_score": total_score / successful_extractions,
            "success_rate": successful_extractions / len(self.validation_examples),
            "total_examples": len(self.validation_examples)
        }


def create_sample_training_data() -> List[ExtractionExample]:
    """Create sample training data for DSPy optimization."""
    examples = [
        ExtractionExample(
            text="Apple Inc. is a technology company founded by Steve Jobs in Cupertino, California.",
            entities=[
                {"name": "Apple Inc.", "type": "Organization", "description": "Technology company", "confidence": 0.95},
                {"name": "Steve Jobs", "type": "Person", "description": "Founder of Apple", "confidence": 0.9},
                {"name": "Cupertino", "type": "Location", "description": "City in California", "confidence": 0.85},
                {"name": "California", "type": "Location", "description": "US State", "confidence": 0.9}
            ],
            relationships=[
                {"source": "Steve Jobs", "relation": "FOUNDED", "target": "Apple Inc.", "description": "Steve Jobs founded Apple Inc.", "confidence": 0.9},
                {"source": "Apple Inc.", "relation": "LOCATED_IN", "target": "Cupertino", "description": "Apple is located in Cupertino", "confidence": 0.85},
                {"source": "Cupertino", "relation": "LOCATED_IN", "target": "California", "description": "Cupertino is in California", "confidence": 0.9}
            ]
        ),
        ExtractionExample(
            text="Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975. The company is headquartered in Redmond, Washington.",
            entities=[
                {"name": "Microsoft Corporation", "type": "Organization", "description": "Software company", "confidence": 0.95},
                {"name": "Bill Gates", "type": "Person", "description": "Co-founder of Microsoft", "confidence": 0.9},
                {"name": "Paul Allen", "type": "Person", "description": "Co-founder of Microsoft", "confidence": 0.9},
                {"name": "1975", "type": "Date", "description": "Founding year", "confidence": 0.8},
                {"name": "Redmond", "type": "Location", "description": "City in Washington", "confidence": 0.85},
                {"name": "Washington", "type": "Location", "description": "US State", "confidence": 0.9}
            ],
            relationships=[
                {"source": "Bill Gates", "relation": "FOUNDED", "target": "Microsoft Corporation", "description": "Bill Gates co-founded Microsoft", "confidence": 0.9},
                {"source": "Paul Allen", "relation": "FOUNDED", "target": "Microsoft Corporation", "description": "Paul Allen co-founded Microsoft", "confidence": 0.9},
                {"source": "Microsoft Corporation", "relation": "FOUNDED_IN", "target": "1975", "description": "Microsoft was founded in 1975", "confidence": 0.8},
                {"source": "Microsoft Corporation", "relation": "HEADQUARTERED_IN", "target": "Redmond", "description": "Microsoft is headquartered in Redmond", "confidence": 0.85}
            ]
        ),
        ExtractionExample(
            text="The iPhone is a smartphone developed by Apple Inc. It was first released in 2007 and revolutionized the mobile phone industry.",
            entities=[
                {"name": "iPhone", "type": "Product", "description": "Smartphone product", "confidence": 0.95},
                {"name": "Apple Inc.", "type": "Organization", "description": "Technology company", "confidence": 0.95},
                {"name": "2007", "type": "Date", "description": "Release year", "confidence": 0.9},
                {"name": "mobile phone industry", "type": "Concept", "description": "Industry sector", "confidence": 0.8}
            ],
            relationships=[
                {"source": "Apple Inc.", "relation": "DEVELOPED", "target": "iPhone", "description": "Apple developed the iPhone", "confidence": 0.95},
                {"source": "iPhone", "relation": "RELEASED_IN", "target": "2007", "description": "iPhone was first released in 2007", "confidence": 0.9},
                {"source": "iPhone", "relation": "REVOLUTIONIZED", "target": "mobile phone industry", "description": "iPhone revolutionized the mobile phone industry", "confidence": 0.8}
            ]
        )
    ]
    
    return examples
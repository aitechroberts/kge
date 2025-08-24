"""
Tests for DSPy-based extractors.
"""

import pytest
import json
from unittest.mock import Mock, patch

from kge.dspy_extractors import (
    EntityExtractor, RelationshipExtractor, ExtractionValidator,
    DSPyKGExtractor, DSPyOptimizer, ExtractionExample,
    create_sample_training_data
)
from kge.entities import Entity, Relationship


class TestEntityExtractor:
    """Test cases for DSPy EntityExtractor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = EntityExtractor()
    
    @patch('dspy.ChainOfThought')
    def test_entity_extraction_success(self, mock_cot):
        """Test successful entity extraction."""
        # Mock the DSPy response
        mock_result = Mock()
        mock_result.entities = json.dumps([
            {"name": "Apple Inc", "type": "Organization", "description": "Tech company", "confidence": 0.9},
            {"name": "Steve Jobs", "type": "Person", "description": "Founder", "confidence": 0.8}
        ])
        
        mock_cot_instance = Mock()
        mock_cot_instance.return_value = mock_result
        mock_cot.return_value = mock_cot_instance
        
        # Test extraction
        text = "Apple Inc was founded by Steve Jobs."
        entities = self.extractor.forward(text)
        
        assert len(entities) == 2
        assert entities[0].name == "Apple Inc"
        assert entities[0].type == "Organization"
        assert entities[0].confidence == 0.9
        assert entities[1].name == "Steve Jobs"
        assert entities[1].type == "Person"
        assert entities[1].confidence == 0.8
    
    @patch('dspy.ChainOfThought')
    def test_entity_extraction_json_error(self, mock_cot):
        """Test handling of JSON parsing errors."""
        # Mock invalid JSON response
        mock_result = Mock()
        mock_result.entities = "invalid json"
        
        mock_cot_instance = Mock()
        mock_cot_instance.return_value = mock_result
        mock_cot.return_value = mock_cot_instance
        
        text = "Test text"
        entities = self.extractor.forward(text)
        
        assert entities == []
    
    def test_custom_entity_types(self):
        """Test extractor with custom entity types."""
        custom_types = ["Person", "Company"]
        extractor = EntityExtractor(entity_types=custom_types)
        
        assert extractor.entity_types == custom_types


class TestRelationshipExtractor:
    """Test cases for DSPy RelationshipExtractor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = RelationshipExtractor()
    
    @patch('dspy.ChainOfThought')
    def test_relationship_extraction_success(self, mock_cot):
        """Test successful relationship extraction."""
        # Mock the DSPy response
        mock_result = Mock()
        mock_result.relationships = json.dumps([
            {"source": "Steve Jobs", "relation": "FOUNDED", "target": "Apple Inc", "description": "Founded the company", "confidence": 0.9}
        ])
        
        mock_cot_instance = Mock()
        mock_cot_instance.return_value = mock_result
        mock_cot.return_value = mock_cot_instance
        
        # Test extraction
        text = "Steve Jobs founded Apple Inc."
        entities = [
            Entity(name="Steve Jobs", type="Person"),
            Entity(name="Apple Inc", type="Organization")
        ]
        
        relationships = self.extractor.forward(text, entities)
        
        assert len(relationships) == 1
        assert relationships[0].source == "Steve Jobs"
        assert relationships[0].relation == "FOUNDED"
        assert relationships[0].target == "Apple Inc"
        assert relationships[0].confidence == 0.9
    
    @patch('dspy.ChainOfThought')
    def test_relationship_extraction_json_error(self, mock_cot):
        """Test handling of JSON parsing errors."""
        # Mock invalid JSON response
        mock_result = Mock()
        mock_result.relationships = "invalid json"
        
        mock_cot_instance = Mock()
        mock_cot_instance.return_value = mock_result
        mock_cot.return_value = mock_cot_instance
        
        text = "Test text"
        entities = []
        relationships = self.extractor.forward(text, entities)
        
        assert relationships == []


class TestExtractionValidator:
    """Test cases for DSPy ExtractionValidator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = ExtractionValidator()
    
    @patch('dspy.ChainOfThought')
    def test_validation_success(self, mock_cot):
        """Test successful validation."""
        # Mock the DSPy response
        mock_result = Mock()
        mock_result.validated_entities = json.dumps([
            {"name": "Apple Inc", "type": "Organization", "description": "Validated tech company", "confidence": 0.95}
        ])
        mock_result.validated_relationships = json.dumps([
            {"source": "Steve Jobs", "relation": "FOUNDED", "target": "Apple Inc", "description": "Validated founding", "confidence": 0.9}
        ])
        mock_result.quality_score = "0.9"
        
        mock_cot_instance = Mock()
        mock_cot_instance.return_value = mock_result
        mock_cot.return_value = mock_cot_instance
        
        # Test validation
        text = "Test text"
        entities = [Entity(name="Apple Inc", type="Organization")]
        relationships = [Relationship(source="Steve Jobs", relation="FOUNDED", target="Apple Inc")]
        
        validated_entities, validated_relationships, quality_score = self.validator.forward(text, entities, relationships)
        
        assert len(validated_entities) == 1
        assert validated_entities[0].name == "Apple Inc"
        assert validated_entities[0].confidence == 0.95
        
        assert len(validated_relationships) == 1
        assert validated_relationships[0].source == "Steve Jobs"
        assert validated_relationships[0].confidence == 0.9
        
        assert quality_score == 0.9


class TestDSPyKGExtractor:
    """Test cases for complete DSPy KG extractor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = DSPyKGExtractor()
    
    @patch('kge.dspy_extractors.EntityExtractor')
    @patch('kge.dspy_extractors.RelationshipExtractor')
    @patch('kge.dspy_extractors.ExtractionValidator')
    def test_complete_extraction_pipeline(self, mock_validator, mock_rel_extractor, mock_entity_extractor):
        """Test the complete extraction pipeline."""
        # Mock entity extraction
        mock_entities = [Entity(name="Apple Inc", type="Organization")]
        mock_entity_extractor.return_value.return_value = mock_entities
        
        # Mock relationship extraction
        mock_relationships = [Relationship(source="Steve Jobs", relation="FOUNDED", target="Apple Inc")]
        mock_rel_extractor.return_value.return_value = mock_relationships
        
        # Mock validation
        mock_validator.return_value.return_value = (mock_entities, mock_relationships, 0.8)
        
        # Test extraction
        text = "Apple Inc was founded by Steve Jobs."
        result = self.extractor.forward(text)
        
        assert "entities" in result
        assert "relationships" in result
        assert "quality_score" in result
        assert "source_text" in result
        
        assert result["entities"] == mock_entities
        assert result["relationships"] == mock_relationships
        assert result["quality_score"] == 0.8
        assert result["source_text"] == text
    
    def test_extraction_without_validation(self):
        """Test extraction with validation disabled."""
        extractor = DSPyKGExtractor(enable_validation=False)
        assert not hasattr(extractor, 'validator')


class TestDSPyOptimizer:
    """Test cases for DSPy optimizer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = DSPyKGExtractor()
        self.optimizer = DSPyOptimizer(self.extractor)
    
    def test_add_training_example(self):
        """Test adding training examples."""
        example = ExtractionExample(
            text="Test text",
            entities=[{"name": "Test", "type": "Organization"}],
            relationships=[]
        )
        
        self.optimizer.add_training_example(example)
        assert len(self.optimizer.training_examples) == 1
        assert self.optimizer.training_examples[0] == example
    
    def test_add_validation_example(self):
        """Test adding validation examples."""
        example = ExtractionExample(
            text="Test text",
            entities=[{"name": "Test", "type": "Organization"}],
            relationships=[]
        )
        
        self.optimizer.add_validation_example(example)
        assert len(self.optimizer.validation_examples) == 1
        assert self.optimizer.validation_examples[0] == example
    
    def test_create_dspy_examples(self):
        """Test converting examples to DSPy format."""
        examples = [
            ExtractionExample(
                text="Apple Inc is a company.",
                entities=[{"name": "Apple Inc", "type": "Organization"}],
                relationships=[]
            )
        ]
        
        dspy_examples = self.optimizer.create_dspy_examples(examples)
        assert len(dspy_examples) >= 1
        # Note: Actual DSPy example creation would require DSPy to be properly configured
    
    def test_evaluate_empty_validation(self):
        """Test evaluation with no validation examples."""
        result = self.optimizer.evaluate()
        assert result == {}
    
    @patch('kge.dspy_extractors.DSPyKGExtractor')
    def test_evaluate_with_examples(self, mock_extractor):
        """Test evaluation with validation examples."""
        # Mock extractor results
        mock_result = {
            "entities": [Entity(name="Test", type="Organization")],
            "relationships": [],
            "quality_score": 0.8
        }
        mock_extractor.return_value = mock_result
        
        # Add validation example
        example = ExtractionExample(
            text="Test text",
            entities=[{"name": "Test", "type": "Organization"}],
            relationships=[]
        )
        self.optimizer.add_validation_example(example)
        
        # Mock the extractor call
        self.optimizer.extractor = mock_extractor
        
        result = self.optimizer.evaluate()
        
        assert "average_score" in result
        assert "success_rate" in result
        assert "total_examples" in result


class TestExtractionExample:
    """Test cases for ExtractionExample dataclass."""
    
    def test_example_creation(self):
        """Test creating an extraction example."""
        example = ExtractionExample(
            text="Test text",
            entities=[{"name": "Test", "type": "Organization"}],
            relationships=[{"source": "A", "relation": "RELATED_TO", "target": "B"}],
            quality_score=0.9
        )
        
        assert example.text == "Test text"
        assert len(example.entities) == 1
        assert len(example.relationships) == 1
        assert example.quality_score == 0.9
    
    def test_example_default_quality_score(self):
        """Test default quality score."""
        example = ExtractionExample(
            text="Test text",
            entities=[],
            relationships=[]
        )
        
        assert example.quality_score == 1.0


class TestSampleTrainingData:
    """Test cases for sample training data creation."""
    
    def test_create_sample_training_data(self):
        """Test creating sample training data."""
        examples = create_sample_training_data()
        
        assert len(examples) > 0
        assert all(isinstance(ex, ExtractionExample) for ex in examples)
        
        # Check first example structure
        first_example = examples[0]
        assert first_example.text
        assert len(first_example.entities) > 0
        assert len(first_example.relationships) > 0
        
        # Check entity structure
        first_entity = first_example.entities[0]
        assert "name" in first_entity
        assert "type" in first_entity
        assert "confidence" in first_entity
        
        # Check relationship structure
        first_relationship = first_example.relationships[0]
        assert "source" in first_relationship
        assert "relation" in first_relationship
        assert "target" in first_relationship
        assert "confidence" in first_relationship
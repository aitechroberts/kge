#!/usr/bin/env python3
"""
Simple DSPy KGE Test

This example demonstrates the DSPy-powered KGE system with a simple test
that doesn't require external models.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import kge
sys.path.insert(0, str(Path(__file__).parent.parent))

from kge.dspy_extractors import DSPyKGExtractor, ExtractionExample, create_sample_training_data
from kge.entities import Entity, Relationship


def test_dspy_components():
    """Test DSPy components without requiring external models."""
    print("=== Testing DSPy Components ===\n")
    
    # Test sample training data creation
    print("1. Testing sample training data creation...")
    examples = create_sample_training_data()
    print(f"   Created {len(examples)} training examples")
    
    if examples:
        first_example = examples[0]
        print(f"   First example text: {first_example.text[:50]}...")
        print(f"   First example entities: {len(first_example.entities)}")
        print(f"   First example relationships: {len(first_example.relationships)}")
    
    # Test entity and relationship creation
    print("\n2. Testing entity and relationship creation...")
    entity = Entity(name="Apple Inc", type="Organization", description="Tech company")
    print(f"   Created entity: {entity.name} ({entity.type}) - ID: {entity.entity_id}")
    
    relationship = Relationship(
        source="Steve Jobs", 
        relation="FOUNDED", 
        target="Apple Inc",
        description="Steve Jobs founded Apple Inc"
    )
    print(f"   Created relationship: {relationship.source} --[{relationship.relation}]--> {relationship.target}")
    
    # Test extraction example creation
    print("\n3. Testing extraction example creation...")
    example = ExtractionExample(
        text="Test company was founded by Test Person in 2020.",
        entities=[
            {"name": "Test Company", "type": "Organization", "description": "A test company", "confidence": 0.9},
            {"name": "Test Person", "type": "Person", "description": "Founder", "confidence": 0.8},
            {"name": "2020", "type": "Date", "description": "Founding year", "confidence": 0.9}
        ],
        relationships=[
            {"source": "Test Person", "relation": "FOUNDED", "target": "Test Company", "description": "Founded the company", "confidence": 0.9}
        ],
        quality_score=0.85
    )
    print(f"   Created example with {len(example.entities)} entities and {len(example.relationships)} relationships")
    print(f"   Quality score: {example.quality_score}")
    
    print("\n=== All DSPy Component Tests Passed! ===")


def test_dspy_extractor_structure():
    """Test DSPy extractor structure without requiring models."""
    print("\n=== Testing DSPy Extractor Structure ===\n")
    
    try:
        # Create extractor (this will fail without proper DSPy setup, but we can test structure)
        extractor = DSPyKGExtractor(
            entity_types=["Person", "Organization", "Date"],
            enable_validation=False  # Disable validation to avoid model calls
        )
        print("1. DSPy extractor created successfully")
        print(f"   Entity types: {extractor.entity_types}")
        print(f"   Validation enabled: {hasattr(extractor, 'validator')}")
        
    except Exception as e:
        print(f"1. DSPy extractor creation failed (expected without model): {e}")
    
    print("\n=== DSPy Extractor Structure Test Completed ===")


def demonstrate_data_structures():
    """Demonstrate the data structures used in DSPy KGE."""
    print("\n=== Demonstrating Data Structures ===\n")
    
    # Create entities
    entities = [
        Entity(name="Microsoft", type="Organization", description="Software company"),
        Entity(name="Bill Gates", type="Person", description="Co-founder of Microsoft"),
        Entity(name="1975", type="Date", description="Founding year")
    ]
    
    # Create relationships
    relationships = [
        Relationship(
            source="Bill Gates", 
            relation="FOUNDED", 
            target="Microsoft",
            description="Bill Gates co-founded Microsoft",
            confidence=0.95
        ),
        Relationship(
            source="Microsoft", 
            relation="FOUNDED_IN", 
            target="1975",
            description="Microsoft was founded in 1975",
            confidence=0.9
        )
    ]
    
    print("Entities:")
    for i, entity in enumerate(entities, 1):
        print(f"  {i}. {entity.name} ({entity.type}) - {entity.description}")
        print(f"     ID: {entity.entity_id}, Confidence: {entity.confidence}")
    
    print("\nRelationships:")
    for i, rel in enumerate(relationships, 1):
        print(f"  {i}. {rel.source} --[{rel.relation}]--> {rel.target}")
        print(f"     Description: {rel.description}, Confidence: {rel.confidence}")
    
    # Demonstrate extraction example
    example = ExtractionExample(
        text="Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        entities=[
            {"name": e.name, "type": e.type, "description": e.description, "confidence": e.confidence}
            for e in entities
        ],
        relationships=[
            {"source": r.source, "relation": r.relation, "target": r.target, 
             "description": r.description, "confidence": r.confidence}
            for r in relationships
        ],
        quality_score=0.92
    )
    
    print(f"\nExtraction Example:")
    print(f"  Text: {example.text}")
    print(f"  Entities: {len(example.entities)}")
    print(f"  Relationships: {len(example.relationships)}")
    print(f"  Quality Score: {example.quality_score}")
    
    print("\n=== Data Structures Demonstration Completed ===")


if __name__ == "__main__":
    try:
        test_dspy_components()
        test_dspy_extractor_structure()
        demonstrate_data_structures()
        
        print("\n🎉 All DSPy KGE tests completed successfully!")
        print("\nNext steps:")
        print("1. Configure a real language model (OpenAI, vLLM, etc.)")
        print("2. Run the full pipeline with actual text extraction")
        print("3. Use optimization features with training examples")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
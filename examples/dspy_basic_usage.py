#!/usr/bin/env python3
"""
DSPy-powered KGE Basic Usage Example

This example demonstrates how to use the DSPy-powered KGE system for
basic knowledge graph extraction from text.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import kge
sys.path.insert(0, str(Path(__file__).parent.parent))

from kge import create_pipeline, create_test_pipeline, DSPyConfig


def basic_extraction_example():
    """Demonstrate basic text extraction with DSPy."""
    print("=== DSPy KGE Basic Usage Example ===\n")
    
    # Sample text for extraction
    sample_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    Apple is known for its innovative products including the iPhone, iPad, and Mac computers.
    Tim Cook has been the CEO of Apple since 2011, succeeding Steve Jobs.
    The company's headquarters, Apple Park, opened in 2017 and can accommodate 12,000 employees.
    """
    
    # Create a test pipeline (uses mock model for demonstration)
    print("Creating DSPy KGE pipeline...")
    with create_test_pipeline() as pipeline:
        print(f"Pipeline created with model: {pipeline.dspy_manager.get_model_info()['model_name']}")
        
        # Process the text
        print("\nProcessing sample text...")
        result = pipeline.process_text(sample_text, document_id="apple_example")
        
        # Display results
        print(f"\n=== Extraction Results ===")
        print(f"Document ID: {result['document_id']}")
        print(f"Chunks processed: {result['chunks_processed']}")
        print(f"Entities extracted: {result['entities_extracted']}")
        print(f"Final entities: {result['entities_final']}")
        print(f"Relationships extracted: {result['relationships_extracted']}")
        print(f"Final relationships: {result['relationships_final']}")
        print(f"Average quality score: {result['average_quality_score']:.2f}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        
        # Show extracted entities
        print(f"\n=== Extracted Entities ===")
        for i, node in enumerate(result['graph_data']['nodes'][:10], 1):
            print(f"{i}. {node.name} ({node.type}) - Confidence: {node.confidence:.2f}")
        
        # Show extracted relationships
        print(f"\n=== Extracted Relationships ===")
        for i, edge in enumerate(result['graph_data']['edges'][:10], 1):
            print(f"{i}. {edge.source} --[{edge.relation}]--> {edge.target} - Confidence: {edge.confidence:.2f}")
        
        # Query the graph
        print(f"\n=== Graph Queries ===")
        apple_results = pipeline.query_graph("Apple")
        print(f"Nodes containing 'Apple': {len(apple_results.get('nodes', []))}")
        
        # Get statistics
        stats = pipeline.get_statistics()
        print(f"\n=== Pipeline Statistics ===")
        print(f"Storage backend: {stats['configuration']['storage_backend']}")
        print(f"Entity types: {', '.join(stats['configuration']['entity_types'][:5])}...")
        print(f"Validation enabled: {stats['configuration']['enable_validation']}")


def openai_example():
    """Example using OpenAI models (requires API key)."""
    print("\n=== OpenAI DSPy Example ===")
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Set OPENAI_API_KEY environment variable to run this example.")
        return
    
    try:
        # Create pipeline with OpenAI
        with create_pipeline(
            model_type="openai",
            model_name="gpt-3.5-turbo",
            api_key=api_key,
            storage_backend="memory"
        ) as pipeline:
            
            sample_text = """
            Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975.
            The company is headquartered in Redmond, Washington, and is known for
            developing the Windows operating system and Microsoft Office suite.
            Satya Nadella became CEO in 2014 and has led the company's transformation
            to cloud computing with Azure.
            """
            
            print("Processing text with OpenAI model...")
            result = pipeline.process_text(sample_text, document_id="microsoft_example")
            
            print(f"Extracted {result['entities_final']} entities and {result['relationships_final']} relationships")
            print(f"Quality score: {result['average_quality_score']:.2f}")
            
    except Exception as e:
        print(f"Error with OpenAI example: {e}")


def batch_processing_example():
    """Demonstrate batch processing with DSPy."""
    print("\n=== Batch Processing Example ===")
    
    texts = [
        "Google was founded by Larry Page and Sergey Brin in 1998.",
        "Amazon was started by Jeff Bezos in 1994 as an online bookstore.",
        "Tesla was founded by Martin Eberhard and Marc Tarpenning in 2003, with Elon Musk joining later."
    ]
    
    with create_test_pipeline() as pipeline:
        print(f"Processing batch of {len(texts)} texts...")
        
        results = pipeline.process_batch(texts)
        
        print(f"\n=== Batch Results ===")
        total_entities = sum(r['entities_final'] for r in results)
        total_relationships = sum(r['relationships_final'] for r in results)
        avg_quality = sum(r['average_quality_score'] for r in results) / len(results)
        
        print(f"Total entities extracted: {total_entities}")
        print(f"Total relationships extracted: {total_relationships}")
        print(f"Average quality score: {avg_quality:.2f}")
        
        for i, result in enumerate(results, 1):
            print(f"Text {i}: {result['entities_final']} entities, {result['relationships_final']} relationships")


def export_example():
    """Demonstrate graph export functionality."""
    print("\n=== Graph Export Example ===")
    
    with create_test_pipeline() as pipeline:
        # Process some text first
        text = "Netflix was founded by Reed Hastings and Marc Randolph in 1997."
        pipeline.process_text(text, document_id="netflix_example")
        
        # Export in different formats
        print("Exporting graph in different formats...")
        
        # JSON export
        json_export = pipeline.export_graph(format="json", include_metadata=True)
        print(f"JSON export length: {len(json_export)} characters")
        
        # Cypher export
        cypher_export = pipeline.export_graph(format="cypher")
        print(f"Cypher export length: {len(cypher_export)} characters")
        
        # GraphML export
        graphml_export = pipeline.export_graph(format="graphml")
        print(f"GraphML export length: {len(graphml_export)} characters")
        
        # Save to files
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "graph.json", "w") as f:
            f.write(json_export)
        
        with open(output_dir / "graph.cypher", "w") as f:
            f.write(cypher_export)
        
        with open(output_dir / "graph.graphml", "w") as f:
            f.write(graphml_export)
        
        print(f"Exported files saved to {output_dir}/")


def configuration_example():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Example ===")
    
    # Custom DSPy configuration
    from kge import DSPyConfig, DSPyKGEConfig, DSPyKGEPipeline
    
    dspy_config = DSPyConfig(
        model_name="mock-model",
        model_type="mock",
        max_tokens=1024,
        temperature=0.2,
        enable_caching=True
    )
    
    kge_config = DSPyKGEConfig(
        dspy_config=dspy_config,
        entity_types=["Person", "Organization", "Location", "Product"],
        enable_validation=True,
        min_confidence=0.7,
        chunk_size=500,
        chunk_overlap=100,
        storage_backend="memory",
        enable_optimization=False
    )
    
    with DSPyKGEPipeline(kge_config) as pipeline:
        print("Created pipeline with custom configuration:")
        print(f"- Entity types: {kge_config.entity_types}")
        print(f"- Min confidence: {kge_config.min_confidence}")
        print(f"- Chunk size: {kge_config.chunk_size}")
        print(f"- Validation enabled: {kge_config.enable_validation}")
        
        # Process text with custom config
        text = "SpaceX was founded by Elon Musk in 2002 to reduce space transportation costs."
        result = pipeline.process_text(text)
        
        print(f"Processed with custom config: {result['entities_final']} entities")


if __name__ == "__main__":
    try:
        # Run all examples
        basic_extraction_example()
        openai_example()
        batch_processing_example()
        export_example()
        configuration_example()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
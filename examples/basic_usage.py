#!/usr/bin/env python3
"""
Basic usage example for the KGE system.

This example demonstrates:
- Basic pipeline setup
- Text processing
- Querying results
- Exporting data
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import kge
sys.path.insert(0, str(Path(__file__).parent.parent))

from kge import KGEPipeline, KGEConfig
from kge.utils import DataExporter


def main():
    """Main example function."""
    print("KGE Basic Usage Example")
    print("=" * 50)
    
    # Create configuration
    config = KGEConfig(
        model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        storage_backend="memory",
        chunk_size=2500,
        enable_entity_resolution=True,
        confidence_threshold=0.7
    )
    
    # Sample text for processing
    sample_text = """
    Apple Inc is a multinational technology company headquartered in Cupertino, California.
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    Tim Cook is the current CEO of Apple Inc, having taken over from Steve Jobs in 2011.
    
    Apple develops and sells consumer electronics, computer software, and online services.
    The iPhone is Apple's flagship product and one of the most popular smartphones globally.
    Other major products include the iPad, Mac computers, Apple Watch, and AirPods.
    
    The company operates retail stores worldwide and has a significant presence in the
    technology industry. Apple competes with companies like Samsung, Google, and Microsoft
    in various market segments.
    """
    
    try:
        # Initialize pipeline
        print("Initializing KGE pipeline...")
        with KGEPipeline(config) as pipeline:
            print("✓ Pipeline initialized successfully")
            
            # Process the text
            print("\nProcessing sample text...")
            result = pipeline.process_text(sample_text, document_id="apple_example")
            
            print(f"✓ Text processed successfully")
            print(f"  - Chunks processed: {result['chunks_processed']}")
            print(f"  - Entities extracted: {result['entities_extracted']}")
            print(f"  - Relationships extracted: {result['relationships_extracted']}")
            print(f"  - Final entities: {result['entities_final']}")
            print(f"  - Final relationships: {result['relationships_final']}")
            
            # Display extracted entities
            print("\nExtracted Entities:")
            print("-" * 30)
            graph_data = result["graph_data"]
            
            for node in graph_data["nodes"]:
                print(f"  • {node.name} ({node.type}) - Confidence: {node.confidence:.2f}")
                if node.description:
                    print(f"    Description: {node.description}")
                if node.aliases:
                    print(f"    Aliases: {', '.join(node.aliases)}")
                print()
            
            # Display relationships
            print("Extracted Relationships:")
            print("-" * 30)
            
            # Create a mapping from entity IDs to names for display
            id_to_name = {node.entity_id: node.name for node in graph_data["nodes"]}
            
            for edge in graph_data["edges"]:
                source_name = id_to_name.get(edge.source_id, edge.source_id)
                target_name = id_to_name.get(edge.target_id, edge.target_id)
                print(f"  • {source_name} --[{edge.relation}]--> {target_name}")
                print(f"    Confidence: {edge.confidence:.2f}")
                print()
            
            # Query examples
            print("Query Examples:")
            print("-" * 30)
            
            # Query companies
            companies = pipeline.storage.query_nodes({"type": "Company"})
            print(f"Companies found: {len(companies)}")
            for company in companies:
                print(f"  • {company['name']}")
            
            # Query people
            people = pipeline.storage.query_nodes({"type": "Person"})
            print(f"\nPeople found: {len(people)}")
            for person in people:
                print(f"  • {person['name']}")
            
            # Query high-confidence relationships
            high_conf_rels = pipeline.storage.query_edges({"min_confidence": 0.8})
            print(f"\nHigh-confidence relationships: {len(high_conf_rels)}")
            
            # Get statistics
            print("\nSystem Statistics:")
            print("-" * 30)
            stats = pipeline.get_statistics()
            
            storage_stats = stats.get("storage_stats", {})
            print(f"Total nodes: {storage_stats.get('node_count', 0)}")
            print(f"Total edges: {storage_stats.get('edge_count', 0)}")
            
            if "node_types" in storage_stats:
                print("Node types:")
                for node_type, count in storage_stats["node_types"].items():
                    print(f"  • {node_type}: {count}")
            
            # Export example
            print("\nExporting Data:")
            print("-" * 30)
            
            # Export to JSON
            json_data = DataExporter.to_json(graph_data, pretty=True)
            print("✓ Exported to JSON format")
            
            # Save to file
            output_file = Path("knowledge_graph_export.json")
            with open(output_file, 'w') as f:
                f.write(json_data)
            print(f"✓ Saved to {output_file}")
            
            # Export to Cypher
            cypher_data = DataExporter.to_cypher(graph_data)
            cypher_file = Path("knowledge_graph_export.cypher")
            with open(cypher_file, 'w') as f:
                f.write(cypher_data)
            print(f"✓ Saved Cypher statements to {cypher_file}")
            
            print("\n" + "=" * 50)
            print("Example completed successfully!")
            print(f"Check the exported files:")
            print(f"  - {output_file}")
            print(f"  - {cypher_file}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
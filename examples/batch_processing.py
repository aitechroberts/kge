#!/usr/bin/env python3
"""
Batch processing example for the KGE system.

This example demonstrates:
- Processing multiple documents
- Performance monitoring
- Batch optimization
- Results aggregation
"""

import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import kge
sys.path.insert(0, str(Path(__file__).parent.parent))

from kge import KGEPipeline, KGEConfig


def main():
    """Main example function."""
    print("KGE Batch Processing Example")
    print("=" * 50)
    
    # Sample documents for batch processing
    documents = [
        {
            "id": "tech_companies_1",
            "text": """
            Google LLC is an American multinational technology company that specializes in 
            Internet-related services and products. Larry Page and Sergey Brin founded Google 
            while they were PhD students at Stanford University. Sundar Pichai is the current 
            CEO of Google. The company develops the Android operating system and Chrome browser.
            """
        },
        {
            "id": "tech_companies_2", 
            "text": """
            Microsoft Corporation is a multinational technology company founded by Bill Gates 
            and Paul Allen in 1975. Satya Nadella is the current CEO of Microsoft. The company 
            develops Windows operating system, Office productivity suite, and Azure cloud platform.
            Microsoft competes with Google and Amazon in cloud computing.
            """
        },
        {
            "id": "tech_companies_3",
            "text": """
            Amazon.com Inc is an American multinational technology company founded by Jeff Bezos 
            in 1994. Andy Jassy is the current CEO of Amazon. The company operates the largest 
            e-commerce platform and Amazon Web Services (AWS) cloud computing platform. 
            Amazon competes with Microsoft and Google in cloud services.
            """
        },
        {
            "id": "tech_companies_4",
            "text": """
            Meta Platforms Inc, formerly Facebook, is a social media and technology company 
            founded by Mark Zuckerberg in 2004. The company owns Facebook, Instagram, and 
            WhatsApp platforms. Meta is investing heavily in virtual reality and the metaverse 
            through its Reality Labs division.
            """
        },
        {
            "id": "tech_companies_5",
            "text": """
            Tesla Inc is an electric vehicle and clean energy company founded by Elon Musk, 
            Martin Eberhard, and Marc Tarpenning. Elon Musk serves as CEO of Tesla. The company 
            manufactures electric vehicles, energy storage systems, and solar panels. Tesla 
            competes with traditional automakers in the electric vehicle market.
            """
        }
    ]
    
    # Create configuration optimized for batch processing
    config = KGEConfig(
        model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        storage_backend="memory",
        chunk_size=1500,  # Smaller chunks for faster processing
        batch_size=3,     # Process 3 documents at a time
        enable_entity_resolution=True,
        enable_caching=True,
        enable_monitoring=True,
        confidence_threshold=0.6
    )
    
    try:
        print("Initializing KGE pipeline for batch processing...")
        with KGEPipeline(config) as pipeline:
            print("✓ Pipeline initialized successfully")
            
            # Extract texts and document IDs
            texts = [doc["text"] for doc in documents]
            doc_ids = [doc["id"] for doc in documents]
            
            print(f"\nProcessing {len(documents)} documents in batch...")
            start_time = time.time()
            
            # Process all documents in batch
            results = pipeline.process_batch(texts, doc_ids)
            
            processing_time = time.time() - start_time
            print(f"✓ Batch processing completed in {processing_time:.2f} seconds")
            
            # Analyze results
            print("\nBatch Processing Results:")
            print("-" * 40)
            
            total_entities = 0
            total_relationships = 0
            successful_docs = 0
            
            for i, result in enumerate(results):
                doc_id = doc_ids[i]
                
                if "error" in result:
                    print(f"❌ {doc_id}: Error - {result['error']}")
                else:
                    successful_docs += 1
                    entities = result.get('entities_final', 0)
                    relationships = result.get('relationships_final', 0)
                    
                    total_entities += entities
                    total_relationships += relationships
                    
                    print(f"✓ {doc_id}: {entities} entities, {relationships} relationships")
            
            print(f"\nSummary:")
            print(f"  • Successful documents: {successful_docs}/{len(documents)}")
            print(f"  • Total entities extracted: {total_entities}")
            print(f"  • Total relationships extracted: {total_relationships}")
            print(f"  • Average processing time: {processing_time/len(documents):.2f}s per document")
            
            # Query the aggregated knowledge graph
            print("\nAggregated Knowledge Graph:")
            print("-" * 40)
            
            # Get all entities by type
            all_nodes = pipeline.storage.query_nodes()
            companies = [n for n in all_nodes if n.get('type') == 'Company']
            people = [n for n in all_nodes if n.get('type') == 'Person']
            products = [n for n in all_nodes if n.get('type') == 'Product']
            
            print(f"Companies ({len(companies)}):")
            for company in companies:
                print(f"  • {company['name']} (confidence: {company.get('confidence', 0):.2f})")
            
            print(f"\nPeople ({len(people)}):")
            for person in people:
                print(f"  • {person['name']} (confidence: {person.get('confidence', 0):.2f})")
            
            if products:
                print(f"\nProducts ({len(products)}):")
                for product in products:
                    print(f"  • {product['name']} (confidence: {product.get('confidence', 0):.2f})")
            
            # Analyze relationships
            all_edges = pipeline.storage.query_edges()
            relationship_types = {}
            
            for edge in all_edges:
                rel_type = edge.get('relation', 'UNKNOWN')
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            print(f"\nRelationship Types:")
            for rel_type, count in sorted(relationship_types.items()):
                print(f"  • {rel_type}: {count}")
            
            # Performance statistics
            stats = pipeline.get_statistics()
            
            if "performance" in stats:
                print("\nPerformance Statistics:")
                print("-" * 40)
                perf = stats["performance"]
                
                print(f"Total operations: {perf.get('total_operations', 0)}")
                print(f"Success rate: {perf.get('success_rate', 0):.1%}")
                print(f"Average operation duration: {perf.get('average_duration', 0):.3f}s")
            
            # Cache statistics
            if "cache" in stats:
                cache_stats = stats["cache"]
                if "memory_cache" in cache_stats:
                    cache_info = cache_stats["memory_cache"]
                    print(f"\nCache Performance:")
                    print(f"  • Hit rate: {cache_info.get('hit_rate', 0):.1%}")
                    print(f"  • Cache size: {cache_info.get('size', 0)}")
                    print(f"  • Total hits: {cache_info.get('hits', 0)}")
                    print(f"  • Total misses: {cache_info.get('misses', 0)}")
            
            # Find interesting patterns
            print("\nInteresting Patterns:")
            print("-" * 40)
            
            # Find CEOs
            ceo_relationships = [e for e in all_edges if 'CEO' in str(e.get('relation', ''))]
            if ceo_relationships:
                print("CEO relationships found:")
                for rel in ceo_relationships:
                    print(f"  • {rel}")
            
            # Find competing companies
            competition_rels = [e for e in all_edges if 'COMPETES' in str(e.get('relation', ''))]
            if competition_rels:
                print(f"\nCompetition relationships: {len(competition_rels)}")
                for rel in competition_rels:
                    print(f"  • Competition detected")
            
            # Calculate throughput
            throughput = len(documents) / processing_time
            print(f"\nThroughput: {throughput:.2f} documents/second")
            
            print("\n" + "=" * 50)
            print("Batch processing example completed successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
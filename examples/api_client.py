#!/usr/bin/env python3
"""
API client example for the KGE system.

This example demonstrates:
- Starting the API server
- Making API requests
- Processing files via API
- Querying and exporting data
"""

import sys
import time
import json
import requests
import threading
from pathlib import Path

# Add the parent directory to the path so we can import kge
sys.path.insert(0, str(Path(__file__).parent.parent))

from kge.api import run_server
from kge.core import KGEConfig


class KGEAPIClient:
    """Simple client for the KGE API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def process_text(self, text: str, document_id: str = None):
        """Process a single text."""
        data = {"text": text}
        if document_id:
            data["document_id"] = document_id
        
        response = self.session.post(f"{self.base_url}/process/text", json=data)
        return response.json()
    
    def process_batch(self, texts: list, document_ids: list = None):
        """Process multiple texts."""
        data = {"texts": texts}
        if document_ids:
            data["document_ids"] = document_ids
        
        response = self.session.post(f"{self.base_url}/process/batch", json=data)
        return response.json()
    
    def query_nodes(self, filters: dict = None, limit: int = 100):
        """Query graph nodes."""
        data = {"limit": limit}
        if filters:
            data["node_filters"] = filters
        
        response = self.session.post(f"{self.base_url}/query/nodes", json=data)
        return response.json()
    
    def query_edges(self, filters: dict = None, limit: int = 100):
        """Query graph edges."""
        data = {"limit": limit}
        if filters:
            data["edge_filters"] = filters
        
        response = self.session.post(f"{self.base_url}/query/edges", json=data)
        return response.json()
    
    def get_statistics(self):
        """Get system statistics."""
        response = self.session.get(f"{self.base_url}/statistics")
        return response.json()
    
    def export_data(self, format: str = "json", pretty: bool = True):
        """Export graph data."""
        data = {"format": format, "pretty": pretty}
        response = self.session.post(f"{self.base_url}/export", json=data)
        return response.json()
    
    def get_config(self):
        """Get current configuration."""
        response = self.session.get(f"{self.base_url}/config")
        return response.json()


def start_api_server():
    """Start the API server in a separate thread."""
    config = KGEConfig(
        model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        storage_backend="memory",
        enable_monitoring=True,
        enable_caching=True
    )
    
    # Run server in a separate thread
    server_thread = threading.Thread(
        target=run_server,
        kwargs={"host": "0.0.0.0", "port": 8000, "config": config},
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start
    time.sleep(5)
    return server_thread


def main():
    """Main example function."""
    print("KGE API Client Example")
    print("=" * 50)
    
    # Start API server
    print("Starting API server...")
    server_thread = start_api_server()
    
    # Create API client
    client = KGEAPIClient()
    
    try:
        # Wait for server to be ready
        print("Waiting for server to be ready...")
        for i in range(10):
            try:
                health = client.health_check()
                if health.get("status") == "healthy":
                    print("✓ API server is ready")
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            print("❌ Server failed to start")
            return 1
        
        # Test basic functionality
        print("\nTesting API endpoints...")
        
        # Get configuration
        config = client.get_config()
        print(f"✓ Configuration retrieved: {config.get('storage_backend', 'unknown')} backend")
        
        # Process single text
        print("\nProcessing single text...")
        text = "Apple Inc is a technology company. Tim Cook is the CEO of Apple."
        result = client.process_text(text, "api_test_1")
        
        print(f"✓ Text processed:")
        print(f"  - Entities: {result.get('entities_final', 0)}")
        print(f"  - Relationships: {result.get('relationships_final', 0)}")
        
        # Process batch
        print("\nProcessing batch of texts...")
        texts = [
            "Google develops Android operating system.",
            "Microsoft creates Windows and Office software.",
            "Amazon operates AWS cloud platform."
        ]
        doc_ids = ["google_doc", "microsoft_doc", "amazon_doc"]
        
        batch_result = client.process_batch(texts, doc_ids)
        print(f"✓ Batch processed: {len(batch_result.get('results', []))} documents")
        
        for i, result in enumerate(batch_result.get('results', [])):
            if 'error' not in result:
                print(f"  - {doc_ids[i]}: {result.get('entities_final', 0)} entities")
        
        # Query nodes
        print("\nQuerying graph nodes...")
        
        # Get all companies
        companies = client.query_nodes({"type": "Company"})
        print(f"✓ Found {companies.get('count', 0)} companies:")
        for node in companies.get('nodes', [])[:5]:  # Show first 5
            print(f"  - {node.get('name', 'Unknown')} (confidence: {node.get('confidence', 0):.2f})")
        
        # Get all people
        people = client.query_nodes({"type": "Person"})
        print(f"✓ Found {people.get('count', 0)} people:")
        for node in people.get('nodes', [])[:5]:  # Show first 5
            print(f"  - {node.get('name', 'Unknown')} (confidence: {node.get('confidence', 0):.2f})")
        
        # Query edges
        print("\nQuerying graph edges...")
        edges = client.query_edges()
        print(f"✓ Found {edges.get('count', 0)} relationships:")
        for edge in edges.get('edges', [])[:5]:  # Show first 5
            print(f"  - {edge.get('relation', 'UNKNOWN')} (confidence: {edge.get('confidence', 0):.2f})")
        
        # Get statistics
        print("\nGetting system statistics...")
        stats = client.get_statistics()
        
        storage_stats = stats.get('storage_stats', {})
        print(f"✓ System statistics:")
        print(f"  - Total nodes: {storage_stats.get('node_count', 0)}")
        print(f"  - Total edges: {storage_stats.get('edge_count', 0)}")
        print(f"  - Backend: {storage_stats.get('backend', 'unknown')}")
        
        if 'performance' in stats:
            perf = stats['performance']
            print(f"  - Operations: {perf.get('total_operations', 0)}")
            print(f"  - Success rate: {perf.get('success_rate', 0):.1%}")
        
        # Export data
        print("\nExporting graph data...")
        
        # Export as JSON
        json_export = client.export_data("json", pretty=True)
        if json_export.get('data'):
            print("✓ JSON export successful")
            
            # Save to file
            with open("api_export.json", "w") as f:
                f.write(json_export['data'])
            print("  - Saved to api_export.json")
        
        # Export as Cypher
        cypher_export = client.export_data("cypher")
        if cypher_export.get('data'):
            print("✓ Cypher export successful")
            
            # Save to file
            with open("api_export.cypher", "w") as f:
                f.write(cypher_export['data'])
            print("  - Saved to api_export.cypher")
        
        # Demonstrate error handling
        print("\nTesting error handling...")
        try:
            # Try to process empty text
            error_result = client.process_text("")
            print("⚠ Empty text processed (unexpected)")
        except Exception as e:
            print(f"✓ Error handling works: {type(e).__name__}")
        
        # Performance test
        print("\nPerformance test...")
        start_time = time.time()
        
        # Process multiple small texts
        small_texts = [f"Test company {i} is a business." for i in range(5)]
        perf_result = client.process_batch(small_texts)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        successful = len([r for r in perf_result.get('results', []) if 'error' not in r])
        print(f"✓ Processed {successful}/5 texts in {processing_time:.2f}s")
        print(f"  - Throughput: {successful/processing_time:.2f} texts/second")
        
        print("\n" + "=" * 50)
        print("API client example completed successfully!")
        print("Files created:")
        print("  - api_export.json")
        print("  - api_export.cypher")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
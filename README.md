# Knowledge Graph Extraction (KGE) System

An optimized system for creating knowledge graphs from text through information extraction using large language models.

## Features

- **Advanced Text Processing**: Intelligent chunking with sentence boundary preservation
- **LLM-Powered Extraction**: Uses state-of-the-art language models with guided JSON decoding
- **Entity Resolution**: Automatic deduplication and merging of similar entities
- **Graph Optimization**: Confidence-based filtering and graph structure optimization
- **Multiple Storage Backends**: Support for Neo4j, Kuzu, and in-memory storage
- **Performance Optimizations**: Caching, batch processing, and model optimizations
- **REST API**: Complete API for integration with other systems
- **Comprehensive Testing**: Full test suite with unit and integration tests

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd kge

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

### Basic Usage

```python
from kge import KGEPipeline, KGEConfig

# Create configuration
config = KGEConfig(
    model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    storage_backend="memory",  # or "neo4j", "kuzu"
    chunk_size=2500,
    enable_entity_resolution=True
)

# Initialize pipeline
with KGEPipeline(config) as pipeline:
    # Process text
    text = """
    Apple Inc is a technology company based in Cupertino, California.
    Tim Cook is the CEO of Apple Inc. The company develops the iPhone,
    which is one of their most popular products.
    """
    
    result = pipeline.process_text(text, document_id="apple_doc")
    
    print(f"Extracted {result['entities_final']} entities")
    print(f"Extracted {result['relationships_final']} relationships")
    
    # Query the knowledge graph
    nodes = pipeline.storage.query_nodes({"type": "Company"})
    edges = pipeline.storage.query_edges({"relation": "WORKS_FOR"})
```

### API Server

```bash
# Start the API server
python -m kge.api

# Or with custom configuration
python -c "
from kge.api import run_server
from kge.core import KGEConfig

config = KGEConfig(model_path='your/model/path')
run_server(host='0.0.0.0', port=8000, config=config)
"
```

## Architecture

The KGE system consists of several modular components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Text Input     │───▶│  Text Processor  │───▶│  LLM Extractor  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Graph Storage  │◀───│  Graph Builder   │◀───│  Entity/Rel     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Components

1. **Text Processor**: Handles text cleaning, chunking, and preprocessing
2. **LLM Extractor**: Extracts entities and relationships using language models
3. **Graph Builder**: Constructs and optimizes knowledge graphs
4. **Graph Storage**: Manages persistence across multiple backends
5. **Performance Optimizer**: Handles caching and performance optimizations

## Configuration

### Basic Configuration

```python
from kge.core import KGEConfig

config = KGEConfig(
    # Model settings
    model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    quantization="awq_marlin",
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    
    # Processing settings
    chunk_size=2500,
    chunk_overlap=200,
    batch_size=10,
    
    # Extraction settings
    max_tokens=384,
    temperature=0.0,
    confidence_threshold=0.7,
    
    # Graph settings
    entity_similarity_threshold=0.85,
    enable_entity_resolution=True,
    
    # Storage settings
    storage_backend="memory",  # "neo4j", "kuzu", "memory"
    storage_config={},
    
    # Performance settings
    enable_caching=True,
    enable_monitoring=True
)
```

### Storage Backend Configuration

#### Neo4j
```python
config = KGEConfig(
    storage_backend="neo4j",
    storage_config={
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password"
    }
)
```

#### Kuzu
```python
config = KGEConfig(
    storage_backend="kuzu",
    storage_config={
        "db_path": "./kuzu_db"
    }
)
```

## Performance Optimization

The system includes several performance optimizations:

### Model Optimizations
- **AWQ Quantization**: Reduces memory usage by ~70%
- **FlashInfer**: 20-30% speedup in attention computation
- **Speculative Decoding**: 1.5-2x speedup for structured output
- **Prefix Caching**: Reuses computation for common prompt prefixes
- **Chunked Prefill**: Better latency for long contexts

### System Optimizations
- **Extraction Caching**: Caches extraction results to avoid recomputation
- **Batch Processing**: Optimized batch processing for multiple texts
- **Entity Resolution**: Efficient similarity-based entity deduplication

### Example: Optimized Configuration

```python
config = KGEConfig(
    model_path="Qwen/Qwen2.5-72B-Instruct-AWQ",
    quantization="awq_marlin",
    max_model_len=16384,
    gpu_memory_utilization=0.90,
    enable_caching=True,
    batch_size=16
)

# Enable additional optimizations in extractor
pipeline = KGEPipeline(config)
# FlashInfer and other optimizations are enabled automatically
```

## API Reference

### REST API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /process/text` - Process single text
- `POST /process/batch` - Process multiple texts
- `POST /process/file` - Process uploaded file
- `GET /config` - Get configuration
- `PUT /config` - Update configuration
- `POST /query/nodes` - Query graph nodes
- `POST /query/edges` - Query graph edges
- `GET /statistics` - Get system statistics
- `POST /export` - Export graph data
- `GET /types` - Get supported entity/relation types

### Python API

#### Core Classes

```python
from kge import KGEPipeline, KGEConfig
from kge.extractors import LLMExtractor, Entity, Relationship
from kge.graph import GraphBuilder, GraphNode, GraphEdge
from kge.storage import GraphStorage
```

#### Processing Methods

```python
# Single text processing
result = pipeline.process_text(text, document_id="doc1")

# Batch processing
results = pipeline.process_batch(texts, document_ids)

# File processing
result = pipeline.process_file("document.txt")
```

#### Querying

```python
# Query nodes
nodes = pipeline.storage.query_nodes({
    "type": "Company",
    "min_confidence": 0.8
})

# Query edges
edges = pipeline.storage.query_edges({
    "relation": "WORKS_FOR",
    "min_confidence": 0.7
})
```

## Examples

### Example 1: Basic Text Processing

```python
from kge import KGEPipeline, KGEConfig

config = KGEConfig(
    model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    storage_backend="memory"
)

with KGEPipeline(config) as pipeline:
    text = "Microsoft Corporation was founded by Bill Gates and Paul Allen."
    result = pipeline.process_text(text)
    
    print("Entities:")
    for node in result["graph_data"]["nodes"]:
        print(f"  {node.name} ({node.type}) - {node.confidence:.2f}")
    
    print("Relationships:")
    for edge in result["graph_data"]["edges"]:
        print(f"  {edge.source_id} --{edge.relation}--> {edge.target_id}")
```

### Example 2: Batch Processing with Custom Types

```python
from kge import KGEPipeline, KGEConfig
from kge.extractors import ExtractionSchema

# Define custom entity and relation types
entity_types = ["Company", "Person", "Product", "Technology", "Location"]
relation_types = ["FOUNDED", "DEVELOPS", "COMPETES_WITH", "LOCATED_IN", "USES"]

config = KGEConfig(
    model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    storage_backend="memory"
)

with KGEPipeline(config) as pipeline:
    # Update schema with custom types
    pipeline.extractor.update_schema(entity_types, relation_types)
    
    texts = [
        "Google develops Android operating system.",
        "Apple Inc competes with Google in mobile markets.",
        "Both companies are located in California."
    ]
    
    results = pipeline.process_batch(texts)
    
    for i, result in enumerate(results):
        print(f"Document {i+1}: {result['entities_final']} entities, {result['relationships_final']} relationships")
```

### Example 3: Neo4j Integration

```python
from kge import KGEPipeline, KGEConfig

config = KGEConfig(
    model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    storage_backend="neo4j",
    storage_config={
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password"
    }
)

with KGEPipeline(config) as pipeline:
    # Process multiple documents
    documents = [
        ("Apple Inc is a technology company.", "doc1"),
        ("Tim Cook is the CEO of Apple.", "doc2"),
        ("Apple develops the iPhone.", "doc3")
    ]
    
    for text, doc_id in documents:
        result = pipeline.process_text(text, document_id=doc_id)
        print(f"Processed {doc_id}: {result['storage_result']}")
    
    # Query the Neo4j database
    companies = pipeline.storage.query_nodes({"type": "Company"})
    print(f"Found {len(companies)} companies in the knowledge graph")
```

### Example 4: Performance Monitoring

```python
from kge import KGEPipeline, KGEConfig

config = KGEConfig(
    model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    storage_backend="memory",
    enable_monitoring=True,
    enable_caching=True
)

with KGEPipeline(config) as pipeline:
    # Process some texts
    texts = ["Sample text " + str(i) for i in range(10)]
    results = pipeline.process_batch(texts)
    
    # Get performance statistics
    stats = pipeline.get_statistics()
    
    print("Performance Statistics:")
    if "performance" in stats:
        perf = stats["performance"]
        print(f"  Total operations: {perf['total_operations']}")
        print(f"  Success rate: {perf['success_rate']:.2%}")
        print(f"  Average duration: {perf['average_duration']:.3f}s")
    
    if "cache" in stats:
        cache = stats["cache"]
        print(f"  Cache hit rate: {cache['memory_cache']['hit_rate']:.2%}")
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
pytest tests/test_preprocessing.py  # Specific module

# Run with coverage
pytest --cov=kge --cov-report=html
```

## Development

### Project Structure

```
kge/
├── kge/                    # Main package
│   ├── __init__.py
│   ├── core.py            # Main pipeline
│   ├── preprocessing.py   # Text processing
│   ├── extractors.py      # LLM extraction
│   ├── graph.py           # Graph construction
│   ├── storage.py         # Storage backends
│   ├── optimization.py    # Performance optimizations
│   ├── api.py             # REST API
│   └── utils.py           # Utilities
├── tests/                 # Test suite
├── examples/              # Example scripts
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
└── README.md
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of vLLM for efficient LLM inference
- Uses Qwen models for information extraction
- Supports Neo4j and Kuzu graph databases
- Inspired by modern knowledge graph construction techniques
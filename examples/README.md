# KGE Examples

This directory contains example scripts demonstrating various features of the KGE system.

## Examples Overview

### 1. Basic Usage (`basic_usage.py`)

Demonstrates the fundamental features of the KGE system:
- Pipeline initialization and configuration
- Processing a single text document
- Extracting entities and relationships
- Querying the knowledge graph
- Exporting data to different formats

**Run with:**
```bash
python examples/basic_usage.py
```

**What it does:**
- Processes a sample text about Apple Inc
- Shows extracted entities (companies, people, products, locations)
- Displays relationships between entities
- Demonstrates querying by entity type
- Exports results to JSON and Cypher formats

### 2. Batch Processing (`batch_processing.py`)

Shows how to efficiently process multiple documents:
- Batch processing optimization
- Performance monitoring
- Results aggregation
- Pattern analysis

**Run with:**
```bash
python examples/batch_processing.py
```

**What it does:**
- Processes 5 technology company documents in batch
- Analyzes performance metrics and throughput
- Aggregates results across all documents
- Identifies patterns like CEO relationships and competition
- Demonstrates caching benefits

### 3. API Client (`api_client.py`)

Demonstrates the REST API functionality:
- Starting the API server
- Making HTTP requests to process text
- Querying the knowledge graph via API
- Exporting data through API endpoints

**Run with:**
```bash
python examples/api_client.py
```

**What it does:**
- Starts a local API server
- Processes texts via HTTP requests
- Queries nodes and edges through the API
- Exports data in multiple formats
- Tests error handling and performance

## Prerequisites

Before running the examples, ensure you have:

1. **Installed the KGE package:**
   ```bash
   pip install -e .
   ```

2. **Required dependencies:**
   - All dependencies should be installed with the package
   - For GPU acceleration, ensure CUDA is available

3. **Model access:**
   - The examples use `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
   - Make sure you have access to this model or modify the `model_path` in the examples

## Running Examples

### Option 1: Direct execution
```bash
# From the project root directory
python examples/basic_usage.py
python examples/batch_processing.py
python examples/api_client.py
```

### Option 2: As modules
```bash
# From the project root directory
python -m examples.basic_usage
python -m examples.batch_processing
python -m examples.api_client
```

## Example Outputs

### Basic Usage Output
```
KGE Basic Usage Example
==================================================
Initializing KGE pipeline...
✓ Pipeline initialized successfully

Processing sample text...
✓ Text processed successfully
  - Chunks processed: 1
  - Entities extracted: 8
  - Relationships extracted: 6
  - Final entities: 7
  - Final relationships: 5

Extracted Entities:
------------------------------
  • Apple Inc (Company) - Confidence: 0.95
    Description: Technology company
    Aliases: apple corp

  • Tim Cook (Person) - Confidence: 0.90
    Description: CEO of Apple

...
```

### Batch Processing Output
```
KGE Batch Processing Example
==================================================
Initializing KGE pipeline for batch processing...
✓ Pipeline initialized successfully

Processing 5 documents in batch...
✓ Batch processing completed in 12.34 seconds

Batch Processing Results:
----------------------------------------
✓ tech_companies_1: 4 entities, 3 relationships
✓ tech_companies_2: 5 entities, 4 relationships
...

Throughput: 0.41 documents/second
```

### API Client Output
```
KGE API Client Example
==================================================
Starting API server...
✓ API server is ready

Testing API endpoints...
✓ Configuration retrieved: memory backend

Processing single text...
✓ Text processed:
  - Entities: 2
  - Relationships: 1
...
```

## Customization

### Modifying Model Configuration

To use a different model, update the `KGEConfig` in each example:

```python
config = KGEConfig(
    model_path="your/model/path",  # Change this
    storage_backend="memory",
    # ... other settings
)
```

### Changing Storage Backend

To use Neo4j or Kuzu instead of memory storage:

```python
# For Neo4j
config = KGEConfig(
    model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    storage_backend="neo4j",
    storage_config={
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password"
    }
)

# For Kuzu
config = KGEConfig(
    model_path="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    storage_backend="kuzu",
    storage_config={
        "db_path": "./kuzu_db"
    }
)
```

### Adding Custom Entity Types

To use custom entity and relationship types:

```python
from kge.extractors import ExtractionSchema

# Define custom types
entity_types = ["Company", "Person", "Product", "Technology", "Location"]
relation_types = ["FOUNDED", "DEVELOPS", "COMPETES_WITH", "LOCATED_IN", "USES"]

# Update the extractor schema
pipeline.extractor.update_schema(entity_types, relation_types)
```

## Troubleshooting

### Common Issues

1. **Model not found:**
   - Ensure the model path is correct
   - Check if you have access to the Hugging Face model
   - Try using a different model path

2. **GPU memory issues:**
   - Reduce `gpu_memory_utilization` in the config
   - Use a smaller model
   - Reduce `max_model_len` or `chunk_size`

3. **API server not starting:**
   - Check if port 8000 is available
   - Try a different port in the `run_server` call
   - Ensure all dependencies are installed

4. **Storage backend errors:**
   - For Neo4j: Ensure Neo4j is running and accessible
   - For Kuzu: Check write permissions for the database directory
   - Use memory backend for testing

### Performance Tips

1. **Enable caching:**
   ```python
   config = KGEConfig(
       enable_caching=True,
       # ... other settings
   )
   ```

2. **Optimize batch size:**
   ```python
   config = KGEConfig(
       batch_size=8,  # Adjust based on your GPU memory
       # ... other settings
   )
   ```

3. **Use appropriate chunk sizes:**
   ```python
   config = KGEConfig(
       chunk_size=2000,  # Smaller for faster processing
       chunk_overlap=100,
       # ... other settings
   )
   ```

## Next Steps

After running these examples, you can:

1. **Integrate with your data:**
   - Modify the examples to process your own text documents
   - Adapt the entity and relationship types to your domain

2. **Scale up:**
   - Use the batch processing patterns for large document collections
   - Implement distributed processing if needed

3. **Build applications:**
   - Use the API client patterns to build web applications
   - Integrate with existing systems using the REST API

4. **Optimize performance:**
   - Experiment with different model configurations
   - Implement custom caching strategies
   - Use GPU optimizations for production workloads
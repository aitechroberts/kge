# DSPy-Powered KGE System Guide

This guide explains how to use the DSPy-powered Knowledge Graph Extraction (KGE) system, which provides optimized information extraction using the DSPy framework.

## Overview

The DSPy-powered KGE system replaces direct LLM calls with DSPy's optimized approach, providing:

- **Structured Prompting**: DSPy signatures ensure consistent input/output formats
- **Automatic Optimization**: BootstrapFewShot and other optimizers improve performance
- **Better Error Handling**: Robust parsing and validation of LLM responses
- **Modular Design**: Separate modules for entity extraction, relationship extraction, and validation

## Quick Start

### Basic Usage

```python
from kge import create_pipeline

# Create a pipeline with OpenAI
pipeline = create_pipeline(
    model_type="openai",
    model_name="gpt-3.5-turbo",
    api_key="your-openai-key"
)

# Process text
with pipeline:
    text = "Apple Inc. was founded by Steve Jobs in 1976."
    result = pipeline.process_text(text)
    
    print(f"Entities: {result['entities_final']}")
    print(f"Relationships: {result['relationships_final']}")
```

### Using vLLM Backend

```python
from kge import create_vllm_pipeline

# Create a pipeline with vLLM
pipeline = create_vllm_pipeline(
    model_path="microsoft/DialoGPT-medium",
    host="localhost",
    port=8000
)

with pipeline:
    result = pipeline.process_text("Your text here")
```

### Testing Without External Models

```python
from kge import create_test_pipeline

# Create a test pipeline with mock model
with create_test_pipeline() as pipeline:
    result = pipeline.process_text("Test text")
    # Uses mock responses for testing
```

## DSPy Components

### 1. DSPy Signatures

The system uses three main DSPy signatures:

#### ExtractEntities
```python
class ExtractEntities(dspy.Signature):
    """Extract named entities from text."""
    text = dspy.InputField(desc="Input text to analyze")
    entity_types = dspy.InputField(desc="Types of entities to extract")
    entities = dspy.OutputField(desc="JSON list of extracted entities")
```

#### ExtractRelationships
```python
class ExtractRelationships(dspy.Signature):
    """Extract relationships between entities."""
    text = dspy.InputField(desc="Input text to analyze")
    entities = dspy.InputField(desc="JSON list of known entities")
    relationships = dspy.OutputField(desc="JSON list of extracted relationships")
```

#### ValidateExtraction
```python
class ValidateExtraction(dspy.Signature):
    """Validate and improve extracted information."""
    text = dspy.InputField(desc="Original text")
    entities = dspy.InputField(desc="Extracted entities")
    relationships = dspy.InputField(desc="Extracted relationships")
    validated_entities = dspy.OutputField(desc="Validated entities")
    validated_relationships = dspy.OutputField(desc="Validated relationships")
    quality_score = dspy.OutputField(desc="Quality score (0.0-1.0)")
```

## Configuration

### DSPy Configuration

```python
from kge import DSPyConfig, DSPyKGEConfig, DSPyKGEPipeline

# Configure DSPy model
dspy_config = DSPyConfig(
    model_name="gpt-3.5-turbo",
    model_type="openai",
    api_key="your-key",
    max_tokens=2048,
    temperature=0.1,
    enable_caching=True
)

# Configure KGE pipeline
kge_config = DSPyKGEConfig(
    dspy_config=dspy_config,
    entity_types=["Person", "Organization", "Location", "Date", "Product"],
    enable_validation=True,
    min_confidence=0.7,
    chunk_size=1000,
    storage_backend="memory",
    enable_optimization=True
)

# Create pipeline
pipeline = DSPyKGEPipeline(kge_config)
```

## Examples

See the `examples/` directory for complete examples:

- `examples/dspy_basic_usage.py` - Basic usage patterns
- `examples/dspy_optimization.py` - Optimization examples
- `examples/simple_dspy_test.py` - Testing without external models

## Migration from Original KGE

To migrate from the original KGE system:

```python
# Old way
from kge import OriginalKGEPipeline

# New way (DSPy-powered)
from kge import KGEPipeline  # Now defaults to DSPy version
```

The API is largely compatible, with additional optimization features.
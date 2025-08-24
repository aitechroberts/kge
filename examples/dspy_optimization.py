#!/usr/bin/env python3
"""
DSPy Optimization Example

This example demonstrates how to optimize the DSPy-powered KGE system
using training examples and DSPy's optimization techniques.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import kge
sys.path.insert(0, str(Path(__file__).parent.parent))

from kge import (
    create_test_pipeline, 
    DSPyKGEConfig, 
    DSPyKGEPipeline, 
    DSPyConfig,
    ExtractionExample
)
from kge.dspy_extractors import create_sample_training_data


def create_training_examples():
    """Create comprehensive training examples for optimization."""
    print("Creating training examples...")
    
    # Use the built-in sample data
    examples = create_sample_training_data()
    
    # Add more examples for better optimization
    additional_examples = [
        ExtractionExample(
            text="Facebook was founded by Mark Zuckerberg in 2004 while he was a student at Harvard University.",
            entities=[
                {"name": "Facebook", "type": "Organization", "description": "Social media company", "confidence": 0.95},
                {"name": "Mark Zuckerberg", "type": "Person", "description": "Founder of Facebook", "confidence": 0.9},
                {"name": "2004", "type": "Date", "description": "Founding year", "confidence": 0.9},
                {"name": "Harvard University", "type": "Organization", "description": "University", "confidence": 0.85}
            ],
            relationships=[
                {"source": "Mark Zuckerberg", "relation": "FOUNDED", "target": "Facebook", "description": "Mark Zuckerberg founded Facebook", "confidence": 0.95},
                {"source": "Facebook", "relation": "FOUNDED_IN", "target": "2004", "description": "Facebook was founded in 2004", "confidence": 0.9},
                {"source": "Mark Zuckerberg", "relation": "STUDIED_AT", "target": "Harvard University", "description": "Mark Zuckerberg studied at Harvard", "confidence": 0.8}
            ]
        ),
        ExtractionExample(
            text="Tesla Motors was incorporated in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined the company in 2004.",
            entities=[
                {"name": "Tesla Motors", "type": "Organization", "description": "Electric vehicle company", "confidence": 0.95},
                {"name": "Martin Eberhard", "type": "Person", "description": "Co-founder of Tesla", "confidence": 0.9},
                {"name": "Marc Tarpenning", "type": "Person", "description": "Co-founder of Tesla", "confidence": 0.9},
                {"name": "Elon Musk", "type": "Person", "description": "CEO of Tesla", "confidence": 0.95},
                {"name": "2003", "type": "Date", "description": "Incorporation year", "confidence": 0.9},
                {"name": "2004", "type": "Date", "description": "Year Musk joined", "confidence": 0.9}
            ],
            relationships=[
                {"source": "Martin Eberhard", "relation": "FOUNDED", "target": "Tesla Motors", "description": "Martin Eberhard co-founded Tesla", "confidence": 0.9},
                {"source": "Marc Tarpenning", "relation": "FOUNDED", "target": "Tesla Motors", "description": "Marc Tarpenning co-founded Tesla", "confidence": 0.9},
                {"source": "Tesla Motors", "relation": "INCORPORATED_IN", "target": "2003", "description": "Tesla was incorporated in 2003", "confidence": 0.9},
                {"source": "Elon Musk", "relation": "JOINED", "target": "Tesla Motors", "description": "Elon Musk joined Tesla in 2004", "confidence": 0.85}
            ]
        ),
        ExtractionExample(
            text="Amazon Web Services (AWS) was launched in 2006 as a subsidiary of Amazon.com. It provides cloud computing services.",
            entities=[
                {"name": "Amazon Web Services", "type": "Organization", "description": "Cloud computing service", "confidence": 0.95},
                {"name": "AWS", "type": "Organization", "description": "Abbreviation for Amazon Web Services", "confidence": 0.9},
                {"name": "Amazon.com", "type": "Organization", "description": "E-commerce company", "confidence": 0.95},
                {"name": "2006", "type": "Date", "description": "Launch year", "confidence": 0.9},
                {"name": "cloud computing services", "type": "Concept", "description": "Type of services provided", "confidence": 0.8}
            ],
            relationships=[
                {"source": "Amazon Web Services", "relation": "SUBSIDIARY_OF", "target": "Amazon.com", "description": "AWS is a subsidiary of Amazon", "confidence": 0.9},
                {"source": "Amazon Web Services", "relation": "LAUNCHED_IN", "target": "2006", "description": "AWS was launched in 2006", "confidence": 0.9},
                {"source": "Amazon Web Services", "relation": "PROVIDES", "target": "cloud computing services", "description": "AWS provides cloud computing services", "confidence": 0.85},
                {"source": "AWS", "relation": "ABBREVIATION_OF", "target": "Amazon Web Services", "description": "AWS is short for Amazon Web Services", "confidence": 0.95}
            ]
        )
    ]
    
    examples.extend(additional_examples)
    print(f"Created {len(examples)} training examples")
    return examples


def optimization_example():
    """Demonstrate DSPy optimization with training examples."""
    print("=== DSPy Optimization Example ===\n")
    
    # Create training examples
    training_examples = create_training_examples()
    
    # Create a pipeline with optimization enabled
    dspy_config = DSPyConfig(
        model_name="mock-model",
        model_type="mock",
        max_tokens=2048,
        temperature=0.1,
        enable_caching=True
    )
    
    kge_config = DSPyKGEConfig(
        dspy_config=dspy_config,
        storage_backend="memory",
        enable_optimization=True,  # Enable optimization
        max_bootstrapped_demos=6,
        max_labeled_demos=3
    )
    
    with DSPyKGEPipeline(kge_config) as pipeline:
        print("Created pipeline with optimization enabled")
        
        # Test extraction before optimization
        test_text = "Spotify was founded by Daniel Ek and Martin Lorentzon in 2006 in Stockholm, Sweden."
        
        print("\n=== Before Optimization ===")
        result_before = pipeline.process_text(test_text, document_id="spotify_before")
        print(f"Entities extracted: {result_before['entities_final']}")
        print(f"Relationships extracted: {result_before['relationships_final']}")
        print(f"Quality score: {result_before['average_quality_score']:.2f}")
        
        # Optimize the pipeline
        print(f"\n=== Optimizing with {len(training_examples)} examples ===")
        optimization_result = pipeline.optimize_with_examples(training_examples)
        
        print(f"Training examples used: {optimization_result['training_examples']}")
        print(f"Validation examples used: {optimization_result['validation_examples']}")
        print(f"Optimization successful: {optimization_result['optimization_successful']}")
        
        if optimization_result['evaluation_results']:
            eval_results = optimization_result['evaluation_results']
            print(f"Average score: {eval_results.get('average_score', 0):.2f}")
            print(f"Success rate: {eval_results.get('success_rate', 0):.2f}")
        
        # Test extraction after optimization
        print("\n=== After Optimization ===")
        result_after = pipeline.process_text(test_text, document_id="spotify_after")
        print(f"Entities extracted: {result_after['entities_final']}")
        print(f"Relationships extracted: {result_after['relationships_final']}")
        print(f"Quality score: {result_after['average_quality_score']:.2f}")
        
        # Compare results
        print(f"\n=== Optimization Impact ===")
        entity_improvement = result_after['entities_final'] - result_before['entities_final']
        rel_improvement = result_after['relationships_final'] - result_before['relationships_final']
        quality_improvement = result_after['average_quality_score'] - result_before['average_quality_score']
        
        print(f"Entity extraction change: {entity_improvement:+d}")
        print(f"Relationship extraction change: {rel_improvement:+d}")
        print(f"Quality score change: {quality_improvement:+.2f}")


def evaluation_example():
    """Demonstrate evaluation of the optimized system."""
    print("\n=== Evaluation Example ===")
    
    # Create test examples for evaluation
    test_examples = [
        ExtractionExample(
            text="Netflix was founded by Reed Hastings and Marc Randolph in 1997 as a DVD-by-mail service.",
            entities=[
                {"name": "Netflix", "type": "Organization", "description": "Streaming service", "confidence": 0.95},
                {"name": "Reed Hastings", "type": "Person", "description": "Co-founder of Netflix", "confidence": 0.9},
                {"name": "Marc Randolph", "type": "Person", "description": "Co-founder of Netflix", "confidence": 0.9},
                {"name": "1997", "type": "Date", "description": "Founding year", "confidence": 0.9}
            ],
            relationships=[
                {"source": "Reed Hastings", "relation": "FOUNDED", "target": "Netflix", "description": "Reed Hastings co-founded Netflix", "confidence": 0.9},
                {"source": "Marc Randolph", "relation": "FOUNDED", "target": "Netflix", "description": "Marc Randolph co-founded Netflix", "confidence": 0.9},
                {"source": "Netflix", "relation": "FOUNDED_IN", "target": "1997", "description": "Netflix was founded in 1997", "confidence": 0.9}
            ]
        )
    ]
    
    with create_test_pipeline() as pipeline:
        # Add validation examples to the optimizer
        if hasattr(pipeline, 'optimizer') and pipeline.optimizer:
            for example in test_examples:
                pipeline.optimizer.add_validation_example(example)
            
            # Evaluate the system
            eval_results = pipeline.optimizer.evaluate()
            
            print(f"Evaluation results:")
            print(f"- Average score: {eval_results.get('average_score', 0):.2f}")
            print(f"- Success rate: {eval_results.get('success_rate', 0):.2f}")
            print(f"- Total examples: {eval_results.get('total_examples', 0)}")
        else:
            print("Optimizer not available for evaluation")


def comparison_example():
    """Compare optimized vs non-optimized performance."""
    print("\n=== Performance Comparison Example ===")
    
    test_texts = [
        "Uber was founded by Garrett Camp and Travis Kalanick in 2009.",
        "Airbnb was started by Brian Chesky, Joe Gebbia, and Nathan Blecharczyk in 2008.",
        "Slack was created by Stewart Butterfield, Eric Costello, Cal Henderson, and Serguei Mourachov in 2013."
    ]
    
    # Test with non-optimized pipeline
    print("Testing non-optimized pipeline...")
    with create_test_pipeline() as pipeline_basic:
        results_basic = []
        for text in test_texts:
            result = pipeline_basic.process_text(text)
            results_basic.append(result)
    
    # Test with optimization-enabled pipeline
    print("Testing optimization-enabled pipeline...")
    training_examples = create_training_examples()
    
    dspy_config = DSPyConfig(model_name="mock-model", model_type="mock")
    kge_config = DSPyKGEConfig(dspy_config=dspy_config, enable_optimization=True)
    
    with DSPyKGEPipeline(kge_config) as pipeline_opt:
        # Quick optimization with fewer examples for demo
        quick_examples = training_examples[:3]  # Use fewer examples for quick demo
        pipeline_opt.optimize_with_examples(quick_examples)
        
        results_opt = []
        for text in test_texts:
            result = pipeline_opt.process_text(text)
            results_opt.append(result)
    
    # Compare results
    print(f"\n=== Comparison Results ===")
    
    basic_entities = sum(r['entities_final'] for r in results_basic)
    basic_relationships = sum(r['relationships_final'] for r in results_basic)
    basic_quality = sum(r['average_quality_score'] for r in results_basic) / len(results_basic)
    
    opt_entities = sum(r['entities_final'] for r in results_opt)
    opt_relationships = sum(r['relationships_final'] for r in results_opt)
    opt_quality = sum(r['average_quality_score'] for r in results_opt) / len(results_opt)
    
    print(f"Basic Pipeline:")
    print(f"  - Total entities: {basic_entities}")
    print(f"  - Total relationships: {basic_relationships}")
    print(f"  - Average quality: {basic_quality:.2f}")
    
    print(f"Optimized Pipeline:")
    print(f"  - Total entities: {opt_entities}")
    print(f"  - Total relationships: {opt_relationships}")
    print(f"  - Average quality: {opt_quality:.2f}")
    
    print(f"Improvements:")
    print(f"  - Entities: {opt_entities - basic_entities:+d} ({((opt_entities/basic_entities-1)*100):+.1f}%)")
    print(f"  - Relationships: {opt_relationships - basic_relationships:+d} ({((opt_relationships/basic_relationships-1)*100):+.1f}%)")
    print(f"  - Quality: {opt_quality - basic_quality:+.2f} ({((opt_quality/basic_quality-1)*100):+.1f}%)")


def save_optimized_model_example():
    """Demonstrate saving and loading optimized models."""
    print("\n=== Model Persistence Example ===")
    
    # Note: In a real implementation, you would save the optimized DSPy modules
    # For this demo, we'll show the concept
    
    print("Creating and optimizing a model...")
    training_examples = create_training_examples()[:2]  # Use fewer for demo
    
    with create_test_pipeline() as pipeline:
        if hasattr(pipeline, 'optimizer') and pipeline.optimizer:
            # Add training examples
            for example in training_examples:
                pipeline.optimizer.add_training_example(example)
            
            # In a real implementation, you would:
            # 1. Optimize the model
            # 2. Save the optimized DSPy modules to disk
            # 3. Load them in a new pipeline
            
            print("Model optimization and persistence would happen here")
            print("DSPy modules can be saved using pickle or torch.save")
            print("Configuration can be saved as JSON")
        else:
            print("Optimizer not available in test pipeline")


if __name__ == "__main__":
    try:
        # Run optimization examples
        optimization_example()
        evaluation_example()
        comparison_example()
        save_optimized_model_example()
        
        print("\n=== All Optimization Examples Completed Successfully! ===")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running optimization examples: {e}")
        import traceback
        traceback.print_exc()
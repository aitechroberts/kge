"""
Information extraction using large language models.
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

logger = logging.getLogger(__name__)


# Import from entities module to avoid circular imports
from .entities import Entity, Relationship


class ExtractionSchema:
    """Defines the JSON schema for guided extraction."""
    
    # Default entity types for general domain
    DEFAULT_ENTITY_TYPES = [
        "Person", "Organization", "Location", "Technology", 
        "Product", "Event", "Concept", "Date", "Money"
    ]
    
    # Default relationship types
    DEFAULT_RELATION_TYPES = [
        "WORKS_FOR", "LOCATED_IN", "PART_OF", "OWNS", "USES",
        "DEVELOPS", "PARTNERS_WITH", "COMPETES_WITH", "FUNDS",
        "SUPPLIES", "MANAGES", "FOUNDED", "ACQUIRED"
    ]
    
    @classmethod
    def create_schema(cls, entity_types: Optional[List[str]] = None, 
                     relation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create JSON schema for extraction.
        
        Args:
            entity_types: List of allowed entity types
            relation_types: List of allowed relation types
            
        Returns:
            JSON schema dictionary
        """
        entity_types = entity_types or cls.DEFAULT_ENTITY_TYPES
        relation_types = relation_types or cls.DEFAULT_RELATION_TYPES
        
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": entity_types
                            },
                            "description": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            }
                        },
                        "required": ["name", "type"]
                    }
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "enum": relation_types
                            },
                            "target": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            }
                        },
                        "required": ["source", "relation", "target"]
                    }
                }
            },
            "required": ["entities", "relationships"]
        }


class LLMExtractor:
    """Information extractor using large language models."""
    
    def __init__(self, 
                 model_path: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
                 quantization: str = "awq_marlin",
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.85,
                 max_tokens: int = 384,
                 temperature: float = 0.0,
                 entity_types: Optional[List[str]] = None,
                 relation_types: Optional[List[str]] = None,
                 enable_optimizations: bool = True):
        """Initialize LLM extractor.
        
        Args:
            model_path: Path to the model
            quantization: Quantization method
            max_model_len: Maximum model length
            gpu_memory_utilization: GPU memory utilization ratio
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            entity_types: Custom entity types
            relation_types: Custom relation types
            enable_optimizations: Whether to enable vLLM optimizations
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Create extraction schema
        self.schema = ExtractionSchema.create_schema(entity_types, relation_types)
        
        # Initialize LLM with optimizations
        self._initialize_llm(
            quantization=quantization,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_optimizations=enable_optimizations
        )
        
        # Create sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            guided_decoding=GuidedDecodingParams(json=self.schema)
        )
        
        logger.info(f"LLMExtractor initialized with model: {model_path}")
    
    def _initialize_llm(self, quantization: str, max_model_len: int, 
                       gpu_memory_utilization: float, enable_optimizations: bool):
        """Initialize the LLM with optimizations."""
        llm_kwargs = {
            "model": self.model_path,
            "quantization": quantization,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "guided_decoding_backend": "xgrammar"
        }
        
        if enable_optimizations:
            # Add performance optimizations
            llm_kwargs.update({
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": min(8192, max_model_len),
                "enable_prefix_caching": True,
            })
            
            # Try to enable FlashInfer if available
            try:
                llm_kwargs["attention_backend"] = "flashinfer"
                logger.info("FlashInfer attention backend enabled")
            except Exception as e:
                logger.warning(f"FlashInfer not available: {e}")
        
        try:
            self.llm = LLM(**llm_kwargs)
            logger.info("LLM initialized successfully with optimizations")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create extraction prompt for the given text.
        
        Args:
            text: Input text to extract from
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Extract entities and relationships from the following text. Return a JSON object with 'entities' and 'relationships' arrays.

For entities, identify:
- name: The exact name as it appears in text
- type: One of the predefined types
- description: Brief description (optional)
- confidence: Your confidence level (0-1)

For relationships, identify:
- source: Source entity name
- relation: Relationship type from predefined list
- target: Target entity name  
- confidence: Your confidence level (0-1)

Text to analyze:
{text}

Return only valid JSON:"""
        
        return prompt
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary containing extracted entities and relationships
        """
        if not text.strip():
            return {"entities": [], "relationships": []}
        
        try:
            # Create prompt
            prompt = self._create_extraction_prompt(text)
            
            # Generate extraction
            start_time = time.time()
            outputs = self.llm.generate([prompt], self.sampling_params)
            generation_time = time.time() - start_time
            
            # Parse output
            raw_output = outputs[0].outputs[0].text
            
            try:
                result = json.loads(raw_output)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON output: {e}")
                logger.debug(f"Raw output: {raw_output}")
                return {"entities": [], "relationships": []}
            
            # Convert to structured objects
            entities = []
            for ent_data in result.get("entities", []):
                entity = Entity(
                    name=ent_data.get("name", ""),
                    type=ent_data.get("type", ""),
                    description=ent_data.get("description"),
                    confidence=ent_data.get("confidence", 1.0),
                    source_text=text[:200] + "..." if len(text) > 200 else text
                )
                entities.append(entity)
            
            relationships = []
            for rel_data in result.get("relationships", []):
                relationship = Relationship(
                    source=rel_data.get("source", ""),
                    relation=rel_data.get("relation", ""),
                    target=rel_data.get("target", ""),
                    confidence=rel_data.get("confidence", 1.0),
                    source_text=text[:200] + "..." if len(text) > 200 else text,
                    metadata={"generation_time": generation_time}
                )
                relationships.append(relationship)
            
            logger.debug(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            
            return {
                "entities": entities,
                "relationships": relationships,
                "metadata": {
                    "generation_time": generation_time,
                    "input_length": len(text),
                    "raw_output": raw_output
                }
            }
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return {"entities": [], "relationships": [], "error": str(e)}
    
    def extract_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract from multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of extraction results
        """
        if not texts:
            return []
        
        try:
            # Create prompts
            prompts = [self._create_extraction_prompt(text) for text in texts]
            
            # Generate extractions
            start_time = time.time()
            outputs = self.llm.generate(prompts, self.sampling_params)
            total_time = time.time() - start_time
            
            results = []
            for i, (text, output) in enumerate(zip(texts, outputs)):
                try:
                    raw_output = output.outputs[0].text
                    result = json.loads(raw_output)
                    
                    # Convert to structured objects
                    entities = [
                        Entity(
                            name=ent.get("name", ""),
                            type=ent.get("type", ""),
                            description=ent.get("description"),
                            confidence=ent.get("confidence", 1.0),
                            source_text=text[:200] + "..." if len(text) > 200 else text
                        )
                        for ent in result.get("entities", [])
                    ]
                    
                    relationships = [
                        Relationship(
                            source=rel.get("source", ""),
                            relation=rel.get("relation", ""),
                            target=rel.get("target", ""),
                            confidence=rel.get("confidence", 1.0),
                            source_text=text[:200] + "..." if len(text) > 200 else text
                        )
                        for rel in result.get("relationships", [])
                    ]
                    
                    results.append({
                        "entities": entities,
                        "relationships": relationships,
                        "metadata": {
                            "batch_index": i,
                            "input_length": len(text),
                            "raw_output": raw_output
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing batch item {i}: {e}")
                    results.append({
                        "entities": [],
                        "relationships": [],
                        "error": str(e)
                    })
            
            logger.info(f"Batch extraction completed: {len(texts)} texts in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error during batch extraction: {e}")
            return [{"entities": [], "relationships": [], "error": str(e)} for _ in texts]
    
    def get_supported_types(self) -> Dict[str, List[str]]:
        """Get supported entity and relation types.
        
        Returns:
            Dictionary with entity_types and relation_types lists
        """
        return {
            "entity_types": self.schema["properties"]["entities"]["items"]["properties"]["type"]["enum"],
            "relation_types": self.schema["properties"]["relationships"]["items"]["properties"]["relation"]["enum"]
        }
    
    def update_schema(self, entity_types: Optional[List[str]] = None, 
                     relation_types: Optional[List[str]] = None):
        """Update the extraction schema.
        
        Args:
            entity_types: New entity types
            relation_types: New relation types
        """
        if entity_types is not None or relation_types is not None:
            current_entity_types = self.schema["properties"]["entities"]["items"]["properties"]["type"]["enum"]
            current_relation_types = self.schema["properties"]["relationships"]["items"]["properties"]["relation"]["enum"]
            
            self.schema = ExtractionSchema.create_schema(
                entity_types or current_entity_types,
                relation_types or current_relation_types
            )
            
            # Update sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                guided_decoding=GuidedDecodingParams(json=self.schema)
            )
            
            logger.info("Extraction schema updated")
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'llm'):
            # vLLM doesn't have explicit cleanup, but we can clear references
            del self.llm
            logger.info("LLM resources cleaned up")
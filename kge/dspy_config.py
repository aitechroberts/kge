"""
DSPy configuration and setup for the KGE system.

This module handles DSPy language model configuration and provides utilities
for setting up different LM backends (OpenAI, vLLM, local models, etc.).
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

import dspy
from dspy import LM

logger = logging.getLogger(__name__)


@dataclass
class DSPyConfig:
    """Configuration for DSPy language models and optimization."""
    
    # Language Model Configuration
    model_name: str = "gpt-3.5-turbo"
    model_type: str = "openai"  # openai, vllm, local, anthropic
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Model Parameters
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    
    # DSPy Optimization Settings
    max_bootstrapped_demos: int = 8
    max_labeled_demos: int = 4
    optimization_metric: str = "extraction_quality"
    
    # Cache and Performance
    cache_dir: Optional[str] = None
    enable_caching: bool = True
    
    # vLLM Specific Settings
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    vllm_model_path: Optional[str] = None
    
    # Local Model Settings
    local_model_path: Optional[str] = None
    device: str = "auto"
    
    def __post_init__(self):
        """Set default API key from environment if not provided."""
        if not self.api_key:
            if self.model_type == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.model_type == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")


class DSPyModelManager:
    """Manager for DSPy language model configuration and setup."""
    
    def __init__(self, config: DSPyConfig):
        self.config = config
        self.lm = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup the DSPy language model based on configuration."""
        try:
            if self.config.model_type == "openai":
                self._setup_openai_model()
            elif self.config.model_type == "vllm":
                self._setup_vllm_model()
            elif self.config.model_type == "local":
                self._setup_local_model()
            elif self.config.model_type == "anthropic":
                self._setup_anthropic_model()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Configure DSPy to use this model
            dspy.configure(lm=self.lm)
            logger.info(f"Configured DSPy with {self.config.model_type} model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup DSPy model: {e}")
            # Fallback to a mock model for testing
            self._setup_mock_model()
    
    def _setup_openai_model(self):
        """Setup OpenAI model for DSPy."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.lm = dspy.LM(
            model=self.config.model_name,
            api_key=self.config.api_key,
            api_base=self.config.api_base,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            cache=self.config.enable_caching
        )
    
    def _setup_vllm_model(self):
        """Setup vLLM model for DSPy."""
        api_base = f"http://{self.config.vllm_host}:{self.config.vllm_port}/v1"
        
        self.lm = dspy.LM(
            model=self.config.model_name,
            api_base=api_base,
            api_key="EMPTY",  # vLLM doesn't require API key
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            cache=self.config.enable_caching
        )
    
    def _setup_local_model(self):
        """Setup local model for DSPy."""
        # This would require additional setup for local models
        # For now, we'll use a placeholder implementation
        logger.warning("Local model setup not fully implemented, using mock model")
        self._setup_mock_model()
    
    def _setup_anthropic_model(self):
        """Setup Anthropic model for DSPy."""
        if not self.config.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.lm = dspy.LM(
            model=self.config.model_name,
            api_key=self.config.api_key,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            cache=self.config.enable_caching
        )
    
    def _setup_mock_model(self):
        """Setup a mock model for testing purposes."""
        try:
            from dspy import BaseLM
            
            class MockLM(BaseLM):
                def __init__(self):
                    super().__init__("mock-model")
                    self.model = "mock-model"
                
                def __call__(self, messages=None, **kwargs):
                    # Handle both message format and direct prompt
                    if messages:
                        if isinstance(messages, list) and len(messages) > 0:
                            content = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
                        else:
                            content = str(messages)
                    else:
                        content = kwargs.get('prompt', '')
                    
                    # Return mock responses for testing in DSPy format
                    if "extract entities" in content.lower():
                        return ['reasoning: I will extract entities from the text.\nentities: [{"name": "Test Entity", "type": "Organization", "description": "Mock entity", "confidence": 0.8}]']
                    elif "extract relationships" in content.lower():
                        return ['reasoning: I will extract relationships from the text.\nrelationships: [{"source": "Entity A", "relation": "RELATED_TO", "target": "Entity B", "description": "Mock relationship", "confidence": 0.7}]']
                    elif "validate" in content.lower():
                        return ['reasoning: I will validate the extracted information.\nvalidated_entities: []\nvalidated_relationships: []\nquality_score: 0.5']
                    else:
                        return ['reasoning: Processing the request.\nresponse: Mock response']
                
                def basic_request(self, prompt, **kwargs):
                    return self(messages=[{"content": prompt}], **kwargs)
            
            self.lm = MockLM()
            # Configure DSPy to use this mock model
            dspy.configure(lm=self.lm)
            logger.info("Using mock model for testing")
            
        except ImportError:
            # Fallback for older DSPy versions
            class SimpleMockLM:
                def __init__(self):
                    self.model = "mock-model"
                
                def __call__(self, prompt, **kwargs):
                    return ['{"entities": "[]", "relationships": "[]", "quality_score": "0.5"}']
            
            self.lm = SimpleMockLM()
            logger.info("Using simple mock model for testing")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model."""
        return {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "caching_enabled": self.config.enable_caching
        }
    
    def test_model(self) -> bool:
        """Test if the model is working correctly."""
        try:
            # Simple test prompt
            test_signature = dspy.Signature("text -> response", "Generate a simple response to the input text")
            test_module = dspy.ChainOfThought(test_signature)
            
            result = test_module(text="Hello, this is a test.")
            return bool(result and hasattr(result, 'response'))
            
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return False


def setup_dspy_environment(config: DSPyConfig) -> DSPyModelManager:
    """Setup the DSPy environment with the given configuration."""
    
    # Setup caching if enabled
    if config.enable_caching and config.cache_dir:
        os.makedirs(config.cache_dir, exist_ok=True)
        # DSPy will handle caching internally
    
    # Create and configure the model manager
    model_manager = DSPyModelManager(config)
    
    # Test the model
    if model_manager.test_model():
        logger.info("DSPy environment setup successfully")
    else:
        logger.warning("DSPy model test failed, but continuing with setup")
    
    return model_manager


def get_default_config() -> DSPyConfig:
    """Get default DSPy configuration."""
    return DSPyConfig(
        model_name="gpt-3.5-turbo",
        model_type="openai",
        max_tokens=2048,
        temperature=0.1,
        top_p=0.9,
        max_bootstrapped_demos=8,
        max_labeled_demos=4,
        enable_caching=True
    )


def get_vllm_config(model_path: str, host: str = "localhost", port: int = 8000) -> DSPyConfig:
    """Get DSPy configuration for vLLM backend."""
    return DSPyConfig(
        model_name=model_path.split("/")[-1],  # Use model name from path
        model_type="vllm",
        vllm_host=host,
        vllm_port=port,
        vllm_model_path=model_path,
        max_tokens=2048,
        temperature=0.1,
        top_p=0.9,
        enable_caching=True
    )


def get_local_config(model_path: str, device: str = "auto") -> DSPyConfig:
    """Get DSPy configuration for local model."""
    return DSPyConfig(
        model_name=model_path.split("/")[-1],
        model_type="local",
        local_model_path=model_path,
        device=device,
        max_tokens=2048,
        temperature=0.1,
        top_p=0.9,
        enable_caching=True
    )


# Utility functions for common configurations
def configure_for_openai(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> DSPyModelManager:
    """Quick setup for OpenAI models."""
    config = DSPyConfig(
        model_name=model,
        model_type="openai",
        api_key=api_key or os.getenv("OPENAI_API_KEY")
    )
    return setup_dspy_environment(config)


def configure_for_vllm(model_path: str, host: str = "localhost", port: int = 8000) -> DSPyModelManager:
    """Quick setup for vLLM models."""
    config = get_vllm_config(model_path, host, port)
    return setup_dspy_environment(config)


def configure_for_testing() -> DSPyModelManager:
    """Quick setup for testing with mock model."""
    config = DSPyConfig(
        model_name="mock-model",
        model_type="mock",
        enable_caching=False
    )
    manager = DSPyModelManager(config)
    manager._setup_mock_model()  # Force mock model setup
    return manager
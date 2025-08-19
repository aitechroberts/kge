Overview
This guide covers performance optimizations for vLLM when running Qwen 72B AWQ models for knowledge graph extraction from defense industry documents.
Important Clarifications
LoRA (Low-Rank Adaptation)

Purpose: Domain-specific model adaptation, NOT speed optimization
Performance Impact: Actually slows down inference by 5-10%
When to Use: Only if you have fine-tuned adapters for your specific domain

python# Example: Using LoRA for domain-specific knowledge (NOT for speed)
from vllm import LLM, LoRARequest

llm = LLM(
    model="Qwen/Qwen2.5-72B-AWQ",
    enable_lora=True
)

# Load domain-specific adapter
lora_request = LoRARequest(
    "defense-adapter",
    1,  # adapter ID
    lora_local_path="/models/defense-contracts-lora"
)
Speed Optimizations
1. FlashInfer (20-30% speedup) ✅ RECOMMENDED
Optimized attention mechanism that reduces memory bandwidth bottlenecks.
pythonllm = LLM(
    model=str(MODEL_DIR),
    quantization="awq_marlin",
    attention_backend="flashinfer",  # Add this
    # ... other settings
)
Installation:
bashpip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
2. Speculative Decoding (1.5-2x speedup) ✅ RECOMMENDED
Uses a small draft model to predict tokens, then verifies with the main model in batches.
pythonllm = LLM(
    model=str(MODEL_DIR),
    quantization="awq_marlin",
    speculative_model="Qwen/Qwen2.5-1.5B-AWQ",  # Small draft model
    num_speculative_tokens=5,  # Tokens to speculate ahead
    # ... other settings
)
Download draft model:
bashhuggingface-cli download Qwen/Qwen2.5-1.5B-AWQ --local-dir /models/Qwen2.5-1.5B-AWQ
3. Chunked Prefill (Better latency) ✅ RECOMMENDED
Improves time-to-first-token (TTFT) for long contexts.
pythonllm = LLM(
    model=str(MODEL_DIR),
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,
    # ... other settings
)
4. Prefix Caching (1.5x on repeated prompts) ✅ RECOMMENDED
Reuses KV cache for common prompt prefixes.
pythonllm = LLM(
    model=str(MODEL_DIR),
    enable_prefix_caching=True,  # Reuses KV cache for common prefixes
    # ... other settings
)
5. Pipeline Parallelism (Multi-GPU only)
Only use if you have multiple GPUs available.
pythonllm = LLM(
    model=str(MODEL_DIR),
    pipeline_parallel_size=2,  # Split model across 2 GPUs
    # ... other settings
)
Optimized Configurations
Production Configuration (A100 80GB)
pythonfrom vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Optimized for A100 80GB in production
llm = LLM(
    model="/models/Qwen2.5-72B-Instruct-AWQ",
    
    # Quantization (fastest AWQ kernel)
    quantization="awq_marlin",
    
    # Attention optimization (20-30% faster)
    attention_backend="flashinfer",
    
    # Guided generation (fastest JSON enforcement)
    guided_decoding_backend="xgrammar",
    
    # Memory settings
    gpu_memory_utilization=0.90,
    max_model_len=16384,
    
    # Batching optimizations
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,
    
    # Caching (for repetitive prompts)
    enable_prefix_caching=True,
    
    # Speculative decoding (1.5-2x for structured output)
    speculative_model="Qwen/Qwen2.5-1.5B-AWQ",
    num_speculative_tokens=4,
    
    # Hardware
    tensor_parallel_size=1,
    trust_remote_code=True,
)

# Sampling parameters for extraction
sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic for consistency
    max_tokens=384,   # Limit for structured output
    guided_decoding=GuidedDecodingParams(
        json=JSON_SCHEMA  # Your extraction schema
    ),
)
Development Configuration (RTX 4090 24GB)
python# Optimized for RTX 4090 in development
llm = LLM(
    model=str(MODEL_DIR),
    
    # Quantization
    quantization="awq_marlin",
    
    # Attention optimization
    attention_backend="flashinfer",
    
    # Guided generation
    guided_decoding_backend="xgrammar",
    
    # Memory settings (conservative for 24GB)
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    
    # Batching
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,
    
    # Caching
    enable_prefix_caching=True,
    
    # Skip speculative for dev (saves memory)
    # speculative_model=None,
    
    # Hardware
    tensor_parallel_size=1,
    trust_remote_code=True,
)
Performance Impact Summary
OptimizationSpeedupMemory ImpactUse CaseAWQ MarlinBaseline-70% memoryAlways (you have this)xgrammar1.2x for JSONMinimalStructured output (you have this)FlashInfer1.2-1.3xMinimalAlways recommendedSpeculative Decoding1.5-2x+2GB for draftStructured/short outputsPrefix Caching1.5x on repeats+10% memoryRepetitive promptsChunked PrefillBetter TTFTNoneLong contextsLoRA0.9-0.95x (slower!)+100MB/adapterDomain customization only
Expected Performance Gains
Baseline (Current Setup)

AWQ Marlin quantization
xgrammar for JSON
Performance: ~150 tokens/sec

Fully Optimized

AWQ Marlin + xgrammar
FlashInfer attention
Speculative decoding
Prefix caching
Performance: ~250-300 tokens/sec

Real-World Impact

Time per article: 2-3 seconds → 1-1.5 seconds
1000 articles: 33 minutes → 17-25 minutes
GPU cost savings: ~40-50% reduction

Implementation Checklist
Required Components

 AWQ Marlin quantization (you have this)
 xgrammar backend (you have this)
 FlashInfer installation
 Draft model for speculative decoding
 Enable prefix caching
 Enable chunked prefill

Installation Commands
bash# Install FlashInfer
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Download draft model for speculative decoding
huggingface-cli download Qwen/Qwen2.5-1.5B-AWQ \
    --local-dir /models/Qwen2.5-1.5B-AWQ

# Verify xgrammar is installed
pip show xgrammar || pip install xgrammar
Best Practices for Knowledge Graph Extraction
1. Use All Speed Optimizations
For your use case, enable:

FlashInfer (direct speedup)
Speculative decoding (perfect for JSON)
Prefix caching (same prompt template)

2. Skip These Optimizations
Don't use unless needed:

LoRA (only for domain adaptation)
Pipeline parallelism (single GPU is fine)
Tensor parallelism (AWQ 72B fits on single A100)

3. Memory Management
python# Add to your pipeline
import torch
import gc

# Before starting
torch.cuda.empty_cache()
gc.collect()

# After processing batch
if batch_num % 100 == 0:
    torch.cuda.empty_cache()
4. Error Handling
pythontry:
    llm = LLM(
        model=model_path,
        attention_backend="flashinfer",
        # ... other settings
    )
except ValueError as e:
    if "flashinfer" in str(e):
        print("FlashInfer not available, falling back to default")
        llm = LLM(
            model=model_path,
            # ... without flashinfer
        )
Monitoring Performance
python# Add metrics tracking
import time

def benchmark_extraction(llm, articles, batch_size=10):
    start = time.time()
    tokens_generated = 0
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        outputs = llm.generate(prompts, sampling_params)
        tokens_generated += sum(len(o.outputs[0].token_ids) for o in outputs)
    
    elapsed = time.time() - start
    tokens_per_second = tokens_generated / elapsed
    
    print(f"Performance: {tokens_per_second:.1f} tokens/sec")
    print(f"Time per article: {elapsed/len(articles):.2f} sec")
    
    return tokens_per_second
Troubleshooting
Issue: FlashInfer not working
bash# Reinstall with correct CUDA version
pip uninstall flashinfer
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
Issue: Speculative decoding OOM
python# Reduce speculation tokens
num_speculative_tokens=3  # Instead of 5
Issue: Prefix caching not improving performance
python# Ensure prompts share common prefix
# Good: All prompts start with same instruction
# Bad: Each prompt completely different
Next Steps

Immediate: Add FlashInfer to your setup
Next: Test speculative decoding with 1.5B draft model
Monitor: Track tokens/sec to verify improvements
Optimize: Adjust batch sizes based on performance
"""
Performance optimizations and caching for the KGE system.
"""

import logging
import hashlib
import pickle
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    
    def is_expired(self, ttl: float) -> bool:
        """Check if entry is expired."""
        return time.time() - self.timestamp > ttl


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired(self.ttl):
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.access_count += 1
                self.hits += 1
                return entry.value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 0
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            if key in self.cache:
                # Update existing entry
                self.cache[key] = entry
                self.cache.move_to_end(key)
            else:
                # Add new entry
                self.cache[key] = entry
                
                # Remove oldest entries if over limit
                while len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_size_bytes": total_size,
                "ttl": self.ttl
            }


class ExtractionCache:
    """Cache for extraction results."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_memory_entries: int = 500):
        """Initialize extraction cache.
        
        Args:
            cache_dir: Directory for persistent cache
            max_memory_entries: Maximum entries in memory cache
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_cache = LRUCache(max_size=max_memory_entries, ttl=3600)
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExtractionCache initialized with cache_dir={cache_dir}")
    
    def _get_text_hash(self, text: str) -> str:
        """Get hash for text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def get_extraction(self, text: str, model_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached extraction result.
        
        Args:
            text: Input text
            model_config: Model configuration
            
        Returns:
            Cached extraction result or None
        """
        # Create cache key
        text_hash = self._get_text_hash(text)
        config_hash = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()[:8]
        cache_key = f"{text_hash}_{config_hash}"
        
        # Try memory cache first
        result = self.memory_cache.get(cache_key)
        if result is not None:
            logger.debug(f"Cache hit (memory): {cache_key}")
            return result
        
        # Try disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    # Add to memory cache
                    self.memory_cache.put(cache_key, result)
                    logger.debug(f"Cache hit (disk): {cache_key}")
                    return result
                except Exception as e:
                    logger.warning(f"Error loading cache file {cache_file}: {e}")
        
        return None
    
    def store_extraction(self, text: str, model_config: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Store extraction result in cache.
        
        Args:
            text: Input text
            model_config: Model configuration
            result: Extraction result
        """
        # Create cache key
        text_hash = self._get_text_hash(text)
        config_hash = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()[:8]
        cache_key = f"{text_hash}_{config_hash}"
        
        # Store in memory cache
        self.memory_cache.put(cache_key, result)
        
        # Store in disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached extraction result: {cache_key}")
            except Exception as e:
                logger.warning(f"Error saving cache file {cache_file}: {e}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.memory_cache.clear()
        
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache": self.memory_cache.get_stats(),
            "disk_cache_enabled": self.cache_dir is not None
        }
        
        if self.cache_dir and self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            stats["disk_cache"] = {
                "files": len(cache_files),
                "total_size_bytes": total_size
            }
        
        return stats


class BatchProcessor:
    """Optimized batch processing for extractions."""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        """Initialize batch processor.
        
        Args:
            batch_size: Size of processing batches
            max_workers: Maximum number of worker threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        logger.info(f"BatchProcessor initialized with batch_size={batch_size}, max_workers={max_workers}")
    
    def process_texts(self, texts: List[str], extractor, progress_callback=None) -> List[Dict[str, Any]]:
        """Process texts in optimized batches.
        
        Args:
            texts: List of texts to process
            extractor: Extraction function
            progress_callback: Optional progress callback
            
        Returns:
            List of extraction results
        """
        if not texts:
            return []
        
        results = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            try:
                # Use batch extraction if available
                if hasattr(extractor, 'extract_batch'):
                    batch_results = extractor.extract_batch(batch)
                else:
                    # Fall back to individual processing
                    batch_results = [extractor.extract(text) for text in batch]
                
                results.extend(batch_results)
                
                if progress_callback:
                    progress_callback(batch_num, total_batches)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Add error results for failed batch
                error_results = [{"entities": [], "relationships": [], "error": str(e)} for _ in batch]
                results.extend(error_results)
        
        return results


class ModelOptimizer:
    """Optimizations for model inference."""
    
    def __init__(self):
        self.warmup_done = False
        self.optimal_batch_size = None
        
    def warmup_model(self, extractor, sample_texts: List[str]) -> None:
        """Warm up the model with sample texts.
        
        Args:
            extractor: Model extractor
            sample_texts: Sample texts for warmup
        """
        if self.warmup_done:
            return
        
        logger.info("Warming up model...")
        start_time = time.time()
        
        try:
            # Process sample texts to warm up the model
            for text in sample_texts[:3]:  # Use first 3 samples
                extractor.extract(text)
            
            warmup_time = time.time() - start_time
            self.warmup_done = True
            
            logger.info(f"Model warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during model warmup: {e}")
    
    def find_optimal_batch_size(self, extractor, sample_texts: List[str]) -> int:
        """Find optimal batch size for the model.
        
        Args:
            extractor: Model extractor
            sample_texts: Sample texts for testing
            
        Returns:
            Optimal batch size
        """
        if self.optimal_batch_size is not None:
            return self.optimal_batch_size
        
        if not hasattr(extractor, 'extract_batch'):
            self.optimal_batch_size = 1
            return 1
        
        logger.info("Finding optimal batch size...")
        
        batch_sizes = [1, 2, 4, 8, 16]
        best_throughput = 0
        best_batch_size = 1
        
        # Use subset of sample texts
        test_texts = sample_texts[:16] if len(sample_texts) >= 16 else sample_texts
        
        for batch_size in batch_sizes:
            if batch_size > len(test_texts):
                continue
            
            try:
                # Test this batch size
                start_time = time.time()
                
                for i in range(0, len(test_texts), batch_size):
                    batch = test_texts[i:i + batch_size]
                    extractor.extract_batch(batch)
                
                elapsed = time.time() - start_time
                throughput = len(test_texts) / elapsed
                
                logger.debug(f"Batch size {batch_size}: {throughput:.2f} texts/sec")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                    
            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
        
        self.optimal_batch_size = best_batch_size
        logger.info(f"Optimal batch size: {best_batch_size} ({best_throughput:.2f} texts/sec)")
        
        return best_batch_size


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, 
                 enable_caching: bool = True,
                 cache_dir: Optional[Path] = None,
                 max_cache_entries: int = 1000):
        """Initialize performance optimizer.
        
        Args:
            enable_caching: Whether to enable caching
            cache_dir: Directory for persistent cache
            max_cache_entries: Maximum cache entries
        """
        self.enable_caching = enable_caching
        
        # Initialize components
        if enable_caching:
            self.extraction_cache = ExtractionCache(cache_dir, max_cache_entries)
        else:
            self.extraction_cache = None
        
        self.batch_processor = BatchProcessor()
        self.model_optimizer = ModelOptimizer()
        
        logger.info(f"PerformanceOptimizer initialized (caching={'enabled' if enable_caching else 'disabled'})")
    
    def optimize_extraction(self, extractor, texts: List[str], model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize extraction process.
        
        Args:
            extractor: Model extractor
            texts: Texts to process
            model_config: Model configuration
            
        Returns:
            Extraction results
        """
        if not texts:
            return []
        
        # Warm up model if needed
        if not self.model_optimizer.warmup_done:
            self.model_optimizer.warmup_model(extractor, texts[:5])
        
        results = []
        cache_hits = 0
        
        # Check cache for each text
        if self.enable_caching and self.extraction_cache:
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cached_result = self.extraction_cache.get_extraction(text, model_config)
                if cached_result is not None:
                    results.append((i, cached_result))
                    cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            logger.info(f"Cache hits: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
            
            # Process uncached texts
            if uncached_texts:
                uncached_results = self.batch_processor.process_texts(uncached_texts, extractor)
                
                # Store in cache and add to results
                for idx, (text, result) in enumerate(zip(uncached_texts, uncached_results)):
                    original_idx = uncached_indices[idx]
                    results.append((original_idx, result))
                    
                    if self.extraction_cache:
                        self.extraction_cache.store_extraction(text, model_config, result)
        else:
            # Process all texts without caching
            uncached_results = self.batch_processor.process_texts(texts, extractor)
            results = [(i, result) for i, result in enumerate(uncached_results)]
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "caching_enabled": self.enable_caching,
            "warmup_done": self.model_optimizer.warmup_done,
            "optimal_batch_size": self.model_optimizer.optimal_batch_size
        }
        
        if self.extraction_cache:
            stats["cache"] = self.extraction_cache.get_stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        if self.extraction_cache:
            self.extraction_cache.clear()
            logger.info("Cache cleared")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.extraction_cache:
            self.extraction_cache.clear()
        logger.info("PerformanceOptimizer cleaned up")
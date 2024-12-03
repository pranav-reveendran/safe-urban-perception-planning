"""
Memory-Optimized Configuration for Large-Scale Training
======================================================

This configuration is optimized for memory efficiency when training
on the full Woven by Toyota dataset.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os

@dataclass
class MemoryOptimizedConfig:
    """Memory-optimized configuration for large-scale training."""
    
    # Data loading optimizations
    batch_size: int = 8  # Reduced for memory efficiency
    num_workers: int = 6  # Balanced for I/O and memory
    pin_memory: bool = False  # Disable to save GPU memory
    prefetch_factor: int = 2  # Reduce prefetching
    
    # Model optimizations
    use_mixed_precision: bool = True  # Enable AMP
    gradient_accumulation_steps: int = 4  # Simulate larger batch size
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Memory management
    empty_cache_frequency: int = 10  # Clear cache every N batches
    use_checkpoint: bool = True  # Gradient checkpointing
    
    # Data processing optimizations
    bev_size_reduced: Tuple[int, int] = (256, 256)  # Smaller resolution for memory
    num_sweeps_reduced: int = 3  # Fewer sweeps to save memory
    
    # Training optimizations
    save_checkpoint_frequency: int = 5  # Save less frequently
    eval_frequency: int = 2  # Evaluate less frequently
    
    # Distributed training
    use_ddp: bool = False  # Single GPU for memory optimization
    find_unused_parameters: bool = False


def apply_memory_optimizations():
    """Apply memory optimization techniques."""
    import torch
    
    # Set memory allocation settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable memory-efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("✅ Flash Attention enabled")
    except:
        print("⚠️ Flash Attention not available")
    
    # Set memory growth for CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🔧 CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def get_memory_optimized_config():
    """Get complete memory-optimized configuration."""
    config = MemoryOptimizedConfig()
    
    print("🚀 Memory-Optimized Configuration Loaded")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Mixed precision: {config.use_mixed_precision}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Reduced BEV size: {config.bev_size_reduced}")
    
    return config


if __name__ == '__main__':
    config = get_memory_optimized_config()
    apply_memory_optimizations() 
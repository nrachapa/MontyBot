"""Fast CPU configuration for 20-50x training speedup."""

import os
from training.config import Config


def get_fast_cpu_config() -> Config:
    """Get optimized configuration for CPU training with 20-50x speedup."""
    config = Config()
    
    # Device settings
    config.device = "cpu"
    
    # Aggressive parameter reduction for CPU speed
    config.simulations = 25          # 8x reduction from 200
    config.filters = 64             # 2x reduction from 128  
    config.blocks = 2               # 2x reduction from 4
    config.batch_size = 32          # Optimal for CPU
    config.games_per_iteration = 4  # Reduced for faster iterations
    config.train_steps = 5          # 4x reduction from 20
    config.eval_frequency = 100     # 10x reduction from 10
    
    # Buffer optimization
    config.buffer_capacity = 5000   # 100x reduction from 500k
    
    # ONNX Runtime optimization
    config.use_onnx = True
    config.ort_threads = os.cpu_count()
    config.ort_inter_threads = 1
    config.quantize_model = True
    
    # System optimization
    config.num_workers = min(os.cpu_count(), 8)
    config.checkpoint_frequency = 50  # Less frequent checkpoints
    
    return config


def get_ultra_fast_cpu_config() -> Config:
    """Get ultra-fast configuration for maximum CPU speedup (50-200x)."""
    config = get_fast_cpu_config()
    
    # Even more aggressive reductions
    config.simulations = 10          # Ultra-low simulations
    config.filters = 32             # Minimal network
    config.blocks = 1               # Single block
    config.games_per_iteration = 2  # Minimal games
    config.train_steps = 3          # Minimal training
    config.eval_frequency = 200     # Very infrequent evaluation
    config.buffer_capacity = 2000   # Tiny buffer
    
    return config


def get_progressive_configs() -> list[Config]:
    """Get progressive training configurations."""
    return [
        # Stage 1: Ultra-fast warmup
        _create_stage_config(filters=32, blocks=1, simulations=10, iterations=50),
        
        # Stage 2: Fast training
        _create_stage_config(filters=64, blocks=2, simulations=25, iterations=100),
        
        # Stage 3: Full training
        _create_stage_config(filters=128, blocks=4, simulations=50, iterations=200),
    ]


def _create_stage_config(filters: int, blocks: int, simulations: int, iterations: int) -> Config:
    """Create a stage configuration for progressive training."""
    config = get_fast_cpu_config()
    config.filters = filters
    config.blocks = blocks
    config.simulations = simulations
    # Store iterations as a custom attribute
    config.target_iterations = iterations
    return config
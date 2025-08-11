import os
import random
import numpy as np
import torch

from .config import Config
from .trainer import Trainer


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Initialize configuration
    config = Config()
    
    # Create trainer
    trainer = Trainer(config)
    
    print(f"Starting training with config:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Simulations: {config.simulations}")
    print(f"  Network: {config.filters} filters, {config.blocks} blocks")
    print(f"  Buffer capacity: {config.buffer_capacity}")
    
    # Training loop
    num_iterations = 100
    
    for iteration in range(num_iterations):
        # Train one iteration
        results = trainer.train_iteration(iteration)
        
        # Print progress
        if results['win_rate'] is not None:
            print(f"Iter {iteration+1}: Loss={results['loss']:.4f}, "
                  f"Win Rate={results['win_rate']:.2%}, "
                  f"Buffer={results['buffer_size']}, "
                  f"Memory={results['memory_usage']:.1f}GB, "
                  f"LR={results['lr']:.6f}")
        else:
            print(f"Iter {iteration+1}: Loss={results['loss']:.4f}, "
                  f"Buffer={results['buffer_size']}, "
                  f"Memory={results['memory_usage']:.1f}GB")
        
        # Save checkpoint
        if (iteration + 1) % config.checkpoint_frequency == 0:
            checkpoint_path = f'checkpoint_iter_{iteration+1}.pt'
            trainer.save_checkpoint(iteration, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Early stopping check
        if trainer.early_stopping(results['loss']):
            print(f"Early stopping at iteration {iteration+1}")
            break
    
    # Save final model
    torch.save(trainer.net.state_dict(), 'final_model.pt')
    
    # Save training metrics plot
    trainer.metrics.plot_metrics('training_metrics.png')
    print("Training completed!")


if __name__ == "__main__":
    main()
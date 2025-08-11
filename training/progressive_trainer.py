"""Progressive training for gradual model complexity increase."""

from typing import List
from .config import Config
from .trainer import Trainer


class ProgressiveTrainer:
    """Progressive trainer that gradually increases model complexity."""
    
    def __init__(self):
        self.stages = [
            # Stage 1: Ultra-fast warmup (32 filters, 1 block, 10 sims)
            self._create_stage_config(32, 1, 10, 50),
            
            # Stage 2: Fast training (64 filters, 2 blocks, 25 sims)  
            self._create_stage_config(64, 2, 25, 100),
            
            # Stage 3: Full training (128 filters, 4 blocks, 50 sims)
            self._create_stage_config(128, 4, 50, 200),
        ]
        self.current_trainer = None
    
    def _create_stage_config(self, filters: int, blocks: int, simulations: int, iterations: int) -> dict:
        """Create configuration for a training stage."""
        config = Config()
        config.device = "cpu"
        config.filters = filters
        config.blocks = blocks
        config.simulations = simulations
        config.batch_size = 32
        config.games_per_iteration = 4
        config.train_steps = 5
        config.eval_frequency = 100
        config.use_onnx = True
        config.quantize_model = True
        config.buffer_capacity = 5000
        
        return {
            'config': config,
            'iterations': iterations,
            'name': f"Stage({filters}f,{blocks}b,{simulations}s)"
        }
    
    def train_progressive(self) -> List[dict]:
        """Train through all progressive stages."""
        results = []
        
        for i, stage in enumerate(self.stages):
            config = stage['config']
            iterations = stage['iterations']
            name = stage['name']
            
            print(f"\n=== Starting {name} for {iterations} iterations ===")
            
            # Create trainer for this stage
            trainer = Trainer(config)
            
            # Transfer knowledge from previous stage if available
            if self.current_trainer is not None:
                self._transfer_knowledge(self.current_trainer, trainer)
            
            # Train for specified iterations
            stage_results = []
            for iteration in range(iterations):
                result = trainer.train_iteration(iteration)
                stage_results.append(result)
                
                if iteration % 20 == 0:
                    print(f"{name} Iter {iteration}: Loss={result['loss']:.4f}")
                
                # Early stopping check
                if result.get('early_stop', False):
                    print(f"{name} early stopped at iteration {iteration}")
                    break
            
            results.append({
                'stage': i,
                'name': name,
                'config': config,
                'results': stage_results
            })
            
            self.current_trainer = trainer
            print(f"=== Completed {name} ===")
        
        return results
    
    def _transfer_knowledge(self, source_trainer: Trainer, target_trainer: Trainer):
        """Transfer knowledge from source to target trainer."""
        try:
            # Transfer buffer data
            if len(source_trainer.buffer) > 0:
                # Sample some experiences from source buffer
                sample_size = min(1000, len(source_trainer.buffer))
                experiences = source_trainer.buffer.sample(sample_size)
                
                for exp in experiences:
                    target_trainer.buffer.add(*exp)
                
                print(f"Transferred {len(experiences)} experiences to new stage")
            
            # Could also transfer network weights with size matching
            # This is more complex and depends on architecture compatibility
            
        except Exception as e:
            print(f"Knowledge transfer failed: {e}")
    
    def get_final_trainer(self) -> Trainer:
        """Get the final trained model."""
        return self.current_trainer
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Optional

from .config import Config
from .device_manager import DeviceManager
from .game import Game
from .network import AlphaZeroNet
from .replay_buffer import ReplayBuffer
from .self_play import SelfPlayStrategy, ParallelSelfPlay, BatchedSelfPlay
from .evaluate import evaluate_training
from .metrics import TrainingMetrics
try:
    from .onnx_export import export_to_onnx, validate_onnx_model
    from .ort_inference import ORTInferenceEngine
    from .quantization import create_calibration_dataset, quantize_model
    ONNX_INTEGRATION_AVAILABLE = True
except ImportError:
    ONNX_INTEGRATION_AVAILABLE = False


def state_to_tensor(game: Game, state):
    """Convert game state to tensor"""
    board, player = state
    current = (board == player).astype(np.float32)
    opponent = (board == -player).astype(np.float32)
    return torch.from_numpy(np.stack([current, opponent], axis=0))


class EarlyStopping:
    """Early stopping utility with aggressive defaults for CPU training"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """Main training pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device_manager = DeviceManager(config)
        
        # Initialize components
        self.game = Game(config.board_size)
        self.net = self.device_manager.to_device(AlphaZeroNet(config))
        self.buffer = ReplayBuffer(config.buffer_capacity)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Choose self-play strategy
        if config.device == 'cpu':
            self.self_play = ParallelSelfPlay(self.game, self.net, self.buffer, config)
        else:
            self.self_play = BatchedSelfPlay(self.game, self.net, self.buffer, config, self.device_manager)
        
        # Training utilities
        self.metrics = TrainingMetrics()
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.005)
        self.baseline_net = self.device_manager.to_device(AlphaZeroNet(config))
        
        # ONNX Runtime setup
        self.onnx_engine = None
        if config.use_onnx:
            self._setup_onnx_inference()
        
        print(f"Training on: {config.device}")
        print(f"Device info: {self.device_manager.get_device_info()}")
    
    def _setup_onnx_inference(self) -> None:
        """Initialize ONNX Runtime engine for faster inference."""
        if not ONNX_INTEGRATION_AVAILABLE:
            print("ONNX integration not available, falling back to PyTorch")
            return
            
        try:
            onnx_path = "temp_model.onnx"
            export_to_onnx(self.net, onnx_path, batch_size=1)
            self.onnx_engine = ORTInferenceEngine(
                onnx_path, batch_size=1,
                intra_op_threads=self.config.ort_threads,
                inter_op_threads=self.config.ort_inter_threads
            )
            print(f"ONNX Runtime initialized with {self.config.ort_threads} threads")
        except Exception as e:
            print(f"ONNX setup failed, falling back to PyTorch: {e}")
            self.onnx_engine = None
    
    def _update_onnx_model(self, iteration: int) -> None:
        """Periodically refresh ONNX model with quantization."""
        if iteration % 10 != 0 or not self.config.use_onnx or not ONNX_INTEGRATION_AVAILABLE:
            return
        
        try:
            onnx_path = "temp_model.onnx"
            export_to_onnx(self.net, onnx_path, batch_size=1)
            
            if self.config.quantize_model:
                qpath = "temp_model_int8.onnx"
                calibration_data = create_calibration_dataset(self.game, 100)
                quantize_model(onnx_path, qpath, calibration_data)
                self.onnx_engine = ORTInferenceEngine(qpath, batch_size=1,
                    intra_op_threads=self.config.ort_threads,
                    inter_op_threads=self.config.ort_inter_threads)
                print(f"Updated to quantized ONNX model at iteration {iteration}")
            else:
                self.onnx_engine = ORTInferenceEngine(onnx_path, batch_size=1,
                    intra_op_threads=self.config.ort_threads,
                    inter_op_threads=self.config.ort_inter_threads)
                print(f"Updated ONNX model at iteration {iteration}")
        except Exception as e:
            print(f"ONNX update failed at iteration {iteration}: {e}")
    
    def _prepare_batch_fast(self, batch) -> tuple:
        """Vectorized batch preparation for CPU optimization."""
        states, policies, values = zip(*batch)
        
        # Vectorized state conversion
        state_arrays = []
        for state in states:
            tensor = state_to_tensor(self.game, state)
            state_arrays.append(tensor.numpy())
        
        state_array = np.stack(state_arrays, axis=0).astype(np.float32)
        inputs = torch.from_numpy(state_array)
        target_p = torch.tensor(np.array(policies), dtype=torch.float32)
        target_v = torch.tensor(values, dtype=torch.float32)
        
        return inputs, target_p, target_v
    
    def train_step(self) -> float:
        """Single training step"""
        if len(self.buffer) < self.config.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)
        
        # Use fast batch preparation
        inputs, target_p, target_v = self._prepare_batch_fast(batch)
        inputs = self.device_manager.to_device(inputs)
        target_p = self.device_manager.to_device(target_p)
        target_v = self.device_manager.to_device(target_v)
        
        # Forward pass with mixed precision
        with self.device_manager.autocast_context():
            log_p, v = self.net(inputs)
            
            # Compute losses
            value_loss = F.mse_loss(v, target_v)
            policy_loss = -torch.mean(torch.sum(target_p * log_p, dim=1))
            l2_loss = self.config.l2_reg * sum(p.pow(2).sum() for p in self.net.parameters())
            
            total_loss = value_loss + policy_loss + l2_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        self.device_manager.backward_and_step(total_loss, self.optimizer)
        
        return total_loss.item()
    
    def train_iteration(self, iteration: int) -> dict:
        """Complete training iteration"""
        # Self-play
        self.net.eval()
        trajectories = self.self_play.play_games(self.config.games_per_iteration)
        
        # Training
        self.net.train()
        losses = []
        for _ in range(self.config.train_steps):
            loss = self.train_step()
            if loss > 0:
                losses.append(loss)
        
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        # Learning rate scheduling (after optimizer steps)
        if losses:  # Only step if we actually trained
            self.scheduler.step()
        
        # Early stopping check
        if self.early_stopping(avg_loss):
            return {'early_stop': True, 'iteration': iteration, 'loss': avg_loss}
        
        # Evaluation (less frequent)
        win_rate = None
        if iteration % self.config.eval_frequency == 0:
            self.net.eval()
            metrics = evaluate_training(self.game, self.net, self.baseline_net, num_games=10)
            win_rate = metrics['win_rate']
        
        # Periodically refresh ONNX/INT8
        self._update_onnx_model(iteration)
        
        # Record metrics
        self.metrics.add_metrics(iteration, avg_loss, win_rate)
        
        # Memory cleanup
        if iteration % 5 == 0:
            self.device_manager.cleanup_memory()
        
        return {
            'iteration': iteration,
            'loss': avg_loss,
            'win_rate': win_rate,
            'buffer_size': len(self.buffer),
            'memory_usage': self.device_manager.get_memory_usage(),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def save_checkpoint(self, iteration: int, filepath: str):
        """Save training checkpoint"""
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'buffer': self.buffer,
            'metrics': self.metrics
        }, filepath)
    
    def load_checkpoint(self, filepath: str) -> int:
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device_manager.device)
        
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.buffer = checkpoint['buffer']
        self.metrics = checkpoint['metrics']
        
        return checkpoint['iteration']
    
    def export_onnx_model(self, filepath: str) -> None:
        """Export current model to ONNX; run validation; log diffs."""
        if not ONNX_INTEGRATION_AVAILABLE:
            raise ImportError("ONNX integration not available. Install with: pip install onnx onnxruntime")
        
        export_to_onnx(self.net, filepath, batch_size=self.config.onnx_batch_size)
        policy_diff, value_diff = validate_onnx_model(filepath, self.net)
        print(f"ONNX export completed - Policy diff: {policy_diff:.6f}, Value diff: {value_diff:.6f}")
    
    def create_ort_engine(self, onnx_path: str):
        """Create ORT engine with config.onnx_batch_size and thread settings; return engine."""
        if not ONNX_INTEGRATION_AVAILABLE:
            raise ImportError("ONNX integration not available. Install with: pip install onnx onnxruntime")
        
        return ORTInferenceEngine(
            onnx_path=onnx_path,
            batch_size=self.config.onnx_batch_size,
            intra_op_threads=self.config.ort_threads,
            inter_op_threads=self.config.ort_inter_threads
        )
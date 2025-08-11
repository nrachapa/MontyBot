from dataclasses import dataclass
import os
import torch


@dataclass
class Config:
    # Device management
    device: str = 'auto'
    use_mixed_precision: bool = False
    
    # ONNX/ORT settings
    use_onnx: bool = True
    onnx_batch_size: int = 16
    ort_threads: int = os.cpu_count()
    ort_inter_threads: int = 1
    quantize_model: bool = True
    preserve_value_head_precision: bool = True
    
    # Training parameters
    board_size: int = 3
    action_size: int = 9
    
    # Network architecture
    input_planes: int = 2
    filters: int = 128
    blocks: int = 4
    
    # MCTS parameters
    simulations: int = 200
    c_puct: float = 1.5
    temperature: float = 1.0
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    l2_reg: float = 1e-4
    
    # Self-play parameters
    games_per_iteration: int = 10
    train_steps: int = 20
    
    # Buffer parameters
    buffer_capacity: int = 50_000
    
    # Evaluation parameters
    eval_games: int = 20
    eval_frequency: int = 10
    
    # System parameters
    num_workers: int = 4
    checkpoint_frequency: int = 10
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = self._detect_device()
        
        if self.device != 'cpu':
            self._set_gpu_config()
        else:
            self._set_cpu_config()
    
    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _set_gpu_config(self):
        """Optimize for GPU training"""
        self.batch_size = 64
        self.simulations = 800
        self.filters = 256
        self.blocks = 10
        self.buffer_capacity = 500_000
        self.games_per_iteration = 25
        self.train_steps = 50
        self.use_mixed_precision = True
    
    def _set_cpu_config(self):
        """Optimize for CPU training with aggressive speedups"""
        torch.set_num_threads(os.cpu_count())
        self.num_workers = min(os.cpu_count(), 8)
        # Aggressive parameter reduction for CPU speed
        self.simulations = 25
        self.filters = 64
        self.blocks = 2
        self.batch_size = 32
        self.games_per_iteration = os.cpu_count()
        self.train_steps = 5
        self.eval_frequency = 100
        self.buffer_capacity = 5000
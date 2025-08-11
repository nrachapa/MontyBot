import unittest
import torch
import numpy as np
import os
from training.config import Config
from training.device_manager import DeviceManager
from training.game import Game
from training.replay_buffer import ReplayBuffer
from training.metrics import TrainingMetrics
from training.train import set_seeds


class TestCore(unittest.TestCase):
    
    def setUp(self):
        set_seeds(42)
    
    def test_config_and_device_detection(self):
        """Merged: config validation, device detection, auto-selection"""
        # Default config
        config = Config()
        self.assertIn(config.device, ['cpu', 'cuda', 'mps'])
        self.assertEqual(config.board_size, 3)
        self.assertEqual(config.action_size, 9)
        
        # CPU config
        cpu_config = Config(device='cpu')
        self.assertEqual(cpu_config.device, 'cpu')
        self.assertEqual(cpu_config.batch_size, 16)
        self.assertFalse(cpu_config.use_mixed_precision)
        
        # Auto device selection
        auto_config = Config(device='auto')
        self.assertIn(auto_config.device, ['cpu', 'cuda', 'mps'])
    
    def test_device_manager_operations(self):
        """Merged: device initialization, tensor operations, memory management"""
        config = Config(device='cpu')
        device_manager = DeviceManager(config)
        
        # Initialization
        self.assertEqual(str(device_manager.device), 'cpu')
        self.assertFalse(device_manager.use_amp)
        self.assertIsNone(device_manager.scaler)
        
        # Tensor operations
        torch.manual_seed(42)
        tensor = torch.randn(2, 3)
        device_tensor = device_manager.to_device(tensor)
        self.assertEqual(device_tensor.device.type, 'cpu')
        
        # Memory operations
        device_manager.cleanup_memory()
        usage = device_manager.get_memory_usage()
        self.assertIsInstance(usage, float)
        self.assertGreater(usage, 0)
        
        # Device info
        info = device_manager.get_device_info()
        self.assertEqual(info['device'], 'cpu')
        self.assertFalse(info['mixed_precision'])
    
    def test_game_mechanics(self):
        """Merged: initial state, legal moves, winner detection"""
        game = Game()
        
        # Initial state
        state = game.get_initial_state()
        board, player = state
        self.assertEqual(board.shape, (3, 3))
        self.assertTrue(np.all(board == 0))
        self.assertEqual(player, 1)
        
        # Legal moves
        moves = game.get_legal_moves(state)
        self.assertEqual(len(moves), 9)
        state = game.apply_move(state, moves[0])
        moves = game.get_legal_moves(state)
        self.assertEqual(len(moves), 8)
        
        # Winner detection
        state = game.get_initial_state()
        moves = [0, 3, 1, 4, 2]  # X wins first row
        for m in moves:
            state = game.apply_move(state, m)
        self.assertTrue(game.is_terminal(state))
        self.assertEqual(game.get_winner(state), 1)
    
    def test_buffer_and_metrics(self):
        """Merged: replay buffer operations, metrics tracking"""
        # Buffer operations
        buffer = ReplayBuffer(capacity=3)
        for i in range(5):
            buffer.add(i, i, i)
        self.assertEqual(len(buffer), 3)
        
        import random
        random.seed(42)
        sample = buffer.sample(2)
        self.assertEqual(len(sample), 2)
        for entry in sample:
            self.assertEqual(len(entry), 3)
        
        # Metrics tracking
        metrics = TrainingMetrics()
        metrics.add_metrics(1, 0.5, 0.6)
        self.assertEqual(len(metrics.losses), 1)
        self.assertEqual(len(metrics.win_rates), 1)
        
        # Plot metrics
        for i in range(5):
            metrics.add_metrics(i, 0.5 - i*0.1, 0.5 + i*0.1)
        
        test_path = 'test_metrics.png'
        metrics.plot_metrics(test_path)
        self.assertTrue(os.path.exists(test_path))
        os.remove(test_path)


if __name__ == '__main__':
    unittest.main()
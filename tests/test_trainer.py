import unittest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import patch
from training.config import Config
from training.trainer import Trainer, EarlyStopping, state_to_tensor
from training.game import Game
from training.network import AlphaZeroNet
from training.replay_buffer import ReplayBuffer
from training.self_play import ParallelSelfPlay


class TestTrainer(unittest.TestCase):
    
    def setUp(self):
        self.config = Config(device='cpu', simulations=10, games_per_iteration=2, train_steps=2, eval_frequency=100)
        self.trainer = Trainer(self.config)
    
    def test_trainer_initialization_and_components(self):
        """Merged: trainer initialization, state conversion, device consistency"""
        # Trainer components
        self.assertIsNotNone(self.trainer.game)
        self.assertIsNotNone(self.trainer.net)
        self.assertIsNotNone(self.trainer.buffer)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.self_play)
        
        # State to tensor conversion
        game = Game()
        state = game.get_initial_state()
        tensor = state_to_tensor(game, state)
        self.assertEqual(tensor.shape, (2, 3, 3))
        self.assertEqual(tensor.dtype, torch.float32)
        
        # Device consistency
        device = self.trainer.device_manager.device
        for param in self.trainer.net.parameters():
            self.assertEqual(param.device, device)
        
        # Self-play strategy
        self.assertIsInstance(self.trainer.self_play, ParallelSelfPlay)
    
    def test_training_operations(self):
        """Merged: train step, buffer operations, early stopping"""
        # Empty buffer
        loss = self.trainer.train_step()
        self.assertEqual(loss, 0.0)
        
        # With data
        game = Game()
        state = game.get_initial_state()
        policy = np.ones(9) / 9
        value = 0.5
        
        for _ in range(20):
            self.trainer.buffer.add(state, policy, value)
        
        loss = self.trainer.train_step()
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        self.assertFalse(early_stopping(1.0))
        self.assertFalse(early_stopping(0.9))
        self.assertFalse(early_stopping(0.91))
        self.assertFalse(early_stopping(0.92))
        self.assertTrue(early_stopping(0.93))
    
    @patch('training.self_play.Pool')
    def test_training_iteration_and_checkpoints(self, mock_pool):
        """Merged: training iteration, checkpoint save/load, mocked self-play"""
        mock_pool.return_value.__enter__.return_value.map.return_value = [[]]
        
        # Training iteration
        results = self.trainer.train_iteration(0)
        expected_keys = ['iteration', 'loss', 'buffer_size', 'memory_usage', 'lr']
        for key in expected_keys:
            self.assertIn(key, results)
        
        self.assertEqual(results['iteration'], 0)
        self.assertIsInstance(results['loss'], float)
        
        # Checkpoint operations
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            self.trainer.save_checkpoint(0, checkpoint_path)
            
            new_trainer = Trainer(self.config)
            loaded_iteration = new_trainer.load_checkpoint(checkpoint_path)
            
            self.assertEqual(loaded_iteration, 0)
            self.assertEqual(len(new_trainer.buffer), len(self.trainer.buffer))
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


if __name__ == '__main__':
    unittest.main()
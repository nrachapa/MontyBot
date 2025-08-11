import unittest
import torch
import os
import tempfile
from unittest.mock import patch
from training.config import Config
from training.trainer import Trainer
from training.train import set_seeds
from training.evaluate import play_game, evaluate_networks, evaluate_training
from training.game import Game
from training.network import AlphaZeroNet


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        set_seeds(42)
        self.config = Config(
            device='cpu',
            simulations=5,
            games_per_iteration=1,
            train_steps=1,
            eval_frequency=100,
            checkpoint_frequency=2,
            num_workers=1
        )
    
    @patch('training.self_play.Pool')
    def test_full_training_workflow(self, mock_pool):
        """Test complete end-to-end training workflow"""
        mock_pool.return_value.__enter__.return_value.map.return_value = [[]]
        trainer = Trainer(self.config)
        
        # Full training iteration
        results = trainer.train_iteration(0)
        expected_keys = ['iteration', 'loss', 'buffer_size', 'memory_usage', 'lr']
        for key in expected_keys:
            self.assertIn(key, results)
        
        self.assertIsInstance(results['loss'], float)
        
        # Multi-iteration training
        for i in range(3):
            trainer.train_iteration(i)
        
        # Checkpoint integration
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            trainer.save_checkpoint(2, checkpoint_path)
            new_trainer = Trainer(self.config)
            loaded_iteration = new_trainer.load_checkpoint(checkpoint_path)
            
            self.assertEqual(loaded_iteration, 2)
            self.assertEqual(len(new_trainer.buffer), len(trainer.buffer))
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
    
    def test_evaluation_pipeline(self):
        """Test evaluation components"""
        game = Game()
        net1 = AlphaZeroNet(self.config)
        net2 = AlphaZeroNet(self.config)
        
        # Single game evaluation
        from training.device_manager import DeviceManager
        device_manager = DeviceManager(self.config)
        result = play_game(game, net1, net2, self.config, device_manager)
        self.assertIn(result, [-1, 0, 1])
        
        # Network comparison
        wins, losses, draws = evaluate_networks(game, net1, net2, self.config, device_manager, num_games=2)
        self.assertEqual(wins + losses + draws, 2)
        
        # Training evaluation
        metrics = evaluate_training(game, net1, net2, num_games=5)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_games', metrics)
        self.assertEqual(metrics['total_games'], 5)
    
    def test_error_recovery_and_robustness(self):
        """Test error handling and system robustness"""
        trainer = Trainer(self.config)
        
        # Empty buffer handling
        loss = trainer.train_step()
        self.assertEqual(loss, 0.0)
        
        # Memory cleanup
        trainer.device_manager.cleanup_memory()
        
        # Seed consistency
        set_seeds(42)
        state1 = torch.get_rng_state()
        set_seeds(42)
        state2 = torch.get_rng_state()
        self.assertTrue(torch.equal(state1, state2))


if __name__ == '__main__':
    unittest.main()
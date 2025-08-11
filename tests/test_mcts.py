import unittest
import torch
import numpy as np
from training.game import Game
from training.mcts import MCTS
from training.config import Config
from training.device_manager import DeviceManager
from training.network import AlphaZeroNet


class TestMCTS(unittest.TestCase):
    
    def setUp(self):
        self.config = Config(device='cpu', simulations=50)
        self.game = Game()
        self.net = AlphaZeroNet(self.config)
        self.device_manager = DeviceManager(self.config)
        self.mcts = MCTS(self.game, self.net, self.config, self.device_manager)
    
    def test_mcts_policy_generation(self):
        """Merged: basic MCTS run, policy properties, state conversion"""
        state = self.game.get_initial_state()
        policy = self.mcts.run(state)
        
        # Policy properties
        self.assertEqual(len(policy), 9)
        self.assertAlmostEqual(policy.sum(), 1.0, places=5)
        self.assertTrue(np.all(policy >= 0))
        
        # Legal moves have non-zero probability
        legal_moves = self.game.get_legal_moves(state)
        legal_policy = policy[legal_moves]
        self.assertTrue(np.any(legal_policy > 0))
        
        # State to tensor conversion
        tensor = self.mcts.state_to_tensor(state)
        self.assertEqual(tensor.shape, (2, 3, 3))
        self.assertEqual(tensor.dtype, torch.float32)
    
    def test_temperature_effects(self):
        """Merged: temperature sampling, entropy comparison"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        state = self.game.get_initial_state()
        
        # High temperature (exploration)
        mcts_high = MCTS(self.game, self.net, 
                        Config(device='cpu', simulations=50, temperature=2.0), 
                        self.device_manager)
        policy_high = mcts_high.run(state)
        
        # Low temperature (exploitation)
        mcts_low = MCTS(self.game, self.net, 
                       Config(device='cpu', simulations=50, temperature=0.1), 
                       self.device_manager)
        policy_low = mcts_low.run(state)
        
        # High temperature should have higher entropy
        entropy_high = -np.sum(policy_high * np.log(policy_high + 1e-8))
        entropy_low = -np.sum(policy_low * np.log(policy_low + 1e-8))
        
        self.assertGreater(entropy_high, entropy_low)
    
    def test_device_integration_and_config(self):
        """Merged: device integration, batch evaluation flags, initialization"""
        # MCTS initialization
        self.assertEqual(self.mcts.simulations, 50)
        self.assertEqual(self.mcts.c_puct, 1.5)
        self.assertEqual(self.mcts.temperature, 1.0)
        self.assertFalse(self.mcts.batch_evaluation)  # CPU config
        
        # Device integration
        state = self.game.get_initial_state()
        tensor = self.mcts.state_to_tensor(state)
        device_tensor = self.device_manager.to_device(tensor)
        self.assertEqual(device_tensor.device.type, 'cpu')
        
        # GPU should use batch evaluation (mock without actual CUDA)
        gpu_config = Config(device='cpu')  # Use CPU to avoid CUDA warnings
        gpu_config.device = 'cuda'  # Set after init to avoid auto-detection
        gpu_mcts = MCTS(self.game, self.net, gpu_config, self.device_manager)
        self.assertTrue(gpu_mcts.batch_evaluation)


if __name__ == '__main__':
    unittest.main()
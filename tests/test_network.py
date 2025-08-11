import unittest
import torch
from training.config import Config
from training.network import AlphaZeroNet, ResidualBlock


class TestNetwork(unittest.TestCase):
    
    def setUp(self):
        self.config = Config(device='cpu')
        self.net = AlphaZeroNet(self.config)
    
    def test_network_architecture_and_shapes(self):
        """Merged: output shapes, forward pass, architecture validation"""
        batch_size = 2
        dummy_input = torch.zeros(batch_size, 2, 3, 3)
        
        log_p, v = self.net(dummy_input)
        
        # Output shapes
        self.assertEqual(log_p.shape, (batch_size, 9))
        self.assertEqual(v.shape, (batch_size,))
        
        # Policy probabilities
        probs = torch.exp(log_p)
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5))
        
        # Value range
        self.assertTrue(torch.all(v >= -1) and torch.all(v <= 1))
        
        # Network properties
        self.assertEqual(self.net.board_size, 3)
        self.assertEqual(self.net.action_size, 9)
        self.assertFalse(self.net.use_checkpointing)  # CPU config
    
    def test_weight_initialization_and_components(self):
        """Merged: weight initialization, residual blocks, config variations"""
        # Weight initialization
        for module in self.net.modules():
            if isinstance(module, torch.nn.Conv2d):
                self.assertFalse(torch.allclose(module.weight, torch.zeros_like(module.weight)))
            elif isinstance(module, torch.nn.Linear):
                self.assertTrue(torch.allclose(module.bias, torch.zeros_like(module.bias)))
        
        # Residual block
        block = ResidualBlock(64)
        x = torch.randn(1, 64, 3, 3)
        out = block(x)
        self.assertEqual(out.shape, x.shape)
        
        # Different config sizes
        small_config = Config(device='cpu', filters=64, blocks=2)
        small_net = AlphaZeroNet(small_config)
        
        large_config = Config(device='cpu', filters=256, blocks=8)
        large_net = AlphaZeroNet(large_config)
        
        small_params = sum(p.numel() for p in small_net.parameters())
        large_params = sum(p.numel() for p in large_net.parameters())
        self.assertLess(small_params, large_params)
    
    def test_gpu_optimizations(self):
        """Test GPU-specific optimizations"""
        gpu_config = Config()
        gpu_config.device = 'cuda'
        gpu_config.blocks = 10
        
        net = AlphaZeroNet(gpu_config)
        self.assertTrue(net.use_checkpointing)


if __name__ == '__main__':
    unittest.main()
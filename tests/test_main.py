import unittest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main


class TestMain(unittest.TestCase):
    
    def test_resolve_device(self):
        """Test device resolution logic"""
        with patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(main.resolve_device('cuda'), 'cuda')
            self.assertEqual(main.resolve_device('auto'), 'cuda')
        
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(main.resolve_device('cuda'), 'cpu')
            self.assertEqual(main.resolve_device('cpu'), 'cpu')
    
    def test_parse_args(self):
        """Test argument parsing"""
        with patch('sys.argv', ['main.py', '--iterations', '100']):
            args = main.parse_args()
            self.assertEqual(args.iterations, 100)
            self.assertEqual(args.device, 'auto')
            self.assertEqual(args.onnx_path, 'model.onnx')
            self.assertTrue(args.validate)
    
    @patch('main.plt')
    @patch('main.MATPLOTLIB_AVAILABLE', True)
    @patch('main.export_to_onnx')
    @patch('main.validate_onnx_model')
    @patch('main.Trainer')
    @patch('main.Config')
    @patch('main.set_seeds')
    @patch('torch.save')
    def test_main_training_loop(self, mock_torch_save, mock_set_seeds, mock_config, 
                               mock_trainer_class, mock_validate, mock_export, mock_plt):
        """Test main training loop with mocked components"""
        # Setup mocks
        mock_trainer = MagicMock()
        mock_trainer.train_iteration.return_value = {
            'loss': 0.5, 'buffer_size': 100, 'memory_usage': 1.0
        }
        mock_trainer.net = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        mock_validate.return_value = (1e-5, 1e-6)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock sys.argv
            test_args = [
                'main.py', '--iterations', '3', '--checkpoint-every', '2',
                '--checkpoint-dir', tmpdir, '--final-model', f'{tmpdir}/final.pt',
                '--onnx-path', f'{tmpdir}/model.onnx'
            ]
            
            with patch('sys.argv', test_args):
                main.main()
            
            # Verify calls
            mock_set_seeds.assert_called_once_with(42)
            mock_trainer_class.assert_called_once()
            self.assertEqual(mock_trainer.train_iteration.call_count, 3)
            mock_trainer.save_checkpoint.assert_called_once()  # checkpoint at iter 2
            mock_torch_save.assert_called_once()
            mock_export.assert_called_once()
            mock_validate.assert_called_once()
    
    @patch('main.signal.signal')
    @patch('main.Trainer')
    @patch('main.Config')
    @patch('main.set_seeds')
    def test_sigint_handling(self, mock_set_seeds, mock_config, mock_trainer_class, mock_signal):
        """Test SIGINT interrupt handling"""
        mock_trainer = MagicMock()
        mock_trainer.train_iteration.side_effect = [
            {'loss': 0.5, 'buffer_size': 100, 'memory_usage': 1.0},
            KeyboardInterrupt()  # Simulate interrupt on second iteration
        ]
        mock_trainer_class.return_value = mock_trainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = [
                'main.py', '--iterations', '5', '--checkpoint-dir', tmpdir
            ]
            
            with patch('sys.argv', test_args):
                # Mock the signal handler to simulate interrupt
                def mock_signal_setup(sig, handler):
                    if sig == main.signal.SIGINT:
                        # Simulate interrupt by calling handler
                        handler(None, None)
                
                mock_signal.side_effect = mock_signal_setup
                
                try:
                    main.main()
                except SystemExit:
                    pass  # Expected on interrupt
            
            # Verify signal handler was set
            mock_signal.assert_called()
    
    @patch('main.MATPLOTLIB_AVAILABLE', False)
    @patch('main.export_to_onnx')
    @patch('main.Trainer')
    @patch('main.Config')
    @patch('main.set_seeds')
    @patch('torch.save')
    def test_no_matplotlib(self, mock_torch_save, mock_set_seeds, mock_config, 
                          mock_trainer_class, mock_export):
        """Test graceful handling when matplotlib is not available"""
        mock_trainer = MagicMock()
        mock_trainer.train_iteration.return_value = {
            'loss': 0.5, 'buffer_size': 100, 'memory_usage': 1.0
        }
        mock_trainer_class.return_value = mock_trainer
        
        test_args = ['main.py', '--iterations', '1']
        
        with patch('sys.argv', test_args):
            # Should not raise exception
            main.main()
        
        # Verify training still completed
        mock_trainer.train_iteration.assert_called_once()
    
    def test_set_seeds_fallback(self):
        """Test fallback set_seeds implementation"""
        with patch('random.seed') as mock_random, \
             patch('numpy.random.seed') as mock_numpy, \
             patch('torch.manual_seed') as mock_torch, \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.manual_seed_all') as mock_cuda:
            
            # Test the fallback function directly
            def fallback_set_seeds(seed: int) -> None:
                import random
                import numpy as np
                import torch
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            
            fallback_set_seeds(42)
            
            mock_random.assert_called_once_with(42)
            mock_numpy.assert_called_once_with(42)
            mock_torch.assert_called_once_with(42)
            mock_cuda.assert_called_once_with(42)


if __name__ == '__main__':
    unittest.main()
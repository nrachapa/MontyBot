import unittest
import tempfile
import os
import torch
import numpy as np
from training.config import Config
from training.network import AlphaZeroNet
from training.onnx_export import export_to_onnx, validate_onnx_model
from training.ort_inference import ORTInferenceEngine
from training.quantization import create_calibration_dataset, quantize_model
from training.game import Game


class TestONNXORT(unittest.TestCase):
    
    def setUp(self):
        # Create small test network
        self.config = Config(
            device='cpu',
            board_size=3,
            action_size=9,
            filters=32,
            blocks=2
        )
        self.net = AlphaZeroNet(self.config)
        self.game = Game(board_size=3)
    
    def test_onnx_export(self):
        """Test ONNX export creates valid file"""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            export_to_onnx(self.net, onnx_path, batch_size=1)
            self.assertTrue(os.path.exists(onnx_path))
            
            # Test file loads without error
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
    
    def test_onnx_parity(self):
        """Test PyTorch vs ONNX output parity"""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            export_to_onnx(self.net, onnx_path, batch_size=1)
            policy_diff, value_diff = validate_onnx_model(onnx_path, self.net, atol=2e-3)
            
            self.assertLess(policy_diff, 2e-3)
            self.assertLess(value_diff, 2e-3)
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
    
    def test_ort_engine(self):
        """Test ORT engine produces correct shapes"""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            export_to_onnx(self.net, onnx_path, batch_size=8)
            engine = ORTInferenceEngine(onnx_path, batch_size=8)
            
            # Test single prediction with correct batch size
            test_input = np.random.randn(8, 2, 3, 3).astype(np.float32)
            policy, value = engine.predict(test_input)
            
            self.assertEqual(policy.shape, (8, 9))
            self.assertEqual(value.shape, (8,))
            
            # Test batch prediction
            states = [np.random.randn(2, 3, 3).astype(np.float32) for _ in range(10)]
            batch_policy, batch_value = engine.predict_batch(states)
            
            self.assertEqual(batch_policy.shape, (10, 9))
            self.assertEqual(batch_value.shape, (10,))
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
    
    def test_deterministic_inference(self):
        """Test inference is deterministic under fixed seeds"""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            torch.manual_seed(42)
            np.random.seed(42)
            
            export_to_onnx(self.net, onnx_path, batch_size=1)
            engine = ORTInferenceEngine(onnx_path, batch_size=1)
            
            test_input = np.random.randn(1, 2, 3, 3).astype(np.float32)
            
            # Run twice
            policy1, value1 = engine.predict(test_input)
            policy2, value2 = engine.predict(test_input)
            
            np.testing.assert_array_equal(policy1, policy2)
            np.testing.assert_array_equal(value1, value2)
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
    
    def test_calibration_dataset(self):
        """Test calibration dataset generation"""
        calibration_data = create_calibration_dataset(self.game, num_samples=10)
        
        self.assertEqual(len(calibration_data), 10)
        for sample in calibration_data:
            self.assertEqual(sample.shape, (2, 3, 3))
            self.assertEqual(sample.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
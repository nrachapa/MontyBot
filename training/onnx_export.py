import io
import os
import logging
from typing import Tuple
import torch
from .network import AlphaZeroNet
from .config import Config

logger = logging.getLogger(__name__)

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX/ORT not available. Install with: pip install onnx onnxruntime")


def export_to_onnx(model: AlphaZeroNet, filepath: str, batch_size: int = 1) -> None:
    """Export PyTorch model to ONNX with static NCHW shapes and opset>=17."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
    
    model.eval()
    
    # Create parent directories
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Infer input channels from model
    input_channels = model.conv.in_channels
    example_input = torch.randn(batch_size, input_channels, model.board_size, model.board_size)
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            filepath,
            opset_version=17,
            do_constant_folding=True,
            input_names=["board"],
            output_names=["policy", "value"],
            dynamic_axes=None  # Static shapes only
        )
    
    # Validate exported model
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX model exported and validated: {filepath}")


def validate_onnx_model(onnx_path: str, pytorch_model: AlphaZeroNet, atol: float = 1e-4) -> Tuple[float, float]:
    """Run same random input through PyTorch and ONNXRuntime; return (max_abs_diff_policy, max_abs_diff_value)."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
    
    pytorch_model.eval()
    
    # Create test input
    input_channels = pytorch_model.conv.in_channels
    test_input = torch.randn(1, input_channels, pytorch_model.board_size, pytorch_model.board_size)
    
    # PyTorch inference
    with torch.no_grad():
        torch_policy, torch_value = pytorch_model(test_input)
    
    # ONNX Runtime inference
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_outputs = sess.run(None, {"board": test_input.numpy()})
    onnx_policy, onnx_value = ort_outputs
    
    # Calculate differences
    policy_diff = float(torch.max(torch.abs(torch_policy - torch.from_numpy(onnx_policy))))
    value_diff = float(torch.max(torch.abs(torch_value - torch.from_numpy(onnx_value))))
    
    assert policy_diff < atol, f"Policy diff {policy_diff} exceeds tolerance {atol}"
    assert value_diff < atol, f"Value diff {value_diff} exceeds tolerance {atol}"
    
    logger.info(f"Validation passed - Policy diff: {policy_diff:.6f}, Value diff: {value_diff:.6f}")
    return policy_diff, value_diff
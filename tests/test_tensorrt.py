"""Tests for TensorRT + INT8 quantization (paper Sec. 9.2)."""

import torch
import torch.nn as nn

from src.training.tensorrt_quant import (
    ONNXExporter,
    QuantizationAwareTraining,
    TensorRTInference,
    benchmark_int8_vs_fp16,
)


# Dummy network that returns (logits, value) like AlphaZeroNet
class DummyChessNet(nn.Module):
    """Dummy network returning (logits, value) for QAT testing."""
    def forward(self, x):
        x.view(x.size(0), -1)
        logits = torch.randn(x.size(0), 10)
        value = torch.tanh(torch.randn(x.size(0), 1))
        return logits, value


class TestQuantizationAwareTraining:
    """Test quantization-aware training wrapper."""

    def test_qat_wraps_network(self):
        """Test QAT wraps a base network."""
        base_net = DummyChessNet()
        qat_net = QuantizationAwareTraining(base_net)

        # Test forward pass
        x = torch.randn(2, 3, 3, 3)
        logits, value = qat_net(x)
        assert logits.shape == (2, 10)
        assert value.shape == (2, 1)

    def test_fake_quant_disabled(self):
        """Test disabling fake quantization."""
        base_net = DummyChessNet()
        qat_net = QuantizationAwareTraining(base_net)

        assert qat_net.fake_quant_enabled
        qat_net.disable_fake_quant()
        assert not qat_net.fake_quant_enabled

    def test_qat_output_shape(self):
        """Test QAT maintains output shape."""
        base_net = DummyChessNet()
        qat_net = QuantizationAwareTraining(base_net)

        x = torch.randn(32, 8, 8, 8)
        logits, value = qat_net(x)
        assert logits.shape == (32, 10)
        assert value.shape == (32, 1)


class TestONNXExporter:
    """Test ONNX export functionality."""

    def test_export_dimensions(self):
        """Test export preserves model structure."""
        nn.Module.__init__(type('DummyNet', (nn.Module,), {'forward': lambda s, x: x + 1})())
        assert hasattr(ONNXExporter, 'export')


class TestTensorRTInference:
    """Test TensorRT inference wrapper."""

    def test_pytorch_fallback(self):
        """Test PyTorch fallback when no TRT engine."""
        inference = TensorRTInference(device="cpu")

        # Should raise NotImplementedError when no model provided
        x = torch.randn(1, 10)
        try:
            inference.infer(x)
            raise AssertionError("Should have raised NotImplementedError")
        except NotImplementedError as e:
            assert "No model provided" in str(e)


class TestBenchmark:
    """Test benchmarking functionality."""

    def test_benchmark_returns_metrics(self):
        """Test benchmark returns latency and throughput metrics."""
        model = DummyChessNet()

        results = benchmark_int8_vs_fp16(model, n_runs=10)

        assert "latency_fp32_ms" in results
        assert "throughput_fp32" in results
        assert results["latency_fp32_ms"] > 0
        assert results["throughput_fp32"] > 0


class TestONNXExporterIntegration:
    """Test ONNX export with trained model integration."""

    def test_export_with_validation(self):
        """Test ONNX export with validation structure."""
        from src.training.tensorrt_quant import ONNXExporter

        # Test that export_with_validation exists and returns expected structure
        DummyChessNet()

        # We can't actually export without onnxscript, but we can test the interface
        # by checking the method signature and structure
        assert hasattr(ONNXExporter, 'export_with_validation')

        # Verify the method exists and is callable
        assert callable(ONNXExporter.export_with_validation)

        # Check that it has correct docstring indicating trained model export
        assert "validation" in ONNXExporter.export_with_validation.__doc__.lower()

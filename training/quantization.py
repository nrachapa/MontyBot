import logging
from typing import Iterable, List
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnx
    import onnxruntime as ort
    import onnxruntime.quantization as ort_quant
    from onnxruntime.quantization import QuantType, CalibrationDataReader
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    logger.warning("ONNX quantization not available. Install with: pip install onnx onnxruntime")


if QUANTIZATION_AVAILABLE:
    class BoardCalibrationDataReader(CalibrationDataReader):
        def __init__(self, calibration_data: List[np.ndarray]):
            self.data = calibration_data
            self.data_index = 0
        
        def get_next(self):
            if self.data_index >= len(self.data):
                return None
            
            input_data = {"board": self.data[self.data_index]}
            self.data_index += 1
            return input_data
else:
    class BoardCalibrationDataReader:
        def __init__(self, calibration_data: List[np.ndarray]):
            pass


def quantize_model(model_path: str, quantized_path: str, calibration_data: Iterable[np.ndarray],
                   preserve_value_head: bool = True) -> None:
    """INT8 QDQ PTQ with per-tensor affine; optionally keep value head FP16/FP32 by op/type filter."""
    if not QUANTIZATION_AVAILABLE:
        raise ImportError("ONNX quantization not available. Install with: pip install onnx onnxruntime")
    
    calibration_list = list(calibration_data)
    calibration_reader = BoardCalibrationDataReader(calibration_list)
    
    # Configure quantization
    extra_options = {}
    nodes_to_exclude = []
    
    if preserve_value_head:
        # Load model to find value head nodes
        model = onnx.load(model_path)
        for node in model.graph.node:
            if "value" in node.name.lower() or any("value" in output for output in node.output):
                nodes_to_exclude.append(node.name)
        
        if nodes_to_exclude:
            extra_options["nodes_to_exclude"] = nodes_to_exclude
            logger.info(f"Excluding value head nodes from quantization: {nodes_to_exclude}")
    
    # Quantize model
    ort_quant.quantize_static(
        model_input=model_path,
        model_output=quantized_path,
        calibration_data_reader=calibration_reader,
        quant_format=ort_quant.QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        extra_options=extra_options
    )
    
    # Validate quantized model
    quantized_model = onnx.load(quantized_path)
    onnx.checker.check_model(quantized_model)
    logger.info(f"Quantized model created and validated: {quantized_path}")


def create_calibration_dataset(game, num_samples: int = 1000) -> List[np.ndarray]:
    """Generate N random legal positions -> np.float32 (C, H, W)."""
    calibration_data = []
    
    for _ in range(num_samples):
        state = game.get_initial_state()
        
        # Make some random moves to create diverse positions
        for _ in range(np.random.randint(0, 5)):
            if not game.is_terminal(state):
                legal_moves = game.get_legal_moves(state)
                if legal_moves:
                    move = np.random.choice(legal_moves)
                    state = game.apply_move(state, move)
        
        # Convert state to tensor format
        board, player = state
        current = (board == player).astype(np.float32)
        opponent = (board == -player).astype(np.float32)
        board_tensor = np.stack([current, opponent], axis=0)
        
        calibration_data.append(board_tensor)
    
    logger.info(f"Generated {len(calibration_data)} calibration samples")
    return calibration_data
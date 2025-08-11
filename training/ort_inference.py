import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("ORT not available. Install with: pip install onnxruntime")


class ORTInferenceEngine:
    def __init__(self, onnx_path: str, batch_size: int = 8, intra_op_threads: int = 8, inter_op_threads: int = 1):
        """Initialize ORT session with oneDNN, extended graph optimizations, and static input shape."""
        if not ORT_AVAILABLE:
            raise ImportError("ORT not available. Install with: pip install onnxruntime")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.inter_op_num_threads = inter_op_threads
        
        providers = [("CPUExecutionProvider", {"use_arena": 1})]
        
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self.batch_size = batch_size
        
        # Cache input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape for pre-allocation
        input_shape = self.session.get_inputs()[0].shape
        self.input_channels = input_shape[1]
        self.board_size = input_shape[2]
        
        # Pre-allocate buffers
        self.input_buffer = np.zeros((batch_size, self.input_channels, self.board_size, self.board_size), dtype=np.float32)
        
        logger.info(f"ORT engine initialized - batch_size: {batch_size}, threads: {intra_op_threads}")
    
    def predict(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run a single batch: inputs shape (B, C, H, W) -> (policy_logits, value)."""
        inputs = np.ascontiguousarray(inputs, dtype=np.float32)
        outputs = self.session.run(self.output_names, {self.input_name: inputs})
        return outputs[0], outputs[1]  # policy, value
    
    def predict_batch(self, states_batch: List[np.ndarray], micro_batch: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Micro-batch a list of states efficiently; pad final slice if needed and trim output."""
        if micro_batch is None:
            micro_batch = self.batch_size
        
        # Ensure micro_batch doesn't exceed the model's expected batch size
        micro_batch = min(micro_batch, self.batch_size)
        
        all_policies = []
        all_values = []
        
        for i in range(0, len(states_batch), micro_batch):
            batch_slice = states_batch[i:i + micro_batch]
            actual_batch_size = len(batch_slice)
            
            # Prepare input batch
            if actual_batch_size == micro_batch:
                batch_input = np.stack(batch_slice, axis=0)
            else:
                # Pad final batch to maintain static shape
                batch_input = np.zeros((micro_batch, self.input_channels, self.board_size, self.board_size), dtype=np.float32)
                for j, state in enumerate(batch_slice):
                    batch_input[j] = state
            
            # Run inference
            policy_batch, value_batch = self.predict(batch_input)
            
            # Trim to actual batch size
            all_policies.append(policy_batch[:actual_batch_size])
            all_values.append(value_batch[:actual_batch_size])
        
        return np.concatenate(all_policies, axis=0), np.concatenate(all_values, axis=0)
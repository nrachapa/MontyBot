import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from training.model.net import SimpleNNUE

def load_model(model_path):
    model = SimpleNNUE()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, 768)
    torch.onnx.export(model, dummy_input, os.path.join(os.path.dirname(model_path), "latest_model.onnx"), input_names=["input"], output_names=['output'], verbose=True)
    return

if __name__ == "__main__":
    model_pth = 'model.pth'
    load_model(model_pth)


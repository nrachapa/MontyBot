#!/usr/bin/env python3
"""Benchmark ORT vs PyTorch inference performance."""

import argparse
import time
import logging
from pathlib import Path
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.config import Config
from training.network import AlphaZeroNet
from training.ort_inference import ORTInferenceEngine
from training.game import Game

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_data(num_samples: int, board_size: int = 3) -> list:
    """Generate random test states."""
    game = Game(board_size=board_size)
    states = []
    
    for _ in range(num_samples):
        state = game.get_initial_state()
        # Make random moves
        for _ in range(np.random.randint(0, 3)):
            if not game.is_terminal(state):
                legal_moves = game.get_legal_moves(state)
                if legal_moves:
                    move = np.random.choice(legal_moves)
                    state = game.apply_move(state, move)
        
        board, player = state
        current = (board == player).astype(np.float32)
        opponent = (board == -player).astype(np.float32)
        board_tensor = np.stack([current, opponent], axis=0)
        states.append(board_tensor)
    
    return states


def benchmark_pytorch(model, states, batch_sizes, device_manager):
    """Benchmark PyTorch inference."""
    results = {}
    model.eval()
    
    for batch_size in batch_sizes:
        times = []
        
        for i in range(0, len(states), batch_size):
            batch = states[i:i + batch_size]
            if len(batch) < batch_size:
                continue
                
            batch_tensor = torch.stack([torch.from_numpy(s) for s in batch])
            batch_tensor = device_manager.to_device(batch_tensor)
            
            start_time = time.perf_counter()
            with torch.no_grad():
                with device_manager.autocast_context():
                    _ = model(batch_tensor)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        if times:
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            results[batch_size] = throughput
            logger.info(f"PyTorch batch_size={batch_size}: {throughput:.1f} pos/sec")
    
    return results


def benchmark_ort(onnx_path, states, batch_sizes, threads, inter_threads):
    """Benchmark ORT inference."""
    results = {}
    
    for batch_size in batch_sizes:
        engine = ORTInferenceEngine(
            onnx_path, 
            batch_size=batch_size,
            intra_op_threads=threads,
            inter_op_threads=inter_threads
        )
        
        times = []
        
        for i in range(0, len(states), batch_size):
            batch = states[i:i + batch_size]
            if len(batch) < batch_size:
                continue
            
            batch_array = np.stack(batch, axis=0)
            
            start_time = time.perf_counter()
            _ = engine.predict(batch_array)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        if times:
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            results[batch_size] = throughput
            logger.info(f"ORT batch_size={batch_size}: {throughput:.1f} pos/sec")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark ORT vs PyTorch')
    parser.add_argument('--onnx', required=True, help='Path to ONNX model')
    parser.add_argument('--int8', help='Path to quantized INT8 model')
    parser.add_argument('--threads', type=int, default=8, help='ORT intra-op threads')
    parser.add_argument('--inter', type=int, default=1, help='ORT inter-op threads')
    parser.add_argument('--samples', type=int, default=2048, help='Number of test samples')
    
    args = parser.parse_args()
    
    if not Path(args.onnx).exists():
        logger.error(f"ONNX model not found: {args.onnx}")
        return
    
    # Generate test data
    logger.info(f"Generating {args.samples} test samples...")
    states = generate_test_data(args.samples)
    
    # Test batch sizes
    batch_sizes = [1, 8, 16, 32]
    
    # Create PyTorch model for comparison
    config = Config(device='cpu', board_size=3, filters=32, blocks=2)
    model = AlphaZeroNet(config)
    
    from training.device_manager import DeviceManager
    device_manager = DeviceManager(config)
    
    # Benchmark PyTorch
    logger.info("Benchmarking PyTorch...")
    torch_results = benchmark_pytorch(model, states, batch_sizes, device_manager)
    
    # Benchmark ORT FP32
    logger.info("Benchmarking ORT FP32...")
    ort_results = benchmark_ort(args.onnx, states, batch_sizes, args.threads, args.inter)
    
    # Benchmark ORT INT8 if provided
    int8_results = {}
    if args.int8 and Path(args.int8).exists():
        logger.info("Benchmarking ORT INT8...")
        int8_results = benchmark_ort(args.int8, states, batch_sizes, args.threads, args.inter)
    
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS (positions/second)")
    print("="*80)
    print(f"{'Batch Size':<12} {'PyTorch':<12} {'ORT FP32':<12} {'ORT INT8':<12} {'Speedup':<12}")
    print("-"*80)
    
    for batch_size in batch_sizes:
        torch_perf = torch_results.get(batch_size, 0)
        ort_perf = ort_results.get(batch_size, 0)
        int8_perf = int8_results.get(batch_size, 0)
        speedup = ort_perf / torch_perf if torch_perf > 0 else 0
        
        print(f"{batch_size:<12} {torch_perf:<12.1f} {ort_perf:<12.1f} {int8_perf:<12.1f} {speedup:<12.2f}x")
    
    print("="*80)


if __name__ == '__main__':
    main()
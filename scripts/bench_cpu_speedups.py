#!/usr/bin/env python3
"""Benchmark CPU training speedups - PyTorch vs ONNX Runtime."""

import argparse
import time
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.config import Config
from training.trainer import Trainer
from training.network import AlphaZeroNet
from training.device_manager import DeviceManager
from training.game import Game
from fast_config import get_fast_cpu_config, get_ultra_fast_cpu_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_training_iteration(config: Config, iterations: int = 5) -> dict:
    """Benchmark training iteration speed."""
    trainer = Trainer(config)
    
    # Warm up
    trainer.train_iteration(0)
    
    # Benchmark
    start_time = time.perf_counter()
    
    for i in range(iterations):
        result = trainer.train_iteration(i)
        if result.get('early_stop', False):
            break
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_iter = total_time / iterations
    
    return {
        'total_time': total_time,
        'avg_time_per_iter': avg_time_per_iter,
        'iterations_per_sec': 1.0 / avg_time_per_iter,
        'config_name': getattr(config, 'name', 'unnamed')
    }


def benchmark_inference_speed(config: Config, num_samples: int = 1000) -> dict:
    """Benchmark inference speed for PyTorch vs ONNX."""
    game = Game(config.board_size)
    net = AlphaZeroNet(config)
    device_manager = DeviceManager(config)
    
    # Generate test states
    states = []
    for _ in range(num_samples):
        state = game.get_initial_state()
        states.append(state)
    
    # Benchmark PyTorch
    net.eval()
    torch_times = []
    
    for state in states[:100]:  # Sample for speed
        board, player = state
        current = (board == player).astype('float32')
        opponent = (board == -player).astype('float32')
        import numpy as np
        import torch
        x = torch.from_numpy(np.stack([current, opponent], axis=0)).unsqueeze(0)
        
        start = time.perf_counter()
        with torch.inference_mode():
            _ = net(x)
        end = time.perf_counter()
        torch_times.append(end - start)
    
    torch_avg = sum(torch_times) / len(torch_times)
    torch_throughput = 1.0 / torch_avg
    
    results = {
        'pytorch_avg_time': torch_avg,
        'pytorch_throughput': torch_throughput,
        'onnx_avg_time': None,
        'onnx_throughput': None,
        'speedup': 1.0
    }
    
    # Benchmark ONNX if available
    try:
        from training.onnx_export import export_to_onnx
        from training.ort_inference import ORTInferenceEngine
        
        onnx_path = "benchmark_model.onnx"
        export_to_onnx(net, onnx_path, batch_size=1)
        
        engine = ORTInferenceEngine(onnx_path, batch_size=1, 
                                   intra_op_threads=config.ort_threads)
        
        onnx_times = []
        for state in states[:100]:
            board, player = state
            current = (board == player).astype('float32')
            opponent = (board == -player).astype('float32')
            x = np.stack([current, opponent], axis=0)[np.newaxis, ...]
            
            start = time.perf_counter()
            _ = engine.predict(x)
            end = time.perf_counter()
            onnx_times.append(end - start)
        
        onnx_avg = sum(onnx_times) / len(onnx_times)
        onnx_throughput = 1.0 / onnx_avg
        speedup = torch_avg / onnx_avg
        
        results.update({
            'onnx_avg_time': onnx_avg,
            'onnx_throughput': onnx_throughput,
            'speedup': speedup
        })
        
        # Clean up
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            
    except Exception as e:
        logger.warning(f"ONNX benchmark failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark CPU training speedups')
    parser.add_argument('--iterations', type=int, default=10, help='Training iterations to benchmark')
    parser.add_argument('--inference-samples', type=int, default=1000, help='Inference samples to benchmark')
    parser.add_argument('--threads', type=int, default=os.cpu_count(), help='ORT threads')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CPU TRAINING SPEEDUP BENCHMARK")
    print("="*80)
    print(f"System: {os.cpu_count()} CPU cores")
    print(f"Benchmark: {args.iterations} training iterations")
    print(f"Inference: {args.inference_samples} samples")
    print()
    
    # Benchmark configurations
    configs = [
        ("Baseline", Config()),
        ("Fast CPU", get_fast_cpu_config()),
        ("Ultra Fast", get_ultra_fast_cpu_config()),
    ]
    
    training_results = []
    inference_results = []
    
    for name, config in configs:
        config.name = name
        if hasattr(config, 'ort_threads'):
            config.ort_threads = args.threads
        
        print(f"Benchmarking {name}...")
        
        # Training benchmark
        try:
            train_result = benchmark_training_iteration(config, args.iterations)
            train_result['config_name'] = name
            training_results.append(train_result)
            print(f"  Training: {train_result['avg_time_per_iter']:.3f}s/iter")
        except Exception as e:
            print(f"  Training benchmark failed: {e}")
        
        # Inference benchmark
        try:
            inf_result = benchmark_inference_speed(config, args.inference_samples)
            inf_result['config_name'] = name
            inference_results.append(inf_result)
            print(f"  Inference: {inf_result['pytorch_throughput']:.1f} pos/sec (PyTorch)")
            if inf_result['onnx_throughput']:
                print(f"             {inf_result['onnx_throughput']:.1f} pos/sec (ONNX, {inf_result['speedup']:.1f}x)")
        except Exception as e:
            print(f"  Inference benchmark failed: {e}")
        
        print()
    
    # Print summary
    print("="*80)
    print("TRAINING BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Config':<15} {'Time/Iter':<12} {'Iter/Sec':<12} {'Speedup':<12}")
    print("-"*60)
    
    baseline_time = None
    for result in training_results:
        name = result['config_name']
        time_per_iter = result['avg_time_per_iter']
        iter_per_sec = result['iterations_per_sec']
        
        if baseline_time is None:
            baseline_time = time_per_iter
            speedup = 1.0
        else:
            speedup = baseline_time / time_per_iter
        
        print(f"{name:<15} {time_per_iter:<12.3f} {iter_per_sec:<12.2f} {speedup:<12.1f}x")
    
    print()
    print("="*80)
    print("INFERENCE BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Config':<15} {'PyTorch':<12} {'ONNX':<12} {'Speedup':<12}")
    print("-"*60)
    
    for result in inference_results:
        name = result['config_name']
        pytorch_tput = result['pytorch_throughput']
        onnx_tput = result.get('onnx_throughput', 0)
        speedup = result.get('speedup', 1.0)
        
        onnx_str = f"{onnx_tput:.1f}" if onnx_tput else "N/A"
        speedup_str = f"{speedup:.1f}x" if onnx_tput else "N/A"
        
        print(f"{name:<15} {pytorch_tput:<12.1f} {onnx_str:<12} {speedup_str:<12}")
    
    print("="*80)


if __name__ == '__main__':
    main()
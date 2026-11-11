#!/usr/bin/env python3
"""Final benchmark: CPU only, report all metrics."""
import time

import torch
import numpy as np
from src.config import MontyBotConfig
from src.training.system import Factory, Buffer
from src.training.training import AlphaZeroTraining

def benchmark_cpu(n_iters=5):
    config = MontyBotConfig(
        filters=128, blocks=4, simulations=10,
        games_per_iteration=32, batch_size=64,
        train_steps=2, learning_rate=0.001, mixed_precision=False,
        gen_device="cpu", device="cpu",
    )
    
    net = Factory.create_network("alphazero", config)
    net.eval()
    game = Factory.create_game("chess")
    search = Factory.create_search("vectorized_mcts", game, network=net, config=config)
    search.net_device = torch.device("cpu")
    search.network = net
    
    from src.training.search import ParallelSelfPlay
    sp = ParallelSelfPlay(game, search, "cpu", batch_size=64)
    
    train_net = Factory.create_network("alphazero", config)
    train_net = train_net.to("cpu").train()
    trainer = AlphaZeroTraining(train_net, torch.device("cpu"), config.learning_rate, False)
    buffer = Buffer(100000)
    
    total_sp = 0
    total_tr = 0
    total_games = 0
    losses = []
    
    # Warmup
    print("Warmup...")
    sp.generate_games(1)
    
    for i in range(n_iters):
        t0 = time.time()
        trajs = sp.generate_games(config.games_per_iteration)
        elapsed_sp = time.time() - t0
        n = len(trajs) * config.games_per_iteration
        
        buffer.add_batch(trajs)
        
        t0_tr = time.time()
        for _ in range(config.train_steps):
            batch = buffer.sample(config.batch_size)
            if not batch:
                break
            loss = trainer.train_step(batch)
            losses.append(loss)
        elapsed_tr = time.time() - t0_tr
        
        total_sp += elapsed_sp
        total_tr += elapsed_tr
        total_games += n
    
    avg_sp = total_sp / n_iters
    avg_tr = total_tr / n_iters
    avg_loss = sum(losses) / len(losses)
    
    # Inference speed
    dummy = torch.randn(1, config.input_planes, 8, 8)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            net(dummy)
    elapsed = time.time() - start
    inference_speed = 100 / elapsed
    
    print(f"\n=== CPU Benchmark ({n_iters} iterations, 32 games/iter) ===")
    print(f"Self-play:   {avg_sp:.1f}s ({32/avg_sp:.1f} games/s, {32/avg_sp*60:.0f} games/min)")
    print(f"Training:    {avg_tr:.1f}s ({config.train_steps/avg_tr:.0f} steps/s)")
    print(f"Total iter:  {avg_sp + avg_tr:.1f}s")
    print(f"Inference:   {inference_speed:.0f} forward passes/s")
    print(f"Params:      {sum(p.numel() for p in net.parameters()):,}")
    print(f"Loss:        {avg_loss:.4f}")
    return {
        "games_per_sec": 32/avg_sp,
        "games_per_min": 32/avg_sp*60,
        "inference_per_sec": inference_speed,
        "params": sum(p.numel() for p in net.parameters()),
    }

if __name__ == "__main__":
    print("=== Benchmarking AlphaZero Chess Engine ===\n")
    results = benchmark_cpu(n_iters=2)
    print(f"\n=== Summary ===")
    print(f"Self-play: {results['games_per_sec']:.1f} games/s ({results['games_per_min']:.0f} games/min)")
    print(f"Inference: {results['inference_per_sec']:.0f} forward passes/s (CPU)")
    print(f"Model: {results['params']:,} parameters, 128 filters, 4 blocks")

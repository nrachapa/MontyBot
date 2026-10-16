"""Ticket 01 benchmark: self-play generation throughput on the M6.

Measures, as-is (no redesign):
  A. MPS + default GPU config (filters 256, blocks 10, sims 800): seconds/move, seconds/game
  B. CPU + fast config (filters 64, blocks 2, sims 25): seconds/move per worker, games/hour with 8 workers
  C. Training: seconds/step (batch 32) on MPS and CPU, both net sizes
"""
import time
import torch
from fast_config import get_fast_cpu_config
from training.config import Config
from training.game import Game
from training.mcts import MCTS
from training.network import AlphaZeroNet
from training.device_manager import DeviceManager
from training.self_play import ParallelSelfPlay
from training.replay_buffer import ReplayBuffer

def bench_move_times(device, cfg, n_moves=5):
    net = AlphaZeroNet(cfg).to(torch.device(device))
    net.eval()
    dm = DeviceManager(cfg)
    game = Game()
    mcts = MCTS(game, net, cfg, dm)
    state = game.get_initial_state()
    ts = []
    for _ in range(n_moves):
        if game.is_terminal(state):
            break
        t0 = time.perf_counter()
        mcts.run(state)
        ts.append(time.perf_counter() - t0)
        moves = game.get_legal_moves(state)
        state = game.apply_move(state, moves[len(moves)//2])
    return sum(ts)/len(ts) if ts else float('nan')

def bench_games(cfg, n_games=5):
    game = Game()
    net = AlphaZeroNet(cfg)
    net = net.to(torch.device(cfg.device))  # workaround: BatchedSelfPlay never moves net to device
    net.eval()
    dm = DeviceManager(cfg)
    from training.self_play import BatchedSelfPlay
    sp = BatchedSelfPlay(game, net, ReplayBuffer(cfg.buffer_capacity), cfg, dm)
    t0 = time.perf_counter()
    trajs = sp.play_games(n_games)
    dt = time.perf_counter() - t0
    n_moves = sum(len(t) for t in trajs)
    return dt/n_games, n_moves/n_games

def bench_train(device, cfg, n_steps=20):
    import torch.nn.functional as F
    net = AlphaZeroNet(cfg).to(torch.device(device))
    dm = DeviceManager(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    buf = ReplayBuffer(cfg.batch_size)  # exact capacity: avoids None slots in random.sample
    B = cfg.batch_size
    for _ in range(40):  # fill buffer with dummy data
        import numpy as np
        for _ in range(B):
            b = np.zeros((cfg.board_size, cfg.board_size))
            buf.add((b, 1),
                    np.ones(cfg.action_size, dtype=np.float32)/cfg.action_size, 0.0)
    ts = []
    for _ in range(n_steps):
        import numpy as np
        samples = buf.sample(B)
        t0 = time.perf_counter()
        net.train()
        boards = np.stack([s[0][0] for s in samples])
        players = np.array([s[0][1] for s in samples])
        x = np.stack([boards*players[:, None, None], -boards*players[:, None, None]], axis=1).astype(np.float32)
        x = torch.from_numpy(x).to(device)
        pis_t = torch.from_numpy(np.stack([p for _, p, _ in samples])).float().to(device)
        wins_t = torch.tensor([z for _, _, z in samples], dtype=torch.float32).to(device)
        log_p, v = net(x)
        p = log_p.exp()
        loss = (F.kl_div(log_p, pis_t, reduction='batchmean') + 0.25*F.mse_loss(v, wins_t) + cfg.l2_reg*(p**2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return sum(ts[:len(ts)//2])/max(1, len(ts)//2)  # median-ish of fastest half

print("== A. MPS, default gpu config (f256 b10 s800) ==")
cfg = Config()  # auto -> mps
print(f"device={cfg.device} filters={cfg.filters} blocks={cfg.blocks} sims={cfg.simulations}")
t_move = bench_move_times('mps', cfg)
print(f"mps seconds/move: {t_move:.3f}")
t_game, mv = bench_games(cfg, n_games=4)
print(f"mps seconds/game: {t_game:.1f}  moves/game: {mv:.1f}")

print("== B. CPU, fast config (f64 b2 s25) ==")
cfgc = get_fast_cpu_config()
t_move_c = bench_move_times('cpu', cfgc)
print(f"cpu seconds/move (single worker): {t_move_c:.3f}")
t_game_c, mv_c = bench_games(cfgc, n_games=8)
print(f"cpu seconds/game: {t_game_c:.2f}  moves/game: {mv_c:.1f}")
import os
w = min(os.cpu_count(), 8)
print(f"cpu games/hour @ {w} workers (est, CPU-bound): {w*3600/t_game_c:.0f}")

print("== C. training seconds/step ==")
print(f"mps (f256 b10) batch {cfg.batch_size}: {bench_train('mps', cfg, 15):.3f}")
print(f"cpu (f64 b2)  batch {cfgc.batch_size}: {bench_train('cpu', cfgc, 15):.3f}")

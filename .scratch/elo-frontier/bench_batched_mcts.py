"""Ticket 06 experiment: batched-leaf MCTS on MPS.

Hypothesis (from 01): 1.94 s/move at 800 sims is kernel-launch-bound, not
FLOP-bound, because mcts.run() does one batch-1 net call per leaf.
Fix shape: select all leaves in an iteration, evaluate them in ONE batched
call, then expand + backup. Measure seconds/move at a few net sizes.
"""
import time
import torch
import numpy as np
from math import sqrt

from training.config import Config
from training.game import Game
from training.network import AlphaZeroNet
from training.device_manager import DeviceManager
from training.mcts import Node


def mcts_run_batched(mcts, state, device_str):
    game, net, dm = mcts.game, mcts.network, mcts.device_manager
    c_puct, sims, temp = mcts.c_puct, mcts.simulations, mcts.temperature
    dev = torch.device(device_str)
    root = Node(state)
    for _ in range(sims):
        leaves = []
        for _ in range(32):  # 32 selections per iteration -> one batch
            node = root
            path = [node]
            while node.expanded() and not game.is_terminal(node.state):
                total_N = sqrt(sum(c.visit_count for c in node.children.values()))
                best, best_c = -float('inf'), None
                for c in node.children.values():
                    u = c_puct * c.prior * total_N / (1 + c.visit_count)
                    s = c.Q + u
                    if s > best:
                        best, best_c = s, c
                node, path = best_c, path + [best_c]
            if game.is_terminal(node.state):
                w = game.get_winner(node.state)
                v = 0.0 if w == 0 else (1.0 if w == node.state[1] else -1.0)
                for n in reversed(path):
                    n.visit_count += 1
                    n.total_value += v
                    v = -v
            else:
                leaves.append((node, path))
        if leaves:
            xs = [mcts.state_to_tensor(n.state).unsqueeze(0) for n, _ in leaves]
            with torch.inference_mode():
                with dm.autocast_context():
                    log_p, v = net(torch.cat(xs).to(dev))
            priors, vals = torch.exp(log_p).cpu().numpy(), v.cpu().numpy()
            for (node, path), pr, val in zip(leaves, priors, vals):
                legal = game.get_legal_moves(node.state)
                pm = pr[legal]
                pm = pm / pm.sum() if pm.sum() > 0 else np.ones_like(pm) / len(pm)
                for a, p in zip(legal, pm):
                    node.children[a] = Node(game.apply_move(node.state, a), node, float(p))
                val = float(val)
                for n in reversed(path):
                    n.visit_count += 1
                    n.total_value += val
                    val = -val
    visits = np.zeros(game.board_size * game.board_size, dtype=np.float32)
    for a, c in root.children.items():
        visits[a] = c.visit_count
    if visits.sum() > 0:
        visits = visits ** (1 / max(temp, 1e-9))
        visits /= visits.sum()
    return visits


def bench(filters, blocks, sims, net_sims=12):
    cfg = Config()
    cfg.device = 'mps'
    cfg.filters, cfg.blocks, cfg.simulations = filters, blocks, sims
    net = AlphaZeroNet(cfg).to(torch.device('mps')).eval()
    dm = DeviceManager(cfg)
    game = Game()
    from training.mcts import MCTS
    mcts = MCTS(game, net, cfg, dm)

    for _ in range(2):
        mcts_run_batched(mcts, game.get_initial_state(), 'mps')
    # warmup above uses the same object; baseline uses mcts.run:
    t0 = time.perf_counter()
    for _ in range(net_sims):
        mcts.run(game.get_initial_state())
    t_base = (time.perf_counter() - t0) / net_sims

    mcts2 = MCTS(game, net, cfg, dm)
    for _ in range(2):
        mcts_run_batched(mcts2, game.get_initial_state(), 'mps')
    t0 = time.perf_counter()
    for _ in range(net_sims):
        mcts_run_batched(mcts2, game.get_initial_state(), 'mps')
    t_new = (time.perf_counter() - t0) / net_sims
    print(f"f{filters} b{blocks} s{sims}: unbatched {t_base*1000:7.1f} ms/move   batched32 {t_new*1000:7.1f} ms/move   speedup {t_base/t_new:5.1f}x")


if __name__ == '__main__':
    for f, b in [(64, 2), (128, 4), (256, 10)]:
        for s in [200, 800]:
            bench(f, b, s)

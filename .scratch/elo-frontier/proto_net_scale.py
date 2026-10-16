"""Ticket 04 prototype: which chess net size wins on the M6?

Three candidates, all from scratch, SAME dataset, same budgets:
  A: f64 /b2   (small)
  B: f128/b4   (the repo's default class, scaled to chess)
  C: f256/b8   (big)
Measures: generation cost (ms/move @100 sims), training cost (s/step,
steps/h on MPS), short-training strength proxy (loss), and a win-rate
tiebreak vs the untrained small net.

Reuses proto_chess_mcts for game/representation/MCTS (ADR-0006/0007).
"""
import time
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

spec = importlib.util.spec_from_file_location("proto", ".scratch/elo-frontier/proto_chess_mcts.py")
proto = importlib.util.module_from_spec(spec)
src = open(".scratch/elo-frontier/proto_chess_mcts.py").read().replace("if __name__ == '__main__':", "if False:")
exec(compile(src, "proto_chess_mcts.py", "exec"), proto.__dict__)


class ChessNet(nn.Module):
    def __init__(self, filters, blocks):
        super().__init__()
        self.conv = nn.Conv2d(8, filters, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.blocks = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(filters, filters, 3, padding=1, bias=False),
                          nn.BatchNorm2d(filters), nn.ReLU(),
                          nn.Conv2d(filters, filters, 3, padding=1, bias=False),
                          nn.BatchNorm2d(filters), nn.ReLU())
            for _ in range(blocks)])
        self.pfc = nn.Linear(filters * 8 * 8, 133)
        self.vfc = nn.Linear(8 * 8, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.blocks(F.relu(self.bn(self.conv(x))))
        return F.log_softmax(self.pfc(h.flatten(1)), 1), torch.tanh(self.vfc(h[:, -1].flatten(1)))


def state_tensor(board, turn_sign):
    return torch.from_numpy(proto.planes(board, turn_sign)).permute(2, 0, 1)


def play_game(net, dev, sims, seed=None):
    """Full self-played game; returns (boards, players, policies, winner)."""
    if seed is not None:
        np.random.seed(seed)
    b = chess.Board()
    boards, players, pols = [], [], []
    for _move in range(150):  # ADR-0006: 150-move hard cap
        if b.is_game_over():
            break
        ts = 1 if b.turn == chess.WHITE else -1
        pi = proto.mcts_run(b, net, dev, sims)
        legal = proto.legal_actions(b)
        la = np.array([a for a, m in legal])
        p = pi[la]
        p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
        a = la[np.random.choice(len(la), p=p)]
        m = [mm for aa, mm in legal if aa == a][0]
        boards.append(proto.planes(b, ts))
        players.append(1 if ts > 0 else 0)
        pols.append(pi)
        b.push(m)
    r = proto.proto_score(b)
    winner = 0.0 if r is None else (1.0 if r > 0 else -1.0)
    return boards, players, pols, winner


def gen_dataset(net, dev, n_games, out, sims=100):
    t0 = time.perf_counter()
    buf = []
    for i in range(n_games):
        boards, players, pols, winner = play_game(net, dev, sims)
        for x, pl, pi in zip(boards, players, pols):
            buf.append((x, pi, winner * (1 if pl else -1)))
        print(f"  game {i+1}/{n_games} done ({len(boards)} moves), total {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    np.savez_compressed(out,
                        x=np.stack([r[0] for r in buf]),
                        y=np.stack([r[1] for r in buf]),
                        z=np.array([r[2] for r in buf]))
    print(f"dataset: {len(buf)} moves in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def measure_cost(net, dev, label):
    b = chess.Board()
    # generation: 100 sims, 6 moves
    for _ in range(2):
        proto.mcts_run(b, net, dev, 100)
    t0 = time.perf_counter()
    for _ in range(6):
        proto.mcts_run(b, net, dev, 100)
    g = (time.perf_counter() - t0) / 6
    # training: batch 64, 30 steps
    x = torch.randn(64, 8, 8, 8, device=dev)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(3):
        lp, v = net(x); opt.zero_grad()
        (F.kl_div(lp, torch.randn_like(lp).clamp_min(1e-8), reduction='batchmean')).backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(30):
        lp, v = net(x); opt.zero_grad()
        (F.kl_div(lp, torch.randn_like(lp).clamp_min(1e-8), reduction='batchmean')).backward()
        opt.step()
    s = (time.perf_counter() - t0) / 30
    print(f"{label}: {g*1000:8.1f} ms/move(100s) ~{3600/(g*80):7.0f} games/h   {s*1000:6.1f} ms/step(batch64) {1/s*3600/3600*3600:7.0f} steps/h", flush=True)
    return g, s


def short_train(net, dev, data, seconds=240):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X = torch.from_numpy(data['x']).permute(0, 3, 1, 2)
    Y = torch.from_numpy(data['y'])
    Z = torch.from_numpy(data['z'])
    N = len(X)
    t0 = time.perf_counter()
    steps, first, last = 0, None, None
    while time.perf_counter() - t0 < seconds:
        i = np.random.randint(0, N, 64)
        lp, v = net(X[i].to(dev))
        opt.zero_grad()
        loss = F.kl_div(lp, Y[i].to(dev), reduction='batchmean') + 0.25 * F.mse_loss(v, Z[i].to(dev))
        loss.backward(); opt.step()
        steps += 1
        if first is None:
            first = loss.item()
        last = loss.item()
    return steps, (time.perf_counter() - t0), first, last


def eval_match(net_a, net_b, dev, n_games, sims=30):
    """net_a vs net_b; returns win-rate of net_a (draws 0.5)."""
    wins = 0.0
    for g in range(n_games):
        a_starts = (g % 2) == 0
        b = chess.Board()
        nets = [net_a, net_b]
        idx = 0 if a_starts else 1
        for _move in range(150):  # ADR-0006: 150-move hard cap
            if b.is_game_over():
                break
            ts = 1 if b.turn == chess.WHITE else -1
            pi = proto.mcts_run(b, nets[idx], dev, sims)
            legal = proto.legal_actions(b)
            la = np.array([a for a, m in legal])
            p = pi[la]
            p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
            a = la[np.random.choice(len(la), p=p)]
            m = [mm for aa, mm in legal if aa == a][0]
            b.push(m)
            idx = 1 - idx
        r = proto.proto_score(b)
        if r is None or r == 0.0:
            wins += 0.5
        else:
            a_won = (r > 0) == a_starts
            wins += 1.0 if a_won else 0.0
        print(f"  game {g+1}/{n_games}: {'A' if a_starts else 'B'} first, r={r}", flush=True)
    return wins / n_games


def main():
    dev = 'mps'
    torch.manual_seed(0)
    A = ChessNet(64, 2).to(dev).eval()
    B = ChessNet(128, 4).to(dev).eval()
    C = ChessNet(256, 8).to(dev).eval()

    print("== 1. per-step + generation cost ==")
    for n, l in [(A, 'A f64/b2 '), (B, 'B f128/b4'), (C, 'C f256/b8')]:
        measure_cost(n, dev, l)

    print("== 2. dataset: 8 games via A (30 sims) ==")
    gen_dataset(A, dev, 8, "/tmp/montybot_ds.npz", sims=30)
    data = np.load('/tmp/montybot_ds.npz')
    data = {'x': data['x'].astype(np.float32), 'y': data['y'].astype(np.float32), 'z': data['z'].astype(np.float32)}

    print("== 3. short training (4 min each, same data) ==")
    results = {}
    for n, l in [(A, 'A'), (B, 'B'), (C, 'C')]:
        steps, dt, first, last = short_train(n, dev, data, 240)
        results[l] = (steps, dt, first, last)
        print(f"{l}: {steps} steps in {dt/60:.1f} min ({steps/dt:.0f} steps/h)  loss {first:.3f} -> {last:.3f}", flush=True)

    print("== 4. strength tiebreak: trained B and C vs untrained A (12 games, 20 sims) ==")
    A_eval = ChessNet(64, 2).to(dev).eval()  # untrained reference
    A.train(); A.eval()  # ensure batchnorm eval mode for MCTS
    wb = eval_match(A_eval, B, dev, 12, sims=20)
    print(f"B (f128/b4, trained) win-rate vs untrained A: {wb:.2f}")
    wc = eval_match(A_eval, C, dev, 12, sims=20)
    print(f"C (f256/b8, trained) win-rate vs untrained A: {wc:.2f}")


if __name__ == '__main__':
    main()

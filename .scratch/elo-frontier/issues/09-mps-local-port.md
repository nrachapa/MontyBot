Type: research
Status: blocked
Blocked by: 08
Label: wayfinder:research

# M6/Metal local port + throughput measurement

The branch was written for an Intel MacBook Pro 2019: `device: "auto"` resolves cuda-or-cpu (`src/config.py`), so on this M6 box everything silently runs on CPU. Port to Metal and measure before investing in vectorization (ADR-0007 says MPS is the generation backend).

Scope:
1. Env rebuild: requirements.txt wants torch 2.2 (the old `.venv` is 1.13-era); confirm MPS availability and the branch's own test suite runs.
2. `device` resolution: `"auto"` → mps (torch.cuda check must not shadow it); BatchManager, trainer, eval all follow.
3. `torch.compile` in `Trainer` (wrapped in try/except) — confirm it works on this torch/MPS or remove.
4. Mixed precision on MPS (`autocast device_type="mps"`) vs off — measure.
5. **Measure** unbatched MCTS games/h on MPS vs CPU at the *current* (pre-vectorization) branch scale, median of ≥3, per CODING_STANDARDS.md — this is the baseline ticket 10's win is measured against.

Deliverable: numbers + a note in the map; no architecture changes beyond device plumbing.

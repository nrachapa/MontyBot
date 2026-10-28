---
id: adr-0007
type: architecture-design
title: "ADR-0007: Vectorized MCTS generation on MPS"
status: draft
parent: goal
---

# Vectorized MCTS generation on MPS

Self-play generation uses a fully vectorized MCTS (flat numpy tree, batched PUCT, one batched net call per iteration) running on the MPS GPU via PyTorch. We rejected (a) the original per-leaf Python loop — measured 141 ms/move unbatched on MPS vs 83 ms CPU, i.e. MPS is the *slower* backend there because batch-1 kernel launches and python-chess per-leaf cost dominate; and (b) naive leaf-batching (32 leaves/call) — measured 2-3× slower than unbatched on both 3×3 and real chess, because per-leaf Python cost is backend-independent and dwarfs the batched net call (~1 ms). ORT/CPU is reserved for evaluation (ADR-0005) and terminal play (ADR-0003); MPS-torch is the training-side generation backend. `python-chess` remains the legality source (ADR-0006); its per-position cost must be managed (caching/deferral) inside the vectorized loop.

**Measured 2026-10-28 (ticket 09, 8-plane / 133-action f128/b4 net, torch 2.2.2 on the M6, `bench_ticket09.py`):** unbatched move 190 ms CPU vs 216 ms MPS at 32 sims — the 141/83 above predate the 8-plane encoding; the 1.7× gap has narrowed to 1.14×, so the decision stands (keep generation on CPU) with lower urgency. Training step at batch 64 = 128 ms CPU vs 14.8 ms MPS — **MPS ≈ 8.7× faster**, satisfying ADR-0008's ≥2× criterion, which is what justifies moving training to MPS. `torch.compile` is **not usable on MPS** in torch 2.2 (inductor: `Device mps not supported`), so no compile path exists on the pinned torch line; `torch.autocast(device_type="mps")` is likewise unusable (crashes with `mixed_precision=True`, silent no-op with `False`) — hence the cuda-autocast / no-op-everywhere-else rule in `training.py`.

---
id: adr-0007
type: architecture-design
title: "ADR-0007: Vectorized MCTS generation on MPS"
status: draft
parent: goal
---

# Vectorized MCTS generation on MPS

Self-play generation uses a fully vectorized MCTS (flat numpy tree, batched PUCT, one batched net call per iteration) running on the MPS GPU via PyTorch. We rejected (a) the original per-leaf Python loop — measured 141 ms/move unbatched on MPS vs 83 ms CPU, i.e. MPS is the *slower* backend there because batch-1 kernel launches and python-chess per-leaf cost dominate; and (b) naive leaf-batching (32 leaves/call) — measured 2-3× slower than unbatched on both 3×3 and real chess, because per-leaf Python cost is backend-independent and dwarfs the batched net call (~1 ms). ORT/CPU is reserved for evaluation (ADR-0005) and terminal play (ADR-0003); MPS-torch is the training-side generation backend. `python-chess` remains the legality source (ADR-0006); its per-position cost must be managed (caching/deferral) inside the vectorized loop.

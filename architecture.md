---
id: architecture
type: architecture-design
status: draft
title: "Architecture: local-first self-play training system"
parent: goal
references: [adr-0004, adr-0006, adr-0007, adr-0010]
---

# Architecture

## Topology

```
src/cli.py (click: train / evaluate / config)
 ├─► src/training   local pipeline: self-play → buffer → training → checkpoint
 └─► src/infra      cloud route: SageMaker launch / S3 checkpoints
        └─► src/training (the Trainer runs inside the SageMaker job)

 src/config.py  pydantic config, global CONFIG, device resolution   (leaf)
 src/core.py    ABC interfaces (Game, Network, Search, …)           (leaf)
```

`src/config` and `src/core` are leaves imported by everything above; nothing imports back
down into them from `src/training` or `src/infra`'s internals.

## Device split (the load-bearing config decision)

`CONFIG.device` (`auto` → MPS) drives **training** steps; `CONFIG.gen_device` (`cpu`) drives
**generation**. Unbatched MCTS is slower on MPS than CPU — measured 1.14× (216 vs 190 ms/move,
32 sims, f128/b4, 2026-10-28; the original 1.7× figure predates the 8-plane encoding) — so self-play stays
on CPU until the vectorized loop lands; ORT/CPU is reserved for evaluation (adr-0005) and
terminal play (adr-0003). The split lives only in `src/config`.

## Invariants

- **The ADRs are the design of record**; where branch code diverges, the code gets fixed, not
  the ADR (adr-0010).
- **`python-chess` is the legality source** — never re-derive move generation; an encoding bug
  that falls back silently would poison self-play data (adr-0006).
- **The representation is fixed**: 8 planes, 133 actions; changing either means a from-scratch
  retrain (adr-0006).
- **Local-first**: local training is the default path; the cloud route is an explicit opt-in
  and no credentials live in source (adr-0004, adr-0010 — currently violated, see src).
- **Branches**: `alphazero` is the working branch; `main` is frozen legacy 3×3 tic-tac-toe and
  is never touched (adr-0010).

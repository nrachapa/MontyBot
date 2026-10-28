---
id: adr-0010
type: architecture-design
title: "ADR-0010: Work on the `alphazero` branch; local-first on Apple M6/Metal"
status: draft
parent: goal
---

# 10. Work on the `alphazero` branch; local-first on Apple M6/Metal

- Status: accepted
- Date: 2026-10-15
- Deciders: owner (session correction) + agent

## Context

The decision record ADR-0001…0009 was authored while working on branch `main`, which contains a 3×3 tic-tac-toe `training/` module (Intel-Mac era) with no chess code. The owner confirmed the work belongs on branch **`alphazero`**, which already implements real 8×8 chess in `src/`: python-chess game, 104-plane/4672-action encoding, unbatched MCTS, parallel self-play with a queue-based batch manager, f128/b4 ResNet, Sunfish/alpha-beta heuristics, a click CLI (`train`, `evaluate`), and SageMaker/S3 **cloud** training — written for an Intel MacBook Pro 2019 that had no local GPU, hence `device: "auto"` resolving to CUDA-or-CPU and cloud as the default path. The machine is now an Apple M6 (Metal) with a local GPU.

## Decision

1. **`alphazero` is the working branch**; `main` is frozen legacy (3×3) and stays untouched. The ADR-0001…0009 record and the `.scratch/elo-frontier` map move with it and remain the design of record.
2. **Hardware pivot: Intel 2019 (CPU-only) → Apple M6 (Metal/MPS).** All device logic (`"auto"` → cuda/cpu) must learn MPS: MPS-torch for training-side generation per ADR-0007, ORT/CPU for evaluation (ADR-0005) and terminal play (ADR-0003).
3. **ADR-0004 (strictly local, $0) is reaffirmed and re-achievable.** With a local GPU the branch's cloud-first default no longer serves it: local is the default; the SageMaker/S3 path remains available as an *optional* override, and its **hardcoded AWS account id / Admin role ARN must be externalized to environment variables** (a secret in source is not acceptable either way).
4. **Where branch code diverges from the ADRs, the ADRs are the design of record** and the branch code is the object of the fix: e.g. 4672-action move space with colliding promotion indices vs ADR-0006's unambiguous 133; 8-ply stub trajectories with dummy 0.0 values vs ADR-0008's real games and live values; 1024-entry buffer vs the ~250k ring; vs-random evaluation vs ADR-0005's reference protocol.

## Consequences

- Measurements authored on `main` (the 3×3 code) describe `main`, not this branch; the **chess-scale prototypes** (`.scratch/elo-frontier/proto_chess_mcts.py`, `proto_net_scale.py`) remain the valid cost evidence, since they measured the design on this same machine.
- New build tickets 08–12 in the map apply the ADRs to the existing branch code in dependency order (data correctness → MPS port → vectorized MCTS → eval/UCI → credential hygiene).
- The branch's 27 committed `__pycache__/*.pyc` files are untracked and ignored; ruff/pytest guardrail (`.ruff.toml`, pre-commit, CI) is now active on `src/` and `tests/`.

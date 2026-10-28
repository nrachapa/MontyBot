---
id: src
type: module-design
status: draft
title: "src: entry, configuration, shared interfaces"
parent: architecture
depends-on: [src-training, src-infra]
references: [adr-0003, adr-0005]
---

## Responsibility

The user surface and the shared spine: the click CLI (`train --cloud/--local`, `evaluate`,
`config show/validate/reset`), the global pydantic `MontyBotConfig` (owns device resolution,
net shape defaults f128/b4 · 8 planes · 133 actions, AWS + eval settings), and the ABC
interfaces in `core.py` (`Game`, `Network`, `SearchStrategy`, `SelfPlayStrategy`,
`TrainingStrategy`, `EvaluationStrategy`, manager interfaces) that src-training and src-infra
program against.

## Boundary

- Imports src-training (Trainer) and src-infra (S3Manager, cloud launch); `config.py` and
  `core.py` stay leaves — they must not import from `training/` or `infra/`.
- CLI overrides mutate the global `CONFIG` in place before delegating.

## Known limitations (draft)

- **Credentials in source**: `config.py` hardcodes a real AWS account id (`086325823665`) and
  Admin role ARN as pydantic defaults — must move to environment variables (wayfinder
  ticket 12; violates adr-0004/adr-0010).
- **`make train` targets cloud**; local is `make local_train` — the local-first default is not
  yet the `make` default (ticket 12).
- **`evaluate` is a weak baseline**: argmax-net vs. a random opponent at a 90% threshold.
  The adr-0005 reference protocol (MCTS-vs-reference, 10-point gate, re-anchoring) and the
  adr-0003 UCI command are not implemented (ticket 11).

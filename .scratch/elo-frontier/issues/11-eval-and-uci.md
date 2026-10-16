Type: implementation
Status: blocked
Blocked by: 08, 09
Label: wayfinder:build

# Strength eval per ADR-0005 + UCI terminal play per ADR-0003

Two missing halves of the destination, both on the inference/eval side (ORT/CPU per ADR-0005/0003):

1. **Eval**: `cli.py evaluate` is argmax-net vs **random** with a 90% threshold — uninformative. Implement the ADR-0005 protocol: self-play MCTS-vs-MCTS (or MCTS-vs-reference) against a **fixed reference policy** checkpoint, win% − loss% ≥ 10 points gate, draws = ½, reference re-anchors on promotion.
2. **UCI**: add a `uci` CLI command — UCI protocol loop (position/go/move) over the ORT/CPU net + MCTS so a human can play in the terminal. No GUI (ADR-0003).

Acceptance: `python -m src.cli uci` plays a legal, sane game against a human; `evaluate --reference <ckpt>` emits the ADR-0005 numbers.

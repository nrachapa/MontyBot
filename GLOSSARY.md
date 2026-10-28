---
id: glossary
type: goal-and-requirements
title: MontyBot domain glossary
status: draft
parent: goal
---

# MontyBot

An AlphaZero-style chess engine trained entirely by self-play on local hardware.
This glossary is the shared language for the strength effort.

## Language

### Training

**Self-play**:
A game played by the current policy (MCTS + network) against itself; its moves and positions are the only training data the project consumes.
_Avoid_: training games, rollouts-as-data

**Generation**:
The act of self-playing to produce data (positions, policy targets, game outcomes). Measured in games-per-hour, it is the dominant local cost.
_Avoid_: data collection, self-play (when specifically meaning the production phase)

**Cost curve**:
The project's headline output: measured strength (win-rate vs. a reference) plotted against compute spent (training-hours), so any later effort knows the marginal value of more time.

**Strength**:
Measured as relative self-play win-rate against a fixed reference policy. Not an absolute Elo — no anchor engine exists in this project (ADR-0002).
_Avoid_: Elo, rating

### Play

**The reference policy**:
The fixed older checkpoint that strength is measured against. Each promotion of the best checkpoint re-anchors it (ADR-0005).

**Promotion**:
A checkpoint that clears the strength gate (win% − loss% ≥ 10 points vs. the reference, ADR-0005) and becomes the new reference.
_Avoid_: upgrade, version bump

### Representation

**The representation**:
The 8-plane board encoding the network consumes: 6 piece types × 2 sides + 2 castling-rights planes (ADR-0006). Not the 15-plane AlphaZero convention.
_Avoid_: input planes (standalone), features

**The action space**:
133 actions: 64 quiet + 64 captures + 5 castles (ADR-0006). Fixed for every position so generation can batch.
_Avoid_: moves (for the encoded vector), legal moves

**Play**:
Running a checkpoint as a UCI engine from the terminal (ADR-0003). There is no GUI in this project.
_Avoid_: GUI, frontend

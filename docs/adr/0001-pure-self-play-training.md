---
id: adr-0001
type: architecture-design
title: "ADR-0001: Pure self-play training"
status: draft
parent: goal
---

# Pure self-play training

MontyBot is trained exclusively on self-play data. We deliberately rejected the standard cheap alternative — bootstrapping a base on supervised data from a strong engine (e.g. Stockfish-generated positions) — because the owner wants the model to arise entirely from self-play, not to reach a rating faster. Consequence: the generation phase dominates local compute, so generation throughput is the primary bottleneck the effort must design around, and architecture is treated as a free variable (bigger is fine) only where the cost curve justifies it.

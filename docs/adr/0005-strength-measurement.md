---
id: adr-0005
type: architecture-design
title: "ADR-0005: Strength measurement and reference re-anchoring"
status: draft
parent: goal
---

# Strength measurement and reference re-anchoring

Strength is measured as self-play win-rate of the current checkpoint against a reference checkpoint. Promotion gate: the win% − loss% margin must be ≥ 10 points; on promotion the reference is **re-anchored** to the promoted checkpoint, so every round measures marginal gain and stays informative until the curve truly flattens. We rejected a significance-test gate (one-sided binomial p < 0.05) in favour of the plain 10-point margin — the owner chose the simpler rule; a future reader should not "fix" the gate back into a statistical test without reopening this ADR. Cost-curve plots use (W + ½D) / games for smoothness. Measurement runs on the ORT/CPU inference path (the terminal-play deployment backend) so the MPS GPU stays dedicated to training.

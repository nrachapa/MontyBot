# Network scale: progressive f64/b2 → f128/b4

The network is the one "bigger is better" variable left open, so we measured instead of guessing: three candidate AlphaZero nets at the 8-plane/133-action chess shape (ADR-0006) — **A: 64 filters / 2 residual blocks, B: 128/4, C: 256/8** — all from scratch, same dataset, same budgets, on MPS. Measured (median of repeated runs): generation cost A ≈ 0.5–0.8 s/move, B ≈ 0.66–0.70 s/move, C ≈ 1.27–1.32 s/move (100 sims, unbatched proto); per-step training cost A 3.6–3.9 ms, B 5.8–5.9 ms, C 84–95 ms (batch 64). The strength tiebreak was **unmeasurable at session scale** — an honest zero, not a result: the prototype's 1200-move dataset came from untrained nets, the KL objective went NaN on zero-probability targets, and a 12-game win-rate has no power (both "win-rates" 0.50).

Decision, driven by the cost side per ADR-0008 (generation is the binding constraint; training capacity is 100–1000× what generation produces): **progressive A → B. C is out.**

- **A (f64/b2)** for the early stages: it is the cheapest game, and early data from a weak policy is the least valuable data we will ever buy.
- **B (f128/b4)** for the main run: it costs A nothing measurable in generation (within run-to-run noise) and doubles capacity. "Bigger is better, tradeoffs respected" — B is the biggest net that doesn't spend the day budget.
- **C (f256/b8) rejected**: ~2× slower generation *and* ~20× slower training; in a concurrent pipeline it buys a third of the data while learning a fraction as fast.

Two build flags the prototype surfaced (they are build tasks, not scale questions): (1) the KL objective must be NaN-safe — `F.kl_div(log_softmax(...), target)` diverges once the net assigns ~0 probability where the target is also 0 (the repo's `trainer.py` has the same shape); (2) early self-play data is mostly 150-halfmove-capped shuffles, so ADR-0008's 25→100→250 sim progression matters more to early progress than the net size does.

Proto and full numbers: `.scratch/elo-frontier/proto_net_scale.py` (run log in the session record). If a later effort's cost curve shows B plateauing with compute to spare, the upgrade path is C or beyond — re-run the scale proto at that scale rather than assuming.

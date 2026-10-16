Type: research
Status: blocked
Blocked by: 08, 09, 10, 11
Label: wayfinder:build

# The cost-curve run (the destination itself)

The ≤24h background local training run the Destination asks for: progressive data engine (ADR-0008) + vectorized MCTS (ADR-0007) + progressive net scale A→B (ADR-0009), checkpointed, with the ADR-0005 strength protocol measured at intervals so we can plot **win-rate vs compute** and state how far more time would carry the model.

Prerequisites: 08 (data is learnable), 09 (runs on MPS), 10 (generation is fast enough that 24h buys data, not Python), 11 (we can measure strength).

Deliverable: the final checkpoint, the UCI-playable export, and the cost-curve report (format is the one open build choice).

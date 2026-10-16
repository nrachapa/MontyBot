Type: prototype
Status: resolved
Blocked by: 01, 03, 05, 06, 07
Label: wayfinder:prototype

# Network scale decision

## Question

Which network size/shape wins on this machine under the "bigger is better, tradeoffs respected" rule? Raise the fidelity with a cheap concrete artifact: train a couple of candidate sizes (e.g. current `AlphaZeroNet` vs. a wider/deeper variant) for a short fixed budget each, measure strength-per-hour and per-step cost on MPS, and let the owner react to the numbers. The winner is recorded as a decision with its cost curve.

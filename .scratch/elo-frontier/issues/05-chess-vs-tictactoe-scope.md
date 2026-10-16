Type: grilling
Status: resolved
Blocked by: (none)
Label: wayfinder:grilling

# The "chess" in this repo is tic-tac-toe — what is the real destination?

## Question

`01-generation-throughput` surfaced that `training/game.py` implements 3×3 tic-tac-toe, not chess. Decide: does real 8×8 chess enter the effort, or does the effort target the 3×3 game?

## Answer (resolved 2026-10-12, HITL)

**(a) Real 8×8 chess is in scope.** The owner confirms the destination is a chess model; the 3×3 `Game` is a leftover (the repo was built for an Intel MacBook Pro; current hardware is an M6 with a much stronger GPU). Verified: `chess` is in `requirements.txt` but never imported anywhere in the code.

**Consequences recorded:**
- The "2000 Elo" language in the ADRs keeps its meaning; the cost curve must be re-anchored to chess economics (~200–250 self-played games/day on MPS at current config, before optimization).
- A real `Game` implementation (legal move generation, castling, repetition/draw rules, ~15 input planes) joins the map — see **06-chess-game-implementation**.
- The 3×3 game code is legacy: kept, not deleted, but out of scope for this effort.

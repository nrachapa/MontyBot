---
id: goal
type: goal-and-requirements
status: draft
title: "MontyBot — strongest local self-play chess engine"
---

# MontyBot

## What it is

An AlphaZero-style 8×8 chess engine trained entirely by self-play on the owner's local
hardware (Apple M6, MPS). The effort's destination: the strongest model a day-scale local
compute budget can buy — a final checkpoint playable from the terminal via UCI, with the
cost curve (measured strength vs. compute spent) documented so any later effort knows the
marginal value of more time. Confirmed with the owner during spec import.

## Why these constraints

Each is an owner decision recorded as an ADR (all adopted into this graph, `docs/adr/`):

- **Pure self-play only**, no external/supervised data, no pre-trained weights — the model must
  arise entirely from self-play, not reach a rating faster (adr-0001).
- **Strictly local, $0, offline** — no cloud by default; the SageMaker/S3 path is an optional
  override (adr-0004, reaffirmed in adr-0010).
- **Strength = relative self-play win-rate** against a fixed reference policy that re-anchors on
  promotion; no absolute Elo (adr-0002, adr-0005).
- **Play through a terminal UCI bridge**; no custom GUI (adr-0003).

## Capabilities (what the code does)

- **Local self-play training**: MCTS self-play (multi-process) → 250k-move ring buffer →
  ResNet policy/value training, checkpointed to disk (src-training).
- **Real-chess representation**: 8-plane board encoding, 133 actions, `python-chess` legality,
  full games with c_discount value targets (src-training, adr-0006).
- **Optional cloud training**: SageMaker/S3 route behind `train --cloud` (src-infra).
- **Evaluation**: argmax-net vs. a random opponent with a win-rate threshold — a weak
  baseline; the adr-0005 reference protocol is **not** implemented yet.
- **Configuration**: pydantic config with `show` / `validate` / `reset` commands.

## Non-goals

- GUI or any frontend (adr-0003).
- Absolute ratings / anchored Elo arenas (adr-0002).
- External, supervised, or downloaded training data (adr-0001).
- Cloud-first default; local is the default path (adr-0004, adr-0010).
- The 15-plane AlphaZero representation and larger action schemes (adr-0006).

## Durable principles

- The ADR record (0001–0010) is the design of record; where the code diverges, the code is the
  object of the fix (adr-0010).
- Prototypes self-check before spending machine time, and numbers are reported with their noise
  (`CODING_STANDARDS.md`).
- Shared vocabulary lives in `GLOSSARY.md` (glossary) and is used in specs, issues, and tests.

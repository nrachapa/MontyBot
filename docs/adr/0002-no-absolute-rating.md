---
id: adr-0002
type: architecture-design
title: "ADR-0002: No absolute rating"
status: draft
parent: goal
---

# No absolute rating

Strength is measured only by relative self-play win-rate against a fixed reference policy. We rejected a Bradley–Terry arena anchored to engines of known rating (Stockfish at various depths), which would have made claims like "≥2000 Elo" verifiable. The owner chose relative-only measurement, so "2000 Elo" is an unverifiable label for this project: it means "the strongest we can make it", reported as win-rate, not as a number on the Elo scale. If absolute measurement is ever wanted, this ADR must be reopened first.

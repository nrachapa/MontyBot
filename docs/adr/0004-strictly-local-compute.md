---
id: adr-0004
type: architecture-design
title: "ADR-0004: Strictly local compute"
status: draft
parent: goal
---

# Strictly local compute

All training and inference must run on the owner's own hardware (Apple M6, 12 cores, 32 GB, MPS GPU); no cloud, no free-tier services, no downloads of pre-trained weights or data. This was chosen over free-tier GPU services (Colab, Kaggle) that would offer far more compute for $0, because the owner wants the whole loop fully local: $0 and offline. Consequences: wall-clock is the true budget (a day-scale run, background-friendly with a turbo flag, checkpointed to survive crashes), and the M6's GPU is the biggest free lever. Reopening this ADR (allowing cloud) is the single largest strength multiplier available.

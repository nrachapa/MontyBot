---
id: src-infra
type: module-design
status: draft
title: "src/infra: cloud deployment path"
parent: src
depends-on: [src, src-training]
references: [adr-0004, adr-0010]
---

## Responsibility

The optional SageMaker/S3 route: `S3Manager` (checkpoint upload/download, old-job cleanup)
and `sagemaker_manager.py` (launch the PyTorch training job, poll to a terminal state, and —
inside the job container — run the Trainer with CUDA AMP and checkpoint to S3).

## Boundary

- Imports `src.config` (all AWS settings) and `src.training.Trainer`; entered only from
  `src/cli.py` via `train --cloud`.
- The double `main()` entry (launcher vs. in-container training script) is how the same file
  runs both locally and in the SageMaker PyTorch container.

## Known limitations (draft)

- Credentials and account/role come from `config.py` defaults, which hardcode a real AWS
  account id and Admin role ARN — externalization to environment variables is wayfinder
  ticket 12 (adr-0010: "a secret in source is not acceptable either way").

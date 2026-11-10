# AWS Setup Reference

This file is **not loaded by default**. Agents reach here when AWS configuration
or SageMaker launches come up.

## Credentials & Role

```
AWS_ROLE_ARN=arn:aws:iam::086325823665:role/GPUuser_Role
```

Account ID: `086325823665`
Region: `us-west-2` (default), also tested `us-east-1`

## Approved Quotas

| Quota | Value | Notes |
|-------|-------|-------|
| Spot `ml.g4dn.xlarge` | 100 | Approved for SageMaker training jobs |
| On-demand `ml.g4dn.xlarge` | **0** | NOT approved — must request via AWS Support |

No on-demand quota exists in any region (us-west-2, us-east-1, us-east-2, eu-west-1, ap-southeast-1).

## Known Issues

### Spot churn cycle
Spot instances cycle through `Training → Interrupted → Starting → Downloading`.
Each cycle loses ~14 minutes of actual training. The job auto-resumes from S3
checkpoints, so progress is preserved, but throughput is ~10 iterations/24h.

### max_wait_seconds vs max_run_seconds
SageMaker requires `max_wait_seconds >= max_run_seconds`. Default config sets
`max_run=86400` (24h) and `max_wait=14400` (4h) — **this will fail**. The fix:
```python
CONFIG.max_run_seconds = 86400
CONFIG.max_wait_seconds = 172800  # must exceed max_run_seconds
```

### Monitor
Use `scripts/monitor.py <job-name> --region us-west-2` to continuously poll
status, model count, and iteration progress (single line per check instead of
dozens of separate API calls).

### Launch commands
```bash
# Cloud training (spot)
AWS_ROLE_ARN=arn:aws:iam::086325823665:role/GPUuser_Role \
  uv run python -m src.cli train --cloud --on-demand --region us-west-2

# Cloud training (on-demand, after quota approval)
AWS_ROLE_ARN=arn:aws:iam::086325823665:role/GPUuser_Role \
  uv run python -m src.cli train --cloud --on-demand --region us-west-2

# Monitor a job
python scripts/monitor.py montybot-gpu-training-2026-11-07-23-06-13-987 --region us-west-2
```

## Import paths (container vs local)

The container entry point runs with `/opt/ml/code` in `sys.path`, which means
**absolute imports** (`from src.core import X`) work, while **relative imports**
(`from ..core import X`) fail with:
```
ImportError: attempted relative import with no known parent package
```

All container code (`sagemaker_manager.py`) uses `from src.core import X`.

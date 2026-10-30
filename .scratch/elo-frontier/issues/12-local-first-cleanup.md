Type: implementation
Status: done
Blocked by: (independent)
Label: wayfinder:build

# Local-first: credential hygiene + cloud as opt-in

ADR-0004 (reaffirmed in ADR-0010) is strictly local / $0, but the branch's `src/config.py` **hardcodes a real AWS account id (`086325823665`) and an Admin role ARN** as pydantic defaults, and `make train`'s default path is cloud.

1. Remove the hardcoded account/role/bucket defaults; load from environment (no credentials in source, no new secret store — just `os.environ` with clear errors).
2. Make local the default training path; `--cloud` stays the explicit opt-in (it already exists as a flag — invert the `make` targets or at least document it).
3. `git rm --cached` done for the 27 committed `__pycache__/*.pyc`; verify none return (`.gitignore` now covers them).

# Distributed Self-Play Implementation Brief

## Context
Chess self-play training on AWS (PyTorch). Paper: arxiv 2609.37447v1 "Engineering Efficient Self-Play Chess".

## Pending TODOs

### Task A: Distributed Training (src/training/trainer.py)
- Add torch.distributed.init_process_group in Trainer initialization
- Wrap network with DDP (torch.nn.parallel.DistributedDataParallel)  
- Use DistributedSampler for replay buffer
- Add `--on-demand` CLI flag + config + SageMaker integration

### Task B: GPU-Accelerated Self-Play (src/training/search/*.py)
- Implement CUDA-based move generation (bitboard representation)
- GPU-based MCTS parallel search with CUDA kernel for tree expansion
- Mixed device pipeline: CPU book moves + GPU neural evaluation
- Document CUDA setup requirements

### Task C: Docker/CI/Infra
- PyTorch CUDA Dockerfile (2.14 + CUDA 13.0 + TensorRT)
- GitHub Actions CI: test + Docker build + deploy

### Task D: Multi-GPU SageMaker
- Update SageMaker launcher for multi-GPU instances
- --on-demand CLI flag (override spot_instances)

## Constraints
- Must pass: `uv run ruff check src tests` then `PYTHONPATH=. uv run pytest -q`
- Each commit = one subtask
- Use TDD: red → green loop
- No duplicate work across subagents

## Key Files
- src/config.py: Pydantic model, env vars
- src/cli.py: Click CLI, --cloud/--local/--on-demand
- src/training/trainer.py: Training loop
- src/training/search/selfplay.py: ParallelSelfPlay
- src/training/search/mcts.py: MCTS nodes/search
- src/training/game.py: ChessGame, play_game
- src/infra/sagemaker_manager.py: SageMaker launcher
- tests/: conftest.py with _env_fast_cpu fixture

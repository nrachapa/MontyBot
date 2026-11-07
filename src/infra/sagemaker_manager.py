#!/usr/bin/env python3
import argparse
import contextlib
import os
import sys
import time

import boto3
import torch
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

# Ensure container paths include project root and src for imports
_cwd = os.getcwd()
for p in ["/opt/ml/code", "/opt/ml/code/src", _cwd, os.path.join(_cwd, "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def load_aws_config(config=None):
    """Return AWS/training config as a dict.

    When *config* is not given the function falls back to the global CONFIG
    singleton for backwards compatibility with callers that haven't been
    updated yet.
    """
    if config is None:
        from src.config import CONFIG
        config = CONFIG

    return {
        "training": {
            "iterations": config.iterations,
            "job_name": config.job_name,
            "max_wait_seconds": config.max_wait_seconds,
            "max_run_seconds": config.max_run_seconds,
            "poll_seconds": config.poll_seconds,
            "default_instance_type": config.default_instance_type,
            "spot_instances": config.spot_instances,
        },
        "aws": {"s3": {"bucket": config.bucket}, "credentials": {"role_arn": config.role_arn}},
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to YAML config (deprecated)")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--s3_bucket", type=str, required=False)
    parser.add_argument("--job_name", type=str, required=False)
    return parser.parse_args()


def launch_job(args=None, instance_type: str | None = None, config=None):
    if config is None:
        from src.config import CONFIG
        config = CONFIG
    trn = {
        "iterations": config.iterations,
        "job_name": config.job_name,
        "max_wait_seconds": config.max_wait_seconds,
        "max_run_seconds": config.max_run_seconds,
        "poll_seconds": config.poll_seconds,
        "default_instance_type": config.default_instance_type,
        "spot_instances": config.spot_instances,
    }
    aws = {"s3": {"bucket": config.bucket}, "credentials": {"role_arn": config.role_arn}}
    creds = aws["credentials"]
    job_name = trn["job_name"]
    iterations = trn["iterations"]
    bucket = aws["s3"]["bucket"]
    instance_type = instance_type or trn["default_instance_type"]
    use_spot_instances = bool(trn.get("spot_instances", True))

    try:
        role = get_execution_role()
    except Exception:  # noqa: E722
        role = creds["role_arn"]

    # Use SageMaker's built-in PyTorch container (script mode)
    estimator = PyTorch(
        entry_point="src/infra/sagemaker_manager.py",
        source_dir=".",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.2.0",
        py_version="py310",
        use_spot_instances=use_spot_instances,
        max_wait=trn["max_wait_seconds"],
        max_run=trn["max_run_seconds"],
        hyperparameters={"iterations": iterations, "s3_bucket": bucket, "job_name": job_name},
        output_path=f"s3://{bucket}",
        code_location=f"s3://{bucket}",
    )

    estimator.fit(job_name=job_name, wait=False)
    print(f"Training job launched: {estimator.latest_training_job.name}")
    return estimator.latest_training_job.name


def wait_for_job(job_name: str, poll_seconds: int = 30) -> str:
    """Poll SageMaker training job until it reaches a terminal state.

    Returns the final TrainingJobStatus (Completed | Failed | Stopped).
    """
    sm = boto3.client("sagemaker")
    last_status = None
    last_secondary = None
    start_ts = time.time()
    print(f"Waiting for job to complete: {job_name} (poll {poll_seconds}s)")
    while True:
        resp = sm.describe_training_job(TrainingJobName=job_name)
        status = resp.get("TrainingJobStatus", "Unknown")
        secondary = resp.get("SecondaryStatus", "Unknown")
        if status != last_status or secondary != last_secondary:
            # Try to surface the most recent status message if present
            msg = None
            try:
                transitions = resp.get("SecondaryStatusTransitions") or []
                if transitions:
                    msg = transitions[-1].get("StatusMessage")
            except Exception:
                msg = None
            if msg:
                print(f"Status: {status} ({secondary}) - {msg}")
            else:
                print(f"Status: {status} ({secondary})")
            last_status, last_secondary = status, secondary
        if status in ("Completed", "Failed", "Stopped"):
            if status == "Failed":
                reason = resp.get("FailureReason", "")
                if reason:
                    print(f"FailureReason: {reason}")
            artifacts = (resp.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
            out_cfg = (resp.get("OutputDataConfig") or {}).get("S3OutputPath")
            if artifacts:
                print(f"Model artifacts: {artifacts}")
            if out_cfg:
                print(f"Output S3: {out_cfg}")
            elapsed = int(time.time() - start_ts)
            print(f"Final status: {status} (elapsed {elapsed}s)")
            return status
        time.sleep(poll_seconds)


def launch_training(instance_type: str | None = None, config=None):
    """Launch training without parsing CLI arguments (for use by CLI)"""
    if config is None:
        from src.config import CONFIG
        config = CONFIG
    job_name = launch_job(instance_type=instance_type, config=config)
    print(f"Monitor: aws sagemaker describe-training-job --training-job-name {job_name}")
    trn = {
        "iterations": config.iterations,
        "job_name": config.job_name,
        "max_wait_seconds": config.max_wait_seconds,
        "max_run_seconds": config.max_run_seconds,
        "poll_seconds": config.poll_seconds,
        "default_instance_type": config.default_instance_type,
        "spot_instances": config.spot_instances,
    }
    poll_seconds = int(trn.get("poll_seconds", 30))
    wait_for_job(job_name, poll_seconds)


class SageMakerTrainer:
    def __init__(self, s3_manager, config):
        from src.training.trainer import Trainer

        # Enable TensorFloat-32 for T4 GPU speedup (2-3x faster matmul)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.config = config
        self.trainer = Trainer(config)
        self.s3_manager = s3_manager
        # Enable AMP scaler only when mixed precision is requested and CUDA is available
        self.scaler = (
            torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        )
        self.start_iteration = 0

    def save_checkpoint(self, iteration: int):
        checkpoint = {
            "model_state": self.trainer.network.state_dict(),
            "optimizer_state": self.trainer.training.opt.state_dict(),
            "iteration": iteration,
            "config": self.config.model_dump(),
        }
        self.s3_manager.upload_checkpoint(checkpoint, iteration)
        print(f"Saved checkpoint at iteration {iteration}")

    def load_checkpoint(self, checkpoint: dict):
        """Restore net weights + optimizer state + the resume iteration from a
        downloaded checkpoint (survives spot preemption / container restarts)."""
        self.trainer.network.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.trainer.training.opt.load_state_dict(checkpoint["optimizer_state"])
        self.start_iteration = int(checkpoint.get("iteration", 0)) + 1
        print(f"Resumed from iteration {checkpoint.get('iteration')} (continuing at {self.start_iteration})")

    def train_gpu(self, iterations: int):
        for i in range(self.start_iteration, iterations):
            trajectories = self.trainer.selfplay.generate_games(self.config.games_per_iteration)
            self.trainer.buffer.add_batch(trajectories)

            losses = []
            for _ in range(self.config.train_steps):
                batch = self.trainer.buffer.sample(self.config.batch_size)
                if not batch:
                    continue

                # Use autocast only when mixed precision is requested and CUDA is available
                if self.config.mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        loss = self.trainer.training.train_step(batch)
                else:
                    loss = self.trainer.training.train_step(batch)
                losses.append(loss)

            avg_loss = sum(losses) / len(losses) if losses else 0.0
            print(f"Iteration {i}: Loss = {avg_loss:.4f}")

            self.save_checkpoint(i)

        self.save_checkpoint(iterations)


def main():
    args = parse_args()

    from src.config import CONFIG

    # If running as SageMaker training script
    if args.s3_bucket and args.job_name:
        from src.infra.s3_manager import S3Manager

        s3_manager = S3Manager(args.s3_bucket, prefix=args.job_name)
        # Ensure the checkpoint bucket exists prior to any upload attempts
        s3_manager.ensure_bucket_exists()
        trainer = SageMakerTrainer(s3_manager, CONFIG)
        # Resume from this job's last checkpoint if the container restarted
        # (spot preemption); otherwise start from iteration 0.
        checkpoint = s3_manager.download_latest_for_prefix()
        if checkpoint:
            trainer.load_checkpoint(checkpoint)
        trainer.train_gpu(args.iterations)
        print("Training completed")
    else:
        # Running as launcher
        job_name = launch_job(args, config=CONFIG)
        print(f"Monitor: aws sagemaker describe-training-job --training-job-name {job_name}")
        config = load_aws_config()
        poll_seconds = int(config.get("training", {}).get("poll_seconds", 30))
        wait_for_job(job_name, poll_seconds)


if __name__ == "__main__":  # pragma: no cover
    import multiprocessing

    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)
    main()

#!/usr/bin/env python3
"""Continuous SageMaker job monitor.

Usage:
    python scripts/monitor.py <job-name> [--region us-west-2] [--output results.txt]

Polls job status + S3 model count + log stream summaries every 60s,
printing a single line per check. Writes structured output to
--output for parsing by scripts or CI.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous SageMaker job monitor")
    parser.add_argument("job_name", help="SageMaker training job name")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--output", help="Write structured output to file")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    from botocore.config import Config
    import boto3

    sagemaker = boto3.client(
        "sagemaker",
        region_name=args.region,
        config=Config(retries={"max_attempts": 2}),
    )
    cloudwatch = boto3.client("logs", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    job_name = args.job_name
    bucket = f"montybot-training-{args.region}"
    models_prefix = f"{job_name}/models/"

    # Find the log stream on first poll (log stream is created after training starts)
    log_stream = None

    start = time.time()
    last_iteration_msg = None

    print(f"Monitoring {job_name} in {args.region}...", flush=True)

    while True:
        try:
            status = sagemaker.describe_training_job(TrainingJobName=job_name)
            jstatus = status["TrainingJobStatus"]
            secondary = status.get("SecondaryStatus", "")

            # Count models
            model_count = 0
            try:
                objs = s3.list_objects_v2(Bucket=bucket, Prefix=models_prefix)
                if "Contents" in objs:
                    model_count = len(objs["Contents"])
            except Exception:
                pass

            elapsed = int(time.time() - start)
            line = (
                f"[{jstatus:14s}] {secondary:16s} | "
                f"models={model_count:3d} | "
                f"elapsed={elapsed}s"
            )
            print(line, flush=True)

            # Parse latest log events for iteration progress
            if log_stream is None:
                # Try to find log stream
                try:
                    streams = cloudwatch.describe_log_streams(
                        logGroupName="/aws/sagemaker/TrainingJobs",
                        logStreamNamePrefix=job_name,
                        limit=10,
                        orderBy="LogStreamName",
                    )
                    if streams.get("logStreams"):
                        log_stream = streams["logStreams"][-1]["logStreamName"]
                except Exception:
                    pass

            if log_stream:
                try:
                    events = cloudwatch.get_log_events(
                        logGroupName="/aws/sagemaker/TrainingJobs",
                        logStreamName=log_stream,
                        limit=50,
                    )
                    for event in reversed(events.get("events", [])[-10:]):
                        msg = event["message"]
                        if "Iteration" in msg and "total=" in msg:
                            if msg != last_iteration_msg:
                                last_iteration_msg = msg
                                print(f"  {msg}", flush=True)
                except Exception:
                    pass

            # Write structured output
            if args.output:
                record = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "status": jstatus,
                    "secondary": secondary,
                    "models": model_count,
                    "elapsed": elapsed,
                    "iteration": last_iteration_msg[-200:] if last_iteration_msg else None,
                }
                with open(args.output, "a") as f:
                    f.write(json.dumps(record) + "\n")

            # Exit on terminal state
            if jstatus in ("Completed", "Stopped", "Failed"):
                print(f"\nDone: {jstatus}", flush=True)
                if last_iteration_msg:
                    print(f"Last iteration: {last_iteration_msg}", flush=True)
                break

        except Exception as e:
            print(f"ERROR: {e}", flush=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()

import os

import boto3
import torch


class S3Manager:
    def __init__(self, bucket: str, prefix: str = "montybot"):
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix

    def ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        from botocore.exceptions import ClientError

        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
                if region == "us-east-1":
                    self.s3.create_bucket(Bucket=self.bucket)
                else:
                    self.s3.create_bucket(Bucket=self.bucket, CreateBucketConfiguration={"LocationConstraint": region})
                print(f"✅ Created S3 bucket: {self.bucket}")
            else:
                print(f"❌ Error checking/creating bucket: {e}")
                raise
        except Exception as e:
            print(f"❌ Error checking/creating bucket: {e}")
            raise

    def upload_checkpoint(self, model_state: dict, iteration: int):
        local_path = f"/tmp/model_{iteration}.pt"
        torch.save(model_state, local_path)
        s3_key = f"{self.prefix}/models/model_{iteration}.pt"
        self.s3.upload_file(local_path, self.bucket, s3_key)
        os.remove(local_path)

        # Cleanup: Keep only last 3 training jobs
        self.cleanup_old_jobs(keep=3)
        return s3_key

    def cleanup_old_jobs(self, keep: int = 3):
        """Delete old training jobs, keeping only the most recent 'keep' ones."""
        try:
            # List top-level folders (prefixes)
            response = self.s3.list_objects_v2(Bucket=self.bucket, Delimiter="/")
            prefixes = response.get("CommonPrefixes", [])

            # Filter for montybot training jobs
            job_prefixes = [p["Prefix"] for p in prefixes if p["Prefix"].startswith("montybot-gpu-training-")]

            # Sort by name (which includes timestamp) descending
            job_prefixes.sort(reverse=True)

            if len(job_prefixes) > keep:
                to_delete = job_prefixes[keep:]
                for prefix in to_delete:
                    self._delete_prefix(prefix)
                print(f"🧹 Cleaned up {len(to_delete)} old jobs (kept {keep})")
        except Exception as e:
            print(f"⚠️ Failed to cleanup old jobs: {e}")

    def _delete_prefix(self, prefix: str):
        """Helper to delete all objects under a prefix"""
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" in page:
                delete_keys = [{"Key": obj["Key"]} for obj in page["Contents"]]
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": delete_keys})

    def download_checkpoint(self, iteration: int = None):
        s3_key = None

        if iteration is None:
            # Find latest job first
            response = self.s3.list_objects_v2(Bucket=self.bucket, Delimiter="/")
            prefixes = response.get("CommonPrefixes", [])
            job_prefixes = [p["Prefix"] for p in prefixes if p["Prefix"].startswith("montybot-gpu-training-")]

            if not job_prefixes:
                print("❌ No training jobs found in S3")
                return None

            # Sort by timestamp (descending)
            job_prefixes.sort(reverse=True)
            latest_job = job_prefixes[0]
            print(f"📂 Found latest job: {latest_job}")

            # Find latest model in that job
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"{latest_job}models/")
            if "Contents" not in response:
                print(f"❌ No models found in {latest_job}models/")
                return None

            latest_model = max(response["Contents"], key=lambda x: x["LastModified"])
            s3_key = latest_model["Key"]
        else:
            # If iteration specified, we need to know which job...
            # For now, let's assume we want the latest job's iteration
            # This is a bit tricky if we don't know the job name.
            # Let's search the latest job for that iteration.
            response = self.s3.list_objects_v2(Bucket=self.bucket, Delimiter="/")
            prefixes = response.get("CommonPrefixes", [])
            job_prefixes = [p["Prefix"] for p in prefixes if p["Prefix"].startswith("montybot-gpu-training-")]
            job_prefixes.sort(reverse=True)

            if not job_prefixes:
                return None

            latest_job = job_prefixes[0]
            s3_key = f"{latest_job}models/model_{iteration}.pt"

        print(f"⬇️ Downloading {s3_key}...")
        local_path = "/tmp/checkpoint.pt"
        try:
            self.s3.download_file(self.bucket, s3_key, local_path)
            return torch.load(local_path, map_location="cpu")
        except Exception as e:
            print(f"❌ Failed to download {s3_key}: {e}")
            return None

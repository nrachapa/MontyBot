import os
import sys
from unittest.mock import ANY, MagicMock, Mock, patch

import boto3
import pytest
import yaml
from botocore.exceptions import ClientError, NoCredentialsError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.infra.s3_manager import S3Manager


def load_config():
    from src.config import CONFIG

    return {
        "aws": {
            "credentials": {"region": CONFIG.region, "account_id": CONFIG.account_id, "role_arn": CONFIG.role_arn},
            "s3": {"bucket": CONFIG.bucket, "prefix": CONFIG.prefix},
        }
    }


class TestS3Manager:
    @patch("boto3.client")
    def test_upload_checkpoint(self, mock_boto):
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3

        manager = S3Manager("test-bucket", "test-prefix")
        checkpoint = {"model_state": {}, "iteration": 100}

        with patch("torch.save"), patch("os.remove"):
            key = manager.upload_checkpoint(checkpoint, 100)
            assert "test-prefix/models/model_100.pt" in key
            mock_s3.upload_file.assert_called_once()

            # Verify cleanup called
            mock_s3.list_objects_v2.assert_called()

    @patch("boto3.client")
    def test_download_checkpoint(self, mock_boto):
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3

        # First call lists jobs, second call lists models
        mock_s3.list_objects_v2.side_effect = [
            {"CommonPrefixes": [{"Prefix": "montybot-gpu-training-job1/"}]},
            {"Contents": [{"Key": "montybot-gpu-training-job1/models/model_100.pt", "LastModified": "2024-01-01"}]},
        ]

        manager = S3Manager("test-bucket")

        with patch("torch.load", return_value={"iteration": 100}):
            checkpoint = manager.download_checkpoint()
            assert checkpoint["iteration"] == 100
            mock_s3.download_file.assert_called_once()

    @patch("boto3.client")
    def test_download_missing_checkpoint(self, mock_boto):
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {}  # No Contents

        manager = S3Manager("test-bucket")
        result = manager.download_checkpoint()
        assert result is None

    @patch("boto3.client")
    def test_download_specific_iteration(self, mock_boto):
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3

        # List jobs to find latest
        mock_s3.list_objects_v2.return_value = {"CommonPrefixes": [{"Prefix": "montybot-gpu-training-job1/"}]}

        manager = S3Manager("test-bucket")
        with patch("torch.load", return_value={"iteration": 100}):
            result = manager.download_checkpoint(100)
            assert result["iteration"] == 100
            # Should download from latest job
            mock_s3.download_file.assert_called_with(
                "test-bucket", "montybot-gpu-training-job1/models/model_100.pt", ANY
            )

    @patch("boto3.client")
    def test_ensure_bucket_exists_already_exists(self, mock_boto):
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3
        manager = S3Manager("existing-bucket")
        # head_bucket succeeds, no create_bucket called
        manager.ensure_bucket_exists()
        mock_s3.create_bucket.assert_not_called()

    @patch("boto3.client")
    def test_ensure_bucket_exists_create_in_us_east_1(self, mock_boto):
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3
        from botocore.exceptions import ClientError

        error = ClientError({"Error": {"Code": "404"}}, "HeadBucket")
        mock_s3.head_bucket.side_effect = error
        with patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-east-1"}):
            manager = S3Manager("new-bucket")
            manager.ensure_bucket_exists()
            mock_s3.create_bucket.assert_called_with(Bucket="new-bucket")

    @patch("boto3.client")
    def test_ensure_bucket_exists_create_in_other_region(self, mock_boto):
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3
        from botocore.exceptions import ClientError

        error = ClientError({"Error": {"Code": "404"}}, "HeadBucket")
        mock_s3.head_bucket.side_effect = error
        with patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"}):
            manager = S3Manager("new-bucket-2")
            manager.ensure_bucket_exists()
            mock_s3.create_bucket.assert_called_with(
                Bucket="new-bucket-2", CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
            )

    @patch("boto3.client")
    def test_ensure_bucket_exists_other_client_error(self, mock_boto):
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3
        from botocore.exceptions import ClientError

        error = ClientError({"Error": {"Code": "500"}}, "HeadBucket")
        mock_s3.head_bucket.side_effect = error
        manager = S3Manager("err-bucket")
        with pytest.raises(ClientError):
            manager.ensure_bucket_exists()

    @patch("boto3.client")
    def test_ensure_bucket_exists_generic_exception(self, mock_boto):
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3
        mock_s3.head_bucket.side_effect = RuntimeError("boom")
        manager = S3Manager("boom-bucket")
        with pytest.raises(RuntimeError):
            manager.ensure_bucket_exists()


class TestSageMakerIntegration:
    @patch("src.infra.sagemaker_manager.ModelTrainer")
    @patch("src.infra.sagemaker_manager.get_execution_role")
    def test_launcher_script_mode(self, mock_role, mock_trainer_cls):
        mock_role.return_value = "test-role"
        mock_trainer = Mock()
        mock_trainer.fit.return_value = "test-job-123"
        mock_trainer_cls.return_value = mock_trainer

        from src.infra.sagemaker_manager import launch_job

        args = Mock()
        args.config = "configurations/config.yaml"

        job_name = launch_job(args)
        assert job_name == "test-job-123"
        mock_trainer_cls.assert_called_once()
        mock_trainer.fit.assert_called_once_with(job_name=ANY, wait=False)

    @patch("src.infra.sagemaker_manager.ModelTrainer")
    @patch("src.infra.sagemaker_manager.get_execution_role")
    def test_launcher_script_mode_parameters(self, mock_role, mock_trainer_cls):
        mock_role.return_value = "test-role"
        mock_trainer = Mock()
        mock_trainer.fit.return_value = "job-456"
        mock_trainer_cls.return_value = mock_trainer

        from src.infra.sagemaker_manager import launch_job

        args = Mock()
        args.config = "configurations/config.yaml"

        job_name = launch_job(args)
        assert job_name == "job-456"

        # Verify script mode parameters
        call_args = mock_trainer_cls.call_args
        assert call_args.kwargs["source_code"].entry_script == "src/infra/sagemaker_manager.py"
        assert call_args.kwargs["source_code"].source_dir == "."
        assert call_args.kwargs["stopping_condition"].max_wait_time_in_seconds == 14400
        assert call_args.kwargs["stopping_condition"].max_runtime_in_seconds == 10800

    @patch("src.infra.sagemaker_manager.ModelTrainer")
    @patch("src.infra.sagemaker_manager.get_execution_role")
    def test_launcher_exception_handling(self, mock_role, mock_trainer_cls):
        mock_role.side_effect = Exception("No role")
        mock_trainer = Mock()
        mock_trainer.fit.return_value = "test-job-123"
        mock_trainer_cls.return_value = mock_trainer

        from src.infra.sagemaker_manager import launch_job

        args = Mock()
        args.config = "configurations/config.yaml"

        job_name = launch_job(args)
        assert job_name == "test-job-123"

    @patch("argparse.ArgumentParser.parse_args")
    def test_launcher_main(self, mock_args):
        from src.infra.sagemaker_manager import main

        mock_args.return_value = Mock(config="configurations/config.yaml", s3_bucket=None, job_name=None)

        with (
            patch("src.infra.sagemaker_manager.launch_job", return_value="job-123"),
            patch("src.infra.sagemaker_manager.wait_for_job", return_value="Completed") as mock_wait,
        ):
            main()
            mock_wait.assert_called_once()

    @patch("argparse.ArgumentParser.parse_args")
    @patch("src.infra.sagemaker_manager.SageMakerTrainer")
    @patch("src.infra.s3_manager.S3Manager")
    def test_sagemaker_train_main(self, mock_s3, mock_trainer, mock_args):
        from src.infra.sagemaker_manager import main

        mock_args.return_value = Mock(iterations=100, s3_bucket="bucket", job_name="job")

        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        main()
        mock_trainer_instance.train_gpu.assert_called_once_with(100)

    def test_sagemaker_trainer_empty_batch_continue_path(self):
        from src.config import CONFIG
        from src.infra.sagemaker_manager import SageMakerTrainer

        s3_manager = Mock()

        # Store original values
        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        original_mixed = CONFIG.mixed_precision

        CONFIG.filters = 8
        CONFIG.blocks = 1
        CONFIG.mixed_precision = False

        try:
            trainer = SageMakerTrainer(s3_manager, CONFIG)
            # train_gpu reads these off the nested Trainer: mock there, not on
            # the SageMakerTrainer wrapper (which has no such attributes).
            trainer.trainer.selfplay = Mock()
            trainer.trainer.selfplay.generate_games.return_value = [[("state", "policy", 0.1)]]
            trainer.trainer.buffer = Mock()
            trainer.trainer.buffer.sample.return_value = []
            trainer.trainer.training = Mock()
            trainer.trainer.network = Mock()
            trainer.trainer.network.state_dict.return_value = {"s": 1}
            trainer.train_gpu(1)
            # Should still save checkpoint at end
            s3_manager.upload_checkpoint.assert_called()
        finally:
            CONFIG.filters = original_filters
            CONFIG.blocks = original_blocks
            CONFIG.mixed_precision = original_mixed

    def test_sagemaker_trainer_mixed_precision(self):
        from src.config import CONFIG
        from src.infra.sagemaker_manager import SageMakerTrainer

        s3_manager = Mock()

        # Store original values
        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        original_mixed = CONFIG.mixed_precision

        CONFIG.filters = 8
        CONFIG.blocks = 1
        CONFIG.mixed_precision = True

        try:
            # AMP scaler should only be enabled when CUDA is available
            with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.amp.GradScaler"):
                trainer = SageMakerTrainer(s3_manager, CONFIG)
                assert trainer.scaler is None

            # When CUDA is available, scaler should be created
            with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.amp.GradScaler"):
                trainer = SageMakerTrainer(s3_manager, CONFIG)
                assert trainer.scaler is not None
        finally:
            CONFIG.filters = original_filters
            CONFIG.blocks = original_blocks
            CONFIG.mixed_precision = original_mixed

        trainer.network = Mock()
        trainer.network.state_dict.return_value = {"test": "state"}
        trainer.save_checkpoint(50)
        s3_manager.upload_checkpoint.assert_called_once()

    def test_sagemaker_trainer_workflow(self):
        from src.config import CONFIG
        from src.infra.sagemaker_manager import SageMakerTrainer

        s3_manager = Mock()

        # Store original values
        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        original_mixed = CONFIG.mixed_precision

        CONFIG.filters = 8
        CONFIG.blocks = 1
        CONFIG.mixed_precision = False

        try:
            trainer = SageMakerTrainer(s3_manager, CONFIG)

            # Mock components after trainer initialization
            trainer.trainer.selfplay = Mock()
            trainer.trainer.selfplay.generate_games.return_value = [[("state", "policy", 0.1)]]
            trainer.trainer.buffer = Mock()
            trainer.trainer.buffer.sample.return_value = [("state", "policy", 0.1)]
            trainer.trainer.training = Mock()
            trainer.trainer.training.train_step.return_value = 1.5
            trainer.trainer.network = Mock()
            trainer.trainer.network.state_dict.return_value = {"test": "state"}

            # Test full workflow with print statements
            trainer.train_gpu(11)  # Test iteration 10 for print

            # Verify calls
            assert trainer.trainer.selfplay.generate_games.call_count == 11
            assert s3_manager.upload_checkpoint.call_count >= 1
        finally:
            CONFIG.filters = original_filters
            CONFIG.blocks = original_blocks
            CONFIG.mixed_precision = original_mixed

    def test_sagemaker_trainer_mixed_precision_workflow(self):
        from src.config import CONFIG
        from src.infra.sagemaker_manager import SageMakerTrainer

        s3_manager = Mock()

        # Store original values
        original_filters = CONFIG.filters
        original_blocks = CONFIG.blocks
        original_mixed = CONFIG.mixed_precision

        CONFIG.filters = 8
        CONFIG.blocks = 1
        CONFIG.mixed_precision = True

        try:
            # Ensure mixed precision path is taken by simulating CUDA availability
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.amp.GradScaler"),
                patch("torch.cuda.amp.autocast"),
            ):
                trainer = SageMakerTrainer(s3_manager, CONFIG)

            # Mock components for mixed precision path (nested Trainer —
            # train_gpu reads them off self.trainer)
            trainer.trainer.selfplay = Mock()
            trainer.trainer.selfplay.generate_games.return_value = [[("state", "policy", 0.1)]]
            trainer.trainer.buffer = Mock()
            trainer.trainer.buffer.sample.return_value = [("state", "policy", 0.1)]
            trainer.trainer.training = Mock()
            trainer.trainer.training.train_step.return_value = 1.5
            trainer.trainer.network = Mock()
            trainer.trainer.network.state_dict.return_value = {"test": "state"}

            # Test mixed precision training path
            trainer.train_gpu(1)

            # Verify mixed precision was used
            assert trainer.scaler is not None
        finally:
            CONFIG.filters = original_filters
            CONFIG.blocks = original_blocks
            CONFIG.mixed_precision = original_mixed

    @patch("boto3.client")
    @patch("time.sleep")
    def test_wait_for_job_completed(self, mock_sleep, mock_boto):
        """Test wait_for_job with completed status"""
        from src.infra.sagemaker_manager import wait_for_job

        mock_sm = Mock()
        mock_boto.return_value = mock_sm
        mock_sm.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "SecondaryStatus": "Completed",
            "ModelArtifacts": {"S3ModelArtifacts": "s3://bucket/model.tar.gz"},
            "OutputDataConfig": {"S3OutputPath": "s3://bucket/output/"},
        }

        result = wait_for_job("test-job", 1)
        assert result == "Completed"
        mock_sm.describe_training_job.assert_called_with(TrainingJobName="test-job")

    @patch("boto3.client")
    @patch("time.sleep")
    def test_wait_for_job_failed(self, mock_sleep, mock_boto):
        """Test wait_for_job with failed status"""
        from src.infra.sagemaker_manager import wait_for_job

        mock_sm = Mock()
        mock_boto.return_value = mock_sm
        mock_sm.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed",
            "SecondaryStatus": "Failed",
            "FailureReason": "Out of memory",
        }

        result = wait_for_job("test-job", 1)
        assert result == "Failed"

    @patch("boto3.client")
    @patch("time.sleep")
    def test_wait_for_job_with_transitions(self, mock_sleep, mock_boto):
        """Test wait_for_job with status transitions"""
        from src.infra.sagemaker_manager import wait_for_job

        mock_sm = Mock()
        mock_boto.return_value = mock_sm

        # First call: InProgress, second call: Completed
        mock_sm.describe_training_job.side_effect = [
            {
                "TrainingJobStatus": "InProgress",
                "SecondaryStatus": "Training",
                "SecondaryStatusTransitions": [{"StatusMessage": "Training started"}],
            },
            {"TrainingJobStatus": "Completed", "SecondaryStatus": "Completed"},
        ]

        result = wait_for_job("test-job", 1)
        assert result == "Completed"
        assert mock_sm.describe_training_job.call_count == 2


# Check for usable AWS credentials (skip if credentials are empty or credentials object is None)
def _has_usable_aws_credentials():
    try:
        session = boto3.session.Session()
        creds = session.get_credentials()
        if creds is None:
            return False
        # Check if credentials have valid access_key and secret_key
        if not hasattr(creds, 'access_key') or creds.access_key is None:
            return False
        if not (hasattr(creds, 'secret_key') and creds.secret_key is not None):
            return False
    except Exception:
        return False


@pytest.mark.skipif(
    not _has_usable_aws_credentials(),
    reason="live AWS access tests need valid credentials",
)
class TestAWSAccess:
    def test_aws_credentials(self):
        """Test if AWS credentials are properly configured"""
        try:
            sts = boto3.client("sts")
            identity = sts.get_caller_identity()
            assert "Account" in identity
            print(f"✅ AWS Access: Account {identity['Account']}")
        except (NoCredentialsError, ClientError) as e:
            pytest.fail(f"❌ AWS credentials not configured: {e}")

    def test_s3_bucket_access(self):
        """Test if S3 bucket is accessible"""
        config = load_config()
        bucket = config["aws"]["s3"]["bucket"]

        s3_manager = S3Manager(bucket)
        s3_manager.ensure_bucket_exists()

        try:
            s3 = boto3.client("s3")
            s3.head_bucket(Bucket=bucket)
            print(f"✅ S3 Access: Bucket '{bucket}' accessible")
        except ClientError as e:
            pytest.fail(f"❌ S3 bucket '{bucket}' not accessible: {e}")

    def test_sagemaker_role(self):
        """Test if SageMaker role exists (either via execution role or CONFIG.role_arn)"""
        iam = boto3.client("iam")

        # Try get_execution_role() first (the manager's primary path)
        from sagemaker.core.helper.session_helper import get_execution_role
        try:
            role = get_execution_role()
            role_name = role.split("/")[-1]
            iam.get_role(RoleName=role_name)
            print(f"✅ IAM Role: '{role_name}' exists (via get_execution_role)")
            return
        except Exception:
            pass  # execution role not available, try CONFIG.role_arn

        # Fallback: check CONFIG.role_arn if set
        from src.config import CONFIG
        role_arn = CONFIG.role_arn
        if role_arn is None:
            pytest.skip("Neither get_execution_role nor CONFIG.role_arn available")

        role_name = role_arn.split("/")[-1]
        try:
            iam.get_role(RoleName=role_name)
            print(f"✅ IAM Role: '{role_name}' exists (via CONFIG.role_arn)")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "AccessDenied":
                pytest.skip(f"GPUuser lacks iam:GetRole on '{role_name}' (env limitation)")
            pytest.fail(f"❌ Unexpected IAM error: {e}")

    def test_s3_manager_download_checkpoint_exists(self):
        """Verify download_checkpoint method exists"""
        from src.infra.s3_manager import S3Manager

        s3_manager = S3Manager("test-bucket")
        assert hasattr(s3_manager, "download_checkpoint")


class TestSageMakerMultiGPU:
    """Test SageMaker multi-GPU configuration."""

    def test_gpu_count_config(self):
        """Test instance_gpu_count config field exists."""
        from src.config import CONFIG
        original = CONFIG.model_dump()
        CONFIG.instance_gpu_count = 4
        assert CONFIG.instance_gpu_count == 4
        CONFIG.__init__(**original)

    def test_on_demand_config(self):
        """Test on_demand config field exists."""
        from src.config import CONFIG
        original = CONFIG.model_dump()
        CONFIG.on_demand = True
        assert CONFIG.on_demand
        CONFIG.__init__(**original)
        CONFIG.__init__(**original)

    def test_multi_gpu_instance_selection(self):
        """Test multi-GPU instance type selection logic."""
        from src.config import CONFIG
        from src.infra.sagemaker_manager import load_aws_config

        original = CONFIG.model_dump()
        CONFIG.instance_gpu_count = 4
        CONFIG.on_demand = True
        CONFIG.on_demand_instance_type = "ml.p3.8xlarge"

        try:
            result = load_aws_config(CONFIG)
            # Verify multi-GPU fields are in config
            assert result["training"]["on_demand"] is True
            assert result["training"]["instance_gpu_count"] == 4
            assert result["training"]["on_demand_instance_type"] == "ml.p3.8xlarge"
        finally:
            CONFIG.__init__(**original)

    def test_spot_instances_not_overridden(self):
        """Test spot_instances remains True when on_demand is False."""
        from src.config import CONFIG
        from src.infra.sagemaker_manager import load_aws_config
        original = CONFIG.model_dump()
        CONFIG.spot_instances = True
        CONFIG.on_demand = False

        try:
            result = load_aws_config(CONFIG)
            assert result["training"]["spot_instances"] is True
        finally:
            CONFIG.__init__(**original)

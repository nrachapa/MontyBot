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
    @patch("src.infra.sagemaker_manager.PyTorch")
    @patch("src.infra.sagemaker_manager.get_execution_role")
    def test_launcher_script_mode(self, mock_role, mock_pytorch):
        mock_role.return_value = "test-role"
        mock_estimator_instance = Mock()
        mock_estimator_instance.latest_training_job.name = "test-job-123"
        mock_pytorch.return_value = mock_estimator_instance

        from src.infra.sagemaker_manager import launch_job

        args = Mock()
        args.config = "configurations/config.yaml"

        job_name = launch_job(args)
        assert job_name == "test-job-123"
        mock_pytorch.assert_called_once()
        mock_estimator_instance.fit.assert_called_once_with(job_name=ANY, wait=False)

    @patch("src.infra.sagemaker_manager.PyTorch")
    @patch("src.infra.sagemaker_manager.get_execution_role")
    def test_launcher_script_mode_parameters(self, mock_role, mock_pytorch):
        mock_role.return_value = "test-role"
        mock_estimator_instance = Mock()
        mock_estimator_instance.latest_training_job.name = "job-456"
        mock_pytorch.return_value = mock_estimator_instance

        from src.infra.sagemaker_manager import launch_job

        args = Mock()
        args.config = "configurations/config.yaml"

        job_name = launch_job(args)
        assert job_name == "job-456"

        # Verify script mode parameters
        call_args = mock_pytorch.call_args
        assert call_args.kwargs["entry_point"] == "src/infra/sagemaker_manager.py"
        assert call_args.kwargs["source_dir"] == "."
        assert call_args.kwargs["framework_version"] == "2.2.0"
        assert call_args.kwargs["py_version"] == "py310"

    @patch("src.infra.sagemaker_manager.PyTorch")
    @patch("src.infra.sagemaker_manager.get_execution_role")
    def test_launcher_exception_handling(self, mock_role, mock_pytorch):
        mock_role.side_effect = Exception("No role")
        mock_estimator_instance = Mock()
        mock_estimator_instance.latest_training_job.name = "test-job-123"
        mock_pytorch.return_value = mock_estimator_instance

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
            trainer = SageMakerTrainer(s3_manager)
            # Force buffer.sample to return [] so the 'continue' at line 43 is hit
            trainer.selfplay = Mock()
            trainer.selfplay.generate_games.return_value = [[("state", "policy", 0.1)]]
            trainer.buffer = Mock()
            trainer.buffer.sample.return_value = []
            trainer.training = Mock()
            trainer.network = Mock()
            trainer.network.state_dict.return_value = {"s": 1}
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
                trainer = SageMakerTrainer(s3_manager)
                assert trainer.scaler is None

            # When CUDA is available, scaler should be created
            with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.amp.GradScaler"):
                trainer = SageMakerTrainer(s3_manager)
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
            trainer = SageMakerTrainer(s3_manager)

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
                trainer = SageMakerTrainer(s3_manager)

            # Mock components for mixed precision path
            trainer.selfplay = Mock()
            trainer.selfplay.generate_games.return_value = [[("state", "policy", 0.1)]]
            trainer.buffer = Mock()
            trainer.buffer.sample.return_value = [("state", "policy", 0.1)]
            trainer.training = Mock()
            trainer.training.train_step.return_value = 1.5
            trainer.network = Mock()
            trainer.network.state_dict.return_value = {"test": "state"}

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
        """Test if SageMaker role exists"""
        config = load_config()
        role_arn = config["aws"]["credentials"]["role_arn"]

        try:
            iam = boto3.client("iam")
            role_name = role_arn.split("/")[-1]
            iam.get_role(RoleName=role_name)
            print(f"✅ IAM Role: '{role_name}' exists")
        except ClientError as e:
            pytest.fail(f"❌ IAM role '{role_name}' not found: {e}")

    def test_s3_manager_download_checkpoint_exists(self):
        """Verify download_checkpoint method exists"""
        from src.infra.s3_manager import S3Manager

        s3_manager = S3Manager("test-bucket")
        assert hasattr(s3_manager, "download_checkpoint")

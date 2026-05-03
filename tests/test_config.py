import pytest

from src.config import CONFIG, MontyBotConfig


class TestConfiguration:
    def test_config_validation(self):
        """Test config validation"""
        # Test invalid config raises ValidationError
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MontyBotConfig(
                filters=-1,  # Invalid
                blocks=4,
                input_planes=12,
                action_size=4672,
                learning_rate=1e-3,
                batch_size=64,
                iterations=1000,
                mixed_precision=True,
                simulations=50,
                games_per_iteration=4,
                train_steps=2,
                default_instance_type="test",
                framework_version="2.2.0",
                python_version="py310",
                max_run_seconds=3600,
                max_wait_seconds=14400,
                poll_seconds=20,
                job_name="test",
                spot_instances=True,
                region="us-west-2",
                account_id="123",
                role_arn="arn:test",
                bucket="test",
                prefix="test",
                eval_games=20,
                eval_threshold=0.9,
                device="cpu",
            )

    def test_parameter_modification(self):
        """Test parameters can be modified"""
        original = CONFIG.filters
        CONFIG.filters = 256
        assert CONFIG.filters == 256
        CONFIG.filters = original  # Restore

    def test_pydantic_model(self):
        """Test Pydantic model behavior"""
        # Check that it's a Pydantic model
        from pydantic import BaseModel

        assert isinstance(CONFIG, BaseModel)

    def test_grouped_parameters_logically_organized(self):
        """Test parameter groups make sense"""
        # Model architecture group
        assert all(hasattr(CONFIG, attr) for attr in ["filters", "blocks", "input_planes"])

        # Training group
        assert all(hasattr(CONFIG, attr) for attr in ["learning_rate", "batch_size", "iterations"])

        # AWS group
        assert all(hasattr(CONFIG, attr) for attr in ["bucket", "region", "account_id"])

    def test_type_annotations(self):
        """Test type annotations exist"""
        assert hasattr(MontyBotConfig, "__annotations__")
        assert MontyBotConfig.__annotations__["learning_rate"] == float
        assert MontyBotConfig.__annotations__["filters"] == int

    def test_global_config_singleton(self):
        """Test global CONFIG is accessible"""
        from src.config import CONFIG as CONFIG2

        assert CONFIG is CONFIG2

        # Store original value
        original_filters = CONFIG.filters

        # Modifications are global
        CONFIG.filters = 999
        assert CONFIG2.filters == 999

        # Restore original
        CONFIG.filters = original_filters

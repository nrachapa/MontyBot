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
                input_planes=8,
                action_size=133,
                learning_rate=1e-3,
                batch_size=64,
                iterations=1000,
                mixed_precision=True,
                simulations=50,
                c_discount=0.01,
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
                gen_device="cpu",
            )

        # New runtime fields (ADR-0010 / ticket 08)
        assert CONFIG.c_discount == 0.01
        assert CONFIG.gen_device == "cpu"
        # 'auto' must prefer MPS on Apple Silicon (the M6 box is the default,
        # ADR-0010) while the generation device stays CPU (ADR-0007).
        import torch

        if torch.backends.mps.is_available():
            assert MontyBotConfig().device == "mps"
        assert MontyBotConfig().gen_device == "cpu"

    def test_device_resolution_priority(self):
        """'auto' resolves mps > cuda > cpu, and explicit values pass
        through untouched (ticket 09, ADR-0010)."""
        import torch

        expected = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        assert MontyBotConfig().device == expected
        assert MontyBotConfig(device="cpu").device == "cpu"
        assert MontyBotConfig(device="mps").device == "mps"
        assert MontyBotConfig(device="auto", gen_device="mps").gen_device == "mps"

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

    def test_device_consistent_across_module_imports(self):
        """resolve_device() returns the same value as MontyBotConfig().device
        for any non-'auto' device, and CONFIG.device is already resolved."""
        from src.config import CONFIG, resolve_device, MontyBotConfig

        # CONFIG.device should already be resolved (not 'auto')
        assert CONFIG.device != "auto" or resolve_device("auto") in ("cpu", "cuda", "mps")
        assert resolve_device(CONFIG.device) == CONFIG.device

        # Explicit device values should pass through
        for dev in ("cpu", "mps", "cuda"):
            assert resolve_device(dev) == dev

        # 'auto' should resolve to something valid
        resolved = resolve_device("auto")
        assert resolved in ("cpu", "cuda", "mps")

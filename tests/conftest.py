import os

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _env_fast_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTEST_RUNNING"] = "1"  # disable multiproc paths
    np.random.seed(42)
    torch.manual_seed(42)

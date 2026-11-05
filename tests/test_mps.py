"""MPS device tests (ticket 09, ADR-0010). Skipped when Metal is
unavailable (e.g. the CI runner); the config-resolution seam is tested
device-agnostically in test_config.py instead."""

import chess
import numpy as np
import pytest
import torch

from src.config import CONFIG
from src.training.batch_manager import BatchManager
from src.training.game import Position
from src.training.network import AlphaZeroNet
from src.training.training import AlphaZeroTraining

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available (not on Apple Silicon)"
)


def _batch(n: int = 4) -> list:
    """Synthetic trajectory entries following the game state convention:
    a Position (board, color, fen, tensor) — train_step reads s.tensor."""
    rng = np.random.default_rng(0)
    board = chess.Board()
    fen = board.fen()
    batch = []
    for _ in range(n):
        planes = rng.random((CONFIG.input_planes, 8, 8), dtype=np.float32)
        p = rng.random(CONFIG.action_size).astype(np.float32)
        p /= p.sum()
        state = Position(board, 1, fen, planes)
        batch.append((state, p, float(rng.random())))
    return batch


def test_train_step_mps():
    """With the mixed-precision flag ON, an MPS training step must not
    crash (torch 2.2: no MPS autocast, fp16 BatchNorm aborts Metal — the
    trainer falls back to fp32) and must still learn."""
    saved = CONFIG.mixed_precision
    try:
        CONFIG.mixed_precision = True
        net = AlphaZeroNet()
        trainer = AlphaZeroTraining(net, torch.device("mps"))
        before = {k: v.clone() for k, v in net.state_dict().items()}
        loss = trainer.train_step(_batch())
        assert np.isfinite(loss)
        assert any(not torch.equal(before[k], v) for k, v in net.state_dict().items())
    finally:
        CONFIG.mixed_precision = saved

def test_train_step_mps_no_autocast():
    saved = CONFIG.mixed_precision
    CONFIG.mixed_precision = False
    try:
        net = AlphaZeroNet()
        trainer = AlphaZeroTraining(net, torch.device("mps"))
        loss = trainer.train_step(_batch())
        assert np.isfinite(loss)
    finally:
        CONFIG.mixed_precision = saved


def test_batch_manager_mps():
    """A BatchManager pinned to MPS serves an inference request and
    returns correctly-shaped logits/value."""
    net = AlphaZeroNet()
    manager = BatchManager(net, device="mps", batch_size=4, timeout=0.1)
    manager.start()
    try:
        client = manager.create_client()
        x = torch.randn(1, CONFIG.input_planes, 8, 8)
        logits, value = client(x)
        assert logits.shape == (1, CONFIG.action_size)
        assert value.shape == (1, 1)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    finally:
        manager.stop()

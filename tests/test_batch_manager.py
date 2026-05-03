import time
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from src.config import CONFIG
from src.training.batch_manager import BatchClient, BatchManager


class MockNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        batch_size = x.shape[0]
        # Return dummy logits and values
        logits = torch.zeros(batch_size, CONFIG.action_size)
        values = torch.zeros(batch_size, 1)
        return logits, values


class TestBatchManager:
    def test_batching(self):
        net = MockNetwork()
        manager = BatchManager(net, device="cpu", batch_size=2, timeout=0.1)
        manager.start()

        try:
            client1 = manager.create_client()
            client2 = manager.create_client()

            # Send requests
            x = torch.randn(1, CONFIG.input_planes, 8, 8)

            # We need to run clients in threads or just call them?
            # BatchClient blocks on get(), so we can't call them sequentially in main thread
            # if we want them to be batched together, unless we put requests first.
            # But BatchClient.__call__ puts then gets.

            # So we need threads for clients to simulate concurrent requests
            import threading

            results = {}

            def call_client(c, cid):
                results[cid] = c(x)

            t1 = threading.Thread(target=call_client, args=(client1, 1))
            t2 = threading.Thread(target=call_client, args=(client2, 2))

            t1.start()
            t2.start()

            t1.join()
            t2.join()

            assert 1 in results
            assert 2 in results

            # Check that network was called once (batched)
            # Note: exact timing might make it 1 or 2, but with sufficient timeout it should be 1
            assert net.call_count == 1

        finally:
            manager.stop()

    def test_timeout(self):
        net = MockNetwork()
        manager = BatchManager(net, device="cpu", batch_size=10, timeout=0.05)
        manager.start()

        try:
            client = manager.create_client()
            x = torch.randn(1, CONFIG.input_planes, 8, 8)

            start = time.time()
            _ = client(x)  # Call to test timing, result unused
            end = time.time()

            # Should have waited for timeout
            assert end - start >= 0.05
            assert net.call_count == 1

        finally:
            manager.stop()

    def test_error_propagation(self):
        """Test that exceptions in inference are propagated to clients"""
        net = MockNetwork()
        # Make network raise exception
        net.forward = Mock(side_effect=RuntimeError("GPU Error"))

        manager = BatchManager(net, device="cpu", batch_size=1, timeout=0.1)
        manager.start()

        try:
            client = manager.create_client()
            x = torch.randn(1, CONFIG.input_planes, 8, 8)

            with pytest.raises(RuntimeError, match="GPU Error"):
                client(x)

        finally:
            manager.stop()

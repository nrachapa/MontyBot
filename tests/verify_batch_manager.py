import multiprocessing as mp
import time

import numpy as np
import torch

from src.training.batch_manager import BatchManager


class MockNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)

    def forward(self, x):
        # Simulate some work
        time.sleep(0.001)
        return torch.randn(x.shape[0], 2), torch.randn(x.shape[0], 1)


def worker(client, n_requests):
    for _ in range(n_requests):
        state = torch.randn(1, 10)
        client(state)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    network = MockNetwork()
    manager = BatchManager(network, "cpu", batch_size=32, timeout=0.01)
    manager.start()

    try:
        n_workers = 4
        n_requests = 100

        clients = []
        for _ in range(n_workers):
            clients.append(manager.create_client())

        start = time.time()
        processes = []
        for client in clients:
            p = mp.Process(target=worker, args=(client, n_requests))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end = time.time()
        print(f"Processed {n_workers * n_requests} requests in {end - start:.4f}s")
        print("Verification successful!")

    finally:
        manager.stop()

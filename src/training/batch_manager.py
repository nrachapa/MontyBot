import multiprocessing as mp
import queue
import threading
import time
from typing import Any

import numpy as np
import torch

from ..config import CONFIG


class BatchClient:
    """
    Client that mimics a Network but forwards requests to a BatchManager.
    """

    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, client_id: int):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.client_id = client_id

    def __call__(self, state_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mimics the forward pass of the network.
        state_tensor: (1, C, H, W) tensor
        Returns: (logits, value)
        """
        # Send request
        state_np = state_tensor.cpu().numpy()
        self.input_queue.put((self.client_id, state_np))

        # Wait for response
        response = self.output_queue.get()

        if isinstance(response, Exception):
            raise response

        logits_np, value_np = response
        return torch.from_numpy(logits_np), torch.from_numpy(value_np)


class BatchManager:
    """
    Manages batching of inference requests.
    """

    def __init__(self, network: Any, device: str, batch_size: int = None, timeout: float = 0.05):
        self.network = network
        self.device = device
        self.batch_size = batch_size or CONFIG.batch_size
        self.timeout = timeout
        self.running = False
        self.thread = None

        # Queues - use Manager for cross-process compatibility
        # Manager queues can be pickled and shared across processes
        self.manager = mp.Manager()
        self.input_queue = self.manager.Queue()
        self.output_queues = {}

    def create_client(self) -> BatchClient:
        client_id = len(self.output_queues)
        q = self.manager.Queue()
        self.output_queues[client_id] = q
        return BatchClient(self.input_queue, q, client_id)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        # No manager to shutdown

    def _loop(self):
        self.network.eval()
        self.network.to(self.device)

        while self.running:
            batch = []
            ids = []

            # Collect batch
            start_time = time.monotonic()
            while len(batch) < self.batch_size:
                try:
                    # Calculate remaining time in timeout window
                    elapsed = time.monotonic() - start_time
                    wait = self.timeout - elapsed
                    
                    # If timeout expired and we have at least one item, process the batch
                    if wait <= 0:
                        if len(batch) > 0:
                            break
                        # If batch is empty, don't wait (just check queue)
                        wait = 0

                    item = self.input_queue.get(timeout=wait)
                    cid, state = item
                    batch.append(state)
                    ids.append(cid)
                except queue.Empty:
                    # Queue is empty - if we have items, process what we have
                    if len(batch) > 0:
                        break
                    # If batch is empty, break to outer loop which will skip processing
                    # and continue to next iteration (checking self.running flag)
                    break

            if not batch:
                continue

            # Process batch
            try:
                # Stack states: List[(1, C, H, W)] -> (B, C, H, W)
                # state is (1, C, H, W) numpy array
                states = np.concatenate(batch, axis=0)
                tensor = torch.from_numpy(states).to(self.device)

                with torch.no_grad():
                    logits, values = self.network(tensor)

                logits_np = logits.cpu().numpy()
                values_np = values.cpu().numpy()

                # Distribute results
                for i, cid in enumerate(ids):
                    # Slice back to (1, ...)
                    logits_slice = logits_np[i : i + 1]
                    v = values_np[i : i + 1]
                    self.output_queues[cid].put((logits_slice, v))

            except Exception as e:
                # Send error to all waiting clients
                for cid in ids:
                    self.output_queues[cid].put(e)
                print(f"Batch inference error: {e}")

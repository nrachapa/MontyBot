import numpy as np
import torch

from src.config import CONFIG
from src.training.game import ChessGame
from src.training.search import MCTS, Node


class MockNetwork:
    def __call__(self, x):
        batch_size = x.shape[0]
        logits = torch.randn(batch_size, CONFIG.action_size)
        value = torch.randn(batch_size, 1)
        return logits, value


def verify_subtree_reuse():
    CONFIG.simulations = 10
    game = ChessGame()
    network = MockNetwork()
    mcts = MCTS(game, network)

    # 1. First search
    state = game.get_initial_state()
    policy, root = mcts.search(state)

    print(f"Root visits: {root.visit_count}")
    assert root.visit_count >= CONFIG.simulations

    # 2. Pick a move
    move_idx = np.argmax(policy)
    next_state = game.apply_move(state, move_idx)

    # 3. Get child node to reuse
    child_node = root.children[move_idx]
    previous_visits = child_node.visit_count
    print(f"Child visits (before reuse): {previous_visits}")
    assert previous_visits > 0

    # 4. Second search with reuse
    policy_2, new_root = mcts.search(next_state, root=child_node)

    print(f"New root visits: {new_root.visit_count}")

    # 5. Verify reuse
    # The new root is the same object as child_node
    assert new_root is child_node
    # Visits should have increased by simulations
    expected_visits = previous_visits + CONFIG.simulations
    # Note: It might be slightly different if we add noise or something, but generally yes.
    # Actually, MCTS adds 1 visit per simulation.
    # So it should be exactly previous + simulations.
    assert new_root.visit_count == expected_visits, f"Expected {expected_visits}, got {new_root.visit_count}"

    print("Verification successful: Subtree reuse works correctly.")


if __name__ == "__main__":
    verify_subtree_reuse()

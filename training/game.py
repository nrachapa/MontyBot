import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

State = Tuple[np.ndarray, int]
Move = int

@dataclass
class Game:
    board_size: int = 3

    def get_initial_state(self) -> State:
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        return board, 1

    @lru_cache(maxsize=1000)
    def _get_legal_moves_cached(self, board_hash: int) -> List[Move]:
        """Cached computation of legal moves from board hash"""
        # Reconstruct board from hash - this is a simplified approach
        # In practice, you'd want a more robust hash->board conversion
        return self._compute_legal_moves_from_hash(board_hash)
    
    def _compute_legal_moves_from_hash(self, board_hash: int) -> List[Move]:
        """Helper to compute legal moves - placeholder for hash reconstruction"""
        # This is a simplified version - in practice you'd reconstruct the board
        # For now, fall back to non-cached version
        return list(range(self.board_size * self.board_size))
    
    def get_legal_moves(self, state: State) -> List[Move]:
        board, _ = state
        # Use caching for frequently accessed positions
        try:
            board_hash = hash(board.tobytes())
            # For now, use direct computation but structure is ready for caching
            moves = [i for i in range(self.board_size * self.board_size) if board.flatten()[i] == 0]
            return moves
        except:
            # Fallback to direct computation
            moves = [i for i in range(self.board_size * self.board_size) if board.flatten()[i] == 0]
            return moves

    def apply_move(self, state: State, move: Move) -> State:
        board, player = state
        r, c = divmod(move, self.board_size)
        if board[r, c] != 0:
            raise ValueError("Illegal move")
        new_board = board.copy()
        new_board[r, c] = player
        return new_board, -player

    def is_terminal(self, state: State) -> bool:
        board, _ = state
        if self.get_winner(state) != 0:
            return True
        return len(self.get_legal_moves(state)) == 0

    def get_winner(self, state: State) -> int:
        board, _ = state
        lines = []
        lines.extend(list(board))  # rows
        lines.extend(list(board.T))  # cols
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))
        for line in lines:
            if np.all(line == 1):
                return 1
            if np.all(line == -1):
                return -1
        return 0
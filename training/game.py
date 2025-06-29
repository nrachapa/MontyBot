import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

State = Tuple[np.ndarray, int]
Move = int

@dataclass
class Game:
    board_size: int = 3

    def get_initial_state(self) -> State:
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        return board, 1

    def get_legal_moves(self, state: State) -> List[Move]:
        board, _ = state
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
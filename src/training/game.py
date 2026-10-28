import chess
import numpy as np

from ..core import Game

# --- 8-plane representation (ADR-0006) ---
_PIECE_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

CASTLING_PLANE = 6
ENPASSANT_PLANE = 7


def _king_square(board: chess.Board, color) -> int | None:
    return next(iter(board.pieces(chess.KING, color)), None)


def encode_board(board: chess.Board) -> np.ndarray:
    """8 planes (ADR-0006): 6 piece-type planes signed by side (+1 white /
    -1 black), 1 castling-rights plane (1 at the king's d-file square if that
    side retains any right), 1 en-passant target plane. Current board only —
    no history planes (rejected in ADR-0006)."""
    x = np.zeros((8, 8, 8), dtype=np.float32)
    for sq in range(64):
        pc = board.piece_at(chess.SQUARES[sq])
        if pc is None:
            continue
        x[_PIECE_PLANE[pc.piece_type], sq // 8, sq % 8] = 1.0 if pc.color == chess.WHITE else -1.0
    if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
        x[CASTLING_PLANE, 0, 3] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
        x[CASTLING_PLANE, 7, 3] = 1.0
    if board.ep_square is not None:
        x[ENPASSANT_PLANE, board.ep_square // 8, board.ep_square % 8] = 1.0
    return x


class MoveMapper:
    """133 actions (ADR-0006): 64 quiet + 64 captures, keyed by destination
    square, + 5 castles. Multiple concrete moves can map to one action;
    resolution is against python-chess legal moves (first match wins).
    Promotion piece is chosen at execution, not in the action space."""

    ACTION_SIZE = 133
    CAPTURE_OFFSET = 64
    CASTLE_OFFSET = 128
    _PROMO_PIECE = chess.QUEEN

    @staticmethod
    def to_index(move: chess.Move, board: chess.Board) -> int:
        if board.is_castling(move):
            wk, bk = _king_square(board, chess.WHITE), _king_square(board, chess.BLACK)
            if move.from_square == wk:
                return MoveMapper.CASTLE_OFFSET + (0 if move.to_square == wk + 2 else 1)
            return MoveMapper.CASTLE_OFFSET + (2 if move.to_square == bk + 2 else 3)
        if board.is_capture(move):
            return MoveMapper.CAPTURE_OFFSET + move.to_square
        return move.to_square

    @staticmethod
    def from_index(idx: int, board: chess.Board) -> chess.Move:
        """Resolve an action to a concrete legal move. Raises if the action
        is not legal here — a silent fallback would poison self-play data."""
        if idx >= MoveMapper.CASTLE_OFFSET:
            if idx >= MoveMapper.ACTION_SIZE:
                raise ValueError(f"action {idx} out of range")
            wk, bk = _king_square(board, chess.WHITE), _king_square(board, chess.BLACK)
            # Kingside is a two-file jump (e1→g1); the king never moves one.
            targets = [(wk, wk + 2), (wk, wk - 2), (bk, bk + 2), (bk, bk - 2)]
            want = targets[idx - MoveMapper.CASTLE_OFFSET]
            for m in board.legal_moves:
                if (m.from_square, m.to_square) == want:
                    return m
        else:
            want_capture = idx >= MoveMapper.CAPTURE_OFFSET
            to = idx % 64
            for m in board.legal_moves:
                if m.to_square == to and board.is_capture(m) == want_capture:
                    return m
        raise ValueError(f"action {idx} has no legal move on this board")


class ChessGame(Game):
    """Real 8x8 chess. State: (planes, color, fen, board) with color
    +1 white / -1 black; the board object lets the hot path skip FEN
    re-parsing. Termination (ADR-0006): mate, stalemate, threefold,
    insufficient material, 50/75-move rule, 150-halfmove hard cap."""

    HALFMOVE_CAP = 150

    def get_initial_state(self):
        b = chess.Board()
        return (encode_board(b), 1, b.fen(), b)

    def _board(self, state) -> chess.Board:
        if len(state) > 3 and isinstance(state[3], chess.Board):
            return state[3]
        return chess.Board(state[2])

    def get_legal_moves(self, state) -> list[int]:
        b = self._board(state)
        return list(dict.fromkeys(MoveMapper.to_index(m, b) for m in b.legal_moves))

    def apply_move(self, state, move_idx: int):
        b = self._board(state)
        if self.is_terminal(state):
            return state
        m = MoveMapper.from_index(move_idx, b)
        b2 = b.copy()  # never mutate the stored state (parents/trajectory keep it)
        b2.push(m)
        return (encode_board(b2), -state[1], b2.fen(), b2)

    def is_terminal(self, state) -> bool:
        b = self._board(state)
        return b.is_game_over(claim_draw=True) or b.halfmove_clock >= self.HALFMOVE_CAP

    def get_winner(self, state) -> int:
        """+1 white, -1 black, 0 draw (from white's perspective)."""
        b = self._board(state)
        if b.halfmove_clock >= self.HALFMOVE_CAP:
            return 0
        res = b.result(claim_draw=True)
        if res == "1-0":
            return 1
        if res == "0-1":
            return -1
        return 0

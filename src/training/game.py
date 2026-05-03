import chess
import numpy as np

from ..core import Game

# --- Encoder & Move Mapping ---
PIECE_PLANES = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]  # 12 planes


class MoveMapper:
    """Fixed-size move index space: 4096 (from,to) + 576 promotions = 4672 total."""

    ACTION_SIZE = 4672

    @staticmethod
    def to_index(move: chess.Move) -> int:
        """Convert chess.Move to action index."""
        base = 64 * move.from_square + move.to_square
        if move.promotion:
            # Add promotion offset: q=0, r=1, b=2, n=3
            promo_map = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
            offset = promo_map.get(move.promotion, 0)
            base += 4096 + offset * 144  # 144 = 64*64/64 approx promotion squares
        return min(base, MoveMapper.ACTION_SIZE - 1)

    @staticmethod
    def from_index(idx: int) -> chess.Move:
        """Convert action index to chess.Move (best-effort)."""
        if idx >= 4096:
            # Promotion move
            promo_idx = (idx - 4096) // 144
            promo_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            promotion = promo_pieces[min(promo_idx, 3)]
            base_idx = (idx - 4096) % 144
            from_sq = base_idx // 8
            to_sq = base_idx % 8 + 56  # Assume 8th rank promotion
            return chess.Move(from_sq, to_sq, promotion=promotion)
        from_sq = idx // 64
        to_sq = idx % 64
        return chess.Move(from_sq, to_sq)


def encode_board(board: chess.Board, history: list[chess.Board] = None) -> np.ndarray:
    """
    Encode board state into planes.
    Standard AlphaZero-like encoding:
    - History: T=8 steps (current + 7 prev).
    - Per step: 12 planes (P,N,B,R,Q,K for both colors).
    - Total history planes: 8 * 12 = 96.
    - Auxiliary planes:
        - Color (1): 1 if White, 0 if Black.
        - Castling (4): WK, WQ, BK, BQ.
        - Repetition (1): count.
        - En Passant (1): 1 at target square.
        - Halfmove clock (1): normalized.
    Total: 96 + 8 = 104 planes.
    """
    HISTORY_LEN = 8
    PLANES_PER_BOARD = 12
    TOTAL_PLANES = HISTORY_LEN * PLANES_PER_BOARD + 8

    planes = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)

    # 1. History Planes
    if history is None:
        history = [board]

    # Pad history if needed
    boards_to_process = history[-HISTORY_LEN:]
    if len(boards_to_process) < HISTORY_LEN:
        # Pad with the oldest board (or empty? AlphaZero pads with zeros usually, but repeating oldest is safer for implementation)
        # Actually, standard is to pad with zeros (empty boards).
        padding = [None] * (HISTORY_LEN - len(boards_to_process))
        boards_to_process = padding + boards_to_process

    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for t, b in enumerate(reversed(boards_to_process)):
        if b is None:
            continue

        base_idx = t * PLANES_PER_BOARD
        for p_idx, symbol in enumerate(PIECE_PLANES):
            pt = piece_types[p_idx % 6]
            color = chess.WHITE if symbol.isupper() else chess.BLACK
            for sq in b.pieces(pt, color):
                r, c = divmod(sq, 8)
                planes[base_idx + p_idx, 7 - r, c] = 1.0

    # 2. Auxiliary Planes (based on current board)
    aux_idx = HISTORY_LEN * PLANES_PER_BOARD

    # Color
    if board.turn == chess.WHITE:
        planes[aux_idx, :, :] = 1.0
    aux_idx += 1

    # Castling
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[aux_idx, :, :] = 1.0
    aux_idx += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[aux_idx, :, :] = 1.0
    aux_idx += 1
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[aux_idx, :, :] = 1.0
    aux_idx += 1
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[aux_idx, :, :] = 1.0
    aux_idx += 1

    # Repetition (simplified: just check if current board appeared before)
    # Ideally we need full repetition count.
    # For now, let's use is_repetition(2) and (3)
    if board.is_repetition(2):
        planes[aux_idx, :, :] = 0.5
    if board.is_repetition(3):
        planes[aux_idx, :, :] = 1.0
    aux_idx += 1

    # En Passant
    if board.ep_square:
        r, c = divmod(board.ep_square, 8)
        planes[aux_idx, 7 - r, c] = 1.0
    aux_idx += 1

    # Halfmove clock (50-move rule)
    planes[aux_idx, :, :] = board.halfmove_clock / 100.0
    aux_idx += 1

    return planes


class ChessGame(Game):
    def __init__(self):
        pass

    def get_initial_state(self):
        b = chess.Board()
        # State: (planes, color, fen, history_fens, board)
        # We store history as FENs to be pickleable/serializable easily
        # We also cache the board object to avoid re-parsing FENs
        return (encode_board(b, [b]), 1, b.fen(), [b.fen()], b)

    def _legal_indices(self, b: chess.Board) -> list[int]:
        return [MoveMapper.to_index(m) for m in b.legal_moves]

    def get_legal_moves(self, state):
        if len(state) > 4:
            b = state[4]
        else:
            fen = state[2]
            b = chess.Board(fen)
        return self._legal_indices(b)

    def apply_move(self, state, move_idx: int):
        history_fens = state[3]

        if len(state) > 4:
            b = state[4].copy()
        else:
            fen = state[2]
            b = chess.Board(fen)

        m = MoveMapper.from_index(move_idx)

        if m not in b.legal_moves:
            if b.legal_moves.count() > 0:
                m = next(iter(b.legal_moves))
            else:
                return state

        b.push(m)

        # Update history
        new_history = history_fens + [b.fen()]
        # Keep last 8
        if len(new_history) > 8:
            new_history = new_history[-8:]

        # Reconstruct board objects for encoding
        # This is slightly inefficient but safe.
        # Optimization: cache encoded history?
        history_boards = [chess.Board(f) for f in new_history]

        next_color = -state[1]
        return (encode_board(b, history_boards), next_color, b.fen(), new_history, b)

    def is_terminal(self, state):
        if len(state) > 4:
            b = state[4]
        else:
            fen = state[2]
            b = chess.Board(fen)
        return b.is_game_over()

    def get_winner(self, state):
        if len(state) > 4:
            b = state[4]
        else:
            fen = state[2]
            b = chess.Board(fen)
        res = b.result(claim_draw=True)
        if res == "1-0":
            return 1
        if res == "0-1":
            return -1
        return 0

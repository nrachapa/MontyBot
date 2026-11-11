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

# Board-tensor cache: maps (fen, color) -> np.ndarray.  Avoids re-encoding
# boards that reappear (common in repeated lines / transpositions).
_ENCODE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def encode_board_cached(board: chess.Board, color: int) -> np.ndarray:
    """Encode board, caching by (FEN, side-to-move color).

    The paper (Sec. 5.1) identifies board encoding as part of the Python
    dispatch bottleneck.  This cache eliminates the encoding when a
    position reappears on the tree (e.g. repeated lines).
    """
    fen = board.fen()
    key = (fen, color)
    cached = _ENCODE_CACHE.get(key)
    if cached is not None:
        return cached
    tensor = encode_board(board)
    if len(_ENCODE_CACHE) < 500_000:
        _ENCODE_CACHE[key] = tensor
    return tensor


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


class Position:
    """The state value object behind the ``Game`` seam — one representation
    for the four things every consumer reads: the python-chess ``board``
    (authoritative: legal moves, FEN, terminal and winner all derive from it,
    so the hot path skips FEN re-parsing), ``color`` (+1 white / -1 black, the
    net's input convention, flipped per ply), ``fen`` (kept for logging/debug),
    and ``tensor`` (the encoded 8-plane board, cached). Replaces the old
    positional tuples — two divergent shapes, ``(tensor, color, fen, board)``
    and ``(tensor, color, fen, None, board)`` — that callers told apart with
    ``len()`` guards and index reads. ``tensor`` may be None for heuristic-only
    positions that never feed the network."""

    __slots__ = ("board", "color", "fen", "tensor")

    def __init__(self, board: chess.Board, color: int, fen: str,
                 tensor: np.ndarray | None) -> None:
        self.board = board
        self.color = color
        self.fen = fen
        self.tensor = tensor


class GameTerminated(Exception):  # noqa: N818
    """Raised by a policy_fn to signal the game should end early.

    Used when the search returns a zero policy (no legal moves found),
    which can happen even if the board hasn't reached a formal terminal
    state (e.g. stalemate discovered by search but not yet by the game).
    """
    pass


def play_game(
    game: "Game",
    policy_fn,
    opponent_fn=None,
    rng=None,
    max_moves=200,
):
    """Play a game to completion.

    The caller provides **policy_fn** and optionally **opponent_fn** —
    each is a callable(state) → int (move index).  play_game owns the
    loop: termination via game.is_terminal, state transitions via
    game.apply_move, and the optional move cap.

    Returns
        (trajectory: list[Position], winner: int, is_capped: bool)
        — winner is +1/−1/0; is_capped signals the move limit was hit
        (the game was not decided naturally; caller may do a cutoff
        search — paper Sec. 4.2.6).
    """
    state = game.get_initial_state()
    trajectory = []
    move_count = 0
    is_capped = False

    while not game.is_terminal(state) and move_count < max_moves:
        try:
            move = opponent_fn(state) if opponent_fn is not None and move_count % 2 == 1 else policy_fn(state)
        except GameTerminated:
            break
        trajectory.append(state)
        state = game.apply_move(state, move)
        move_count += 1
    else:
        # Reached max_moves without terminal state.
        is_capped = True
        winner = 0  # capped games count as draws
        return trajectory, winner, is_capped

    winner = game.get_winner(state)
    return trajectory, winner, is_capped


class ChessGame(Game):
    """Real 8x8 chess. State is a ``Position``; the carried board object lets
    the hot path skip FEN re-parsing. Termination (ADR-0006): mate, stalemate,
    threefold, insufficient material, 50/75-move rule, 150-halfmove hard cap."""

    HALFMOVE_CAP = 150

    def get_initial_state(self) -> Position:
        b = chess.Board()
        return Position(board=b, color=1, fen=b.fen(), tensor=encode_board(b))

    def get_legal_moves(self, state: Position) -> list[int]:
        return list(dict.fromkeys(MoveMapper.to_index(m, state.board)
                                  for m in state.board.legal_moves))

    def apply_move(self, state: Position, move_idx: int) -> Position:
        if self.is_terminal(state):
            return state
        m = MoveMapper.from_index(move_idx, state.board)
        b = state.board.copy()  # never mutate the stored state (parents/trajectory keep it)
        b.push(m)
        return Position(board=b, color=-state.color, fen=b.fen(), tensor=encode_board(b))

    def is_terminal(self, state: Position) -> bool:
        return state.board.is_game_over(claim_draw=True) or state.board.halfmove_clock >= self.HALFMOVE_CAP

    def get_winner(self, state: Position) -> int:
        """+1 white, -1 black, 0 draw (from white's perspective)."""
        b = state.board
        if b.halfmove_clock >= self.HALFMOVE_CAP:
            return 0
        res = b.result(claim_draw=True)
        if res == "1-0":
            return 1
        if res == "0-1":
            return -1
        return 0

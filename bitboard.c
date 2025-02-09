#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>

// --------------------------
// Type Definitions
// --------------------------

typedef uint64_t U64;

typedef enum
{
    COLOR_WHITE,
    COLOR_BLACK,
    COLOR_NONE
} Color;

typedef enum
{
    PIECE_PAWN,
    PIECE_KNIGHT,
    PIECE_BISHOP,
    PIECE_ROOK,
    PIECE_QUEEN,
    PIECE_KING,
    PIECE_NONE
} PieceType;

typedef enum
{
    A1,
    B1,
    C1,
    D1,
    E1,
    F1,
    G1,
    H1,
    A2,
    B2,
    C2,
    D2,
    E2,
    F2,
    G2,
    H2,
    A3,
    B3,
    C3,
    D3,
    E3,
    F3,
    G3,
    H3,
    A4,
    B4,
    C4,
    D4,
    E4,
    F4,
    G4,
    H4,
    A5,
    B5,
    C5,
    D5,
    E5,
    F5,
    G5,
    H5,
    A6,
    B6,
    C6,
    D6,
    E6,
    F6,
    G6,
    H6,
    A7,
    B7,
    C7,
    D7,
    E7,
    F7,
    G7,
    H7,
    A8,
    B8,
    C8,
    D8,
    E8,
    F8,
    G8,
    H8,
    SQUARE_NONE
} Square;

typedef enum
{
    BB_WHITE,
    BB_BLACK,
    BB_PAWN,
    BB_KNIGHT,
    BB_BISHOP,
    BB_ROOK,
    BB_QUEEN,
    BB_KING,
    BB_COUNT
} BitboardType;

typedef struct
{
    U64 bitboards[BB_COUNT];
} Board;

/* Board management */
void board_init(Board *board);
void board_setup_initial_position(Board *board);
void board_set_piece(Board *board, Square square, PieceType piece, Color color);
void board_remove_piece(Board *board, Square square, PieceType piece, Color color);
void board_move_piece(Board *board, Square from, Square to, PieceType piece, Color color);

/* Board query */
PieceType board_get_piece_type_at(const Board *board, Square square);
Color board_get_color_at(const Board *board, Square square);

/* Visualization */
void board_print(const Board *board);

/* Utility functions */
Square square_from_string(const char *str);

// --------------------------
// Board Implementation
// --------------------------

void board_init(Board *board)
{
    memset(board, 0, sizeof(Board));
}

void board_set_piece(Board *board, Square square, PieceType piece, Color color)
{
    // Clear existing piece
    PieceType existing_piece = board_get_piece_type_at(board, square);
    Color existing_color = board_get_color_at(board, square);
    if (existing_piece != PIECE_NONE)
    {
        board->bitboards[existing_piece + 2] &= ~(1ULL << square);
        board->bitboards[existing_color] &= ~(1ULL << square);
    }

    // Set new piece
    board->bitboards[piece + 2] |= (1ULL << square);
    board->bitboards[color] |= (1ULL << square);
}

void board_remove_piece(Board *board, Square square, PieceType piece, Color color)
{
    board->bitboards[piece + 2] &= ~(1ULL << square);
    board->bitboards[color] &= ~(1ULL << square);
}

void board_move_piece(Board *board, Square from, Square to, PieceType piece, Color color)
{
    // Remove captured piece
    Color opponent = (color == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE;
    PieceType captured = board_get_piece_type_at(board, to);
    if (captured != PIECE_NONE)
    {
        board_remove_piece(board, to, captured, opponent);
    }

    // Move the piece
    board_remove_piece(board, from, piece, color);
    board_set_piece(board, to, piece, color);
}

PieceType board_get_piece_type_at(const Board *board, Square square)
{
    for (PieceType pt = PIECE_PAWN; pt <= PIECE_KING; pt++)
    {
        if (board->bitboards[pt + 2] & (1ULL << square))
        {
            return pt;
        }
    }
    return PIECE_NONE;
}

Color board_get_color_at(const Board *board, Square square)
{
    if (board->bitboards[BB_WHITE] & (1ULL << square))
        return COLOR_WHITE;
    if (board->bitboards[BB_BLACK] & (1ULL << square))
        return COLOR_BLACK;
    return COLOR_NONE;
}

void board_setup_initial_position(Board *board)
{
    board_init(board);

    // Pawns
    for (int file = 0; file < 8; file++)
    {
        board_set_piece(board, A2 + file, PIECE_PAWN, COLOR_WHITE);
        board_set_piece(board, A7 + file, PIECE_PAWN, COLOR_BLACK);
    }

    // White pieces
    board_set_piece(board, A1, PIECE_ROOK, COLOR_WHITE);
    board_set_piece(board, H1, PIECE_ROOK, COLOR_WHITE);
    board_set_piece(board, B1, PIECE_KNIGHT, COLOR_WHITE);
    board_set_piece(board, G1, PIECE_KNIGHT, COLOR_WHITE);
    board_set_piece(board, C1, PIECE_BISHOP, COLOR_WHITE);
    board_set_piece(board, F1, PIECE_BISHOP, COLOR_WHITE);
    board_set_piece(board, D1, PIECE_QUEEN, COLOR_WHITE);
    board_set_piece(board, E1, PIECE_KING, COLOR_WHITE);

    // Black pieces
    board_set_piece(board, A8, PIECE_ROOK, COLOR_BLACK);
    board_set_piece(board, H8, PIECE_ROOK, COLOR_BLACK);
    board_set_piece(board, B8, PIECE_KNIGHT, COLOR_BLACK);
    board_set_piece(board, G8, PIECE_KNIGHT, COLOR_BLACK);
    board_set_piece(board, C8, PIECE_BISHOP, COLOR_BLACK);
    board_set_piece(board, F8, PIECE_BISHOP, COLOR_BLACK);
    board_set_piece(board, D8, PIECE_QUEEN, COLOR_BLACK);
    board_set_piece(board, E8, PIECE_KING, COLOR_BLACK);
}

void board_print(const Board *board)
{
    const char piece_chars[] = {'P', 'N', 'B', 'R', 'Q', 'K'};

    printf("\n  +---+---+---+---+---+---+---+---+\n");
    for (int rank = 7; rank >= 0; rank--)
    {
        printf("%d |", rank + 1);
        for (int file = 0; file < 8; file++)
        {
            Square sq = rank * 8 + file;
            PieceType pt = board_get_piece_type_at(board, sq);
            Color col = board_get_color_at(board, sq);

            char c = '.';
            if (pt != PIECE_NONE)
            {
                c = piece_chars[pt];
                if (col == COLOR_BLACK)
                    c += 32; // Lowercase for black
            }
            printf(" %c |", c);
        }
        printf("\n  +---+---+---+---+---+---+---+---+\n");
    }
    printf("    a   b   c   d   e   f   g   h\n\n");
}

// --------------------------
// Utility Functions
// --------------------------

Square square_from_string(const char *str)
{
    if (strlen(str) != 2)
        return SQUARE_NONE;
    int file = tolower(str[0]) - 'a';
    int rank = str[1] - '1';
    if (file < 0 || file > 7 || rank < 0 || rank > 7)
        return SQUARE_NONE;
    return (Square)(rank * 8 + file);
}

// --------------------------
// Main Application
// --------------------------

int main()
{
    Board board;
    board_init(&board);
    board_setup_initial_position(&board);

    Color current_player = COLOR_WHITE;
    char input[10];

    while (true)
    {
        board_print(&board);

        printf("%s to move. Enter command (from to) or 'quit': ", current_player == COLOR_WHITE ? "White" : "Black");

        if (fgets(input, sizeof(input), stdin) == NULL)
            break;
        input[strcspn(input, "\n")] = '\0';

        if (strcmp(input, "quit") == 0)
            break;
        if (strcmp(input, "print") == 0)
            continue;

        char from_str[3], to_str[3];
        Square from, to;

        if (sscanf(input, "%2s %2s", from_str, to_str) == 2)
        {
            from = square_from_string(from_str);
            to = square_from_string(to_str);

            if (from == SQUARE_NONE || to == SQUARE_NONE)
            {
                printf("Invalid squares! Use format like 'e2 e4'\n");
                continue;
            }

            PieceType piece = board_get_piece_type_at(&board, from);
            Color color = board_get_color_at(&board, from);

            if (piece == PIECE_NONE || color != current_player)
            {
                printf("No valid piece at source square!\n");
                continue;
            }

            board_move_piece(&board, from, to, piece, color);
            current_player = (current_player == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE;
        }
        else
        {
            printf("Invalid command! Use format 'e2 e4' or 'quit'\n");
        }
    }

    return 0;
}
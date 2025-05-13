#include "engine.h"
#include "moveGen.h"

std::string findBestMove(Bitboards &position, int depth)
{
    auto moves = generateLegalMoves(position);
    return moves.empty() ? "" : moves[0];
}

int alphaBeta(Bitboards &position, int depth, int alpha, int beta)
{
    if (depth == 0) {
        return evaluate(position);
    }

    auto moves = generateLegalMoves(position);

    for (const auto& move: moves) {
        int score = -1 * alphaBeta(position, depth - 1, -1 * beta, -1 * alpha);
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }

    return alpha;
}

int evaluate(const Bitboards &position)
{
    return 0;
}

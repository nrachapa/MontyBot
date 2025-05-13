#include "moveGen.h"

std::vector<std::string> generateLegalMoves(const Bitboards &position) {
    std::vector<std::string> moves;
    if (position.white_move) {
        moves.push_back("e2e4");
    } else {
        moves.push_back("e7e5");
    }

    return moves;
}
#pragma once
#include <cstdint>

using U64 = uint64_t;

enum Piece {
    WP, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK,
    EMPTY
};

struct Bitboards {
    U64 pieces[13], whiteOccupancy, blackOccupancy;
    bool white_move;
    void clear() {
        for (int i = 0; i < 13; ++i) pieces[i] = 0;
        whiteOccupancy = blackOccupancy = 0;
        white_move = true;
    }
};
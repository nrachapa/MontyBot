#pragma once
#include "bitboard.h"
#include <string>

std::string findBestMove(Bitboards& position, int depth = 3);
int alphaBeta(Bitboards& position, int depth, int alpha, int beta);
int evaluate(const Bitboards& position);
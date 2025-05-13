#pragma once 
#include "httplib.h"
#include "bitboard.h"
#include "engine.h"
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <mutex>
#include <string>
#include <iostream>


void startRestServer();
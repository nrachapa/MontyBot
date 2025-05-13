#include "restServer.h"

using json = nlohmann::json;
using namespace httplib;

std::unordered_map<std::string, Bitboards> games;
std::mutex globalMutex;

void startRestServer() {
    Server svr;

    svr.Post("/new-game", [](const Request& req, Response& resp) {
        Bitboards board;
        board.clear();

        std::string id = "game1"; // TODO: Generate real UUID

        {
            std::lock_guard<std::mutex> lock(globalMutex);
            games[id] = board;
        }

        json response = { {"gameId", id} };
        resp.set_content(response.dump(), "application/json");
    });

    svr.Post("/move", [](const Request& req, Response& resp) {
        auto data = json::parse(req.body);
        std::string gameId = data["gameId"];
        std::string from = data["from"];
        std::string to = data["to"];

        Bitboards position;
        {
            std::lock_guard<std::mutex> lock(globalMutex);
            if (games.find(gameId) != games.end()) position = games[gameId];
        }
        // Apply move
        std::string best = findBestMove(position);
        json response = {
            {"bestMove", best},
            {"eval", 0.0}
        };
        
        resp.set_content(response.dump(), "application/json");
    });

    std::cout << "Server running on http://localhost:8080\n";

    svr.listen("0.0.0.0", 8080);
}
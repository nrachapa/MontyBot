cmake_minimum_required(VERSION 3.16)
project(MontyBot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(nlohmann_json CONFIG REQUIRED)

file(GLOB SRC_FILES CONFIGURE_DEPENDS src/*.cpp)
add_executable(chess_engine ${SRC_FILES})

target_include_directories(chess_engine PRIVATE include)

target_link_libraries(chess_engine PRIVATE nlohmann_json::nlohmann_json)
cmake_minimum_required(VERSION 3.16)
project(Spaghetti)

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pthread")
set(CMAKE_CXX_STANDARD 17)

add_executable(Spaghetti main.cpp)
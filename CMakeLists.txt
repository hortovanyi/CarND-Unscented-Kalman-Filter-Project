project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(sources
   src/ukf.cpp
   src/main.cpp)

add_executable(UnscentedKF ${sources})

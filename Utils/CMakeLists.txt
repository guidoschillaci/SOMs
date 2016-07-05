cmake_minimum_required(VERSION 2.8.10)
set(NAME Utils)

project(${NAME})

set(_srcs
  "Common.h"
  "Statistics.h"
  "Statistics.cpp"
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_library(${NAME} SHARED ${_srcs})
target_link_libraries(${NAME})

# variables to change into the libnameConfig.cmake.in file
set(CONF_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}")
set(LIBNAME ${NAME})

configure_file("${CMAKE_BINARY_DIR}/../config.cmake/emptyConfig.cmake.in"
            "${CMAKE_BINARY_DIR}/config.cmake-files/${NAME}Config.cmake" @ONLY)

set_target_properties(${NAME} PROPERTIES LINKER_LANGUAGE CXX)


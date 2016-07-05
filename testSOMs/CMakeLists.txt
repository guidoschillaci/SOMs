cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(NAME testSOMs)
project(${NAME})

# Check modules
set( CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/config.cmake-files/" )

find_package(SOMs)
find_package(ClassicSOM)

SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

if(SOMs_FOUND AND ClassicSOM_FOUND)
    include_directories("${CMAKE_SOURCE_DIR}" ${SOMs_INCLUDE_DIRS} ${ClassicSOM_INCLUDE_DIRS} )
    add_executable(${NAME} main.cpp)
    target_link_libraries(${NAME} ${SOMs_LIBRARIES} ${ClassicSOM_LIBRARIES} )
endif(SOMs_FOUND AND ClassicSOM_FOUND)


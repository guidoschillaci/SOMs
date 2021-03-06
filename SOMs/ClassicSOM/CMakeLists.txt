cmake_minimum_required(VERSION 2.8)
set(NAME ClassicSOM)
project(${NAME})

set(_srcs
  "Neuron.h"
  "Neuron.cpp"
  "ClassicSOM.h"
  "ClassicSOM.cpp"
  "HebbianTable.h"
  "HebbianTable.cpp"
)

# Check modules
set( CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/config.cmake-files/" )

find_package(Boost)
find_package(SOMs)
# this changes the path where to copy the .so file. It matches the sdk/lib path used by qibuild, but you may want to change it if you don't compile with qibuild
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

if(Boost_FOUND AND SOMs_FOUND)

    include_directories("${CMAKE_SOURCE_DIR}" ${Boost_INCLUDE_DIRS} ${SOMs_INCLUDE_DIRS})
    add_library(${NAME} SHARED ${_srcs})
    target_link_libraries(${NAME} ${Boost_LIBRARIES} ${SOMs_LIBRARIES})

    # variables to change into the libnameConfig.cmake.in file
    set(CONF_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/../")
    set(LIBNAME ${NAME})

    configure_file("${CMAKE_BINARY_DIR}/../config.cmake/emptyConfig.cmake.in"
        "${CMAKE_BINARY_DIR}/config.cmake-files/${NAME}Config.cmake" @ONLY)

endif(Boost_FOUND AND SOMs_FOUND)




cmake_minimum_required (VERSION 2.6)
project (InternalModels)

# This variable is set so that libraries are not configured using Aldebaran qibuild (for NAOqi)
set(STAND_ALONE_SOM_LIBRARY 1)

message("Configuring the Internal Models project...")
message("CMAKE_BINARY_DIR " ${CMAKE_BINARY_DIR})

add_subdirectory(Utils "${CMAKE_BINARY_DIR}/Utils")
add_subdirectory(SOMs "${CMAKE_BINARY_DIR}/SOMs")
add_subdirectory(SOMs/ClassicSOM "${CMAKE_BINARY_DIR}/SOMs/ClassicSOM")
add_subdirectory(SOMs/DSOM "${CMAKE_BINARY_DIR}/SOMs/DSOM")

add_subdirectory(testSOMs "${CMAKE_BINARY_DIR}/testSOMs")


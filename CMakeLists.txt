# set(CMAKE_CXX_FLAGS "-O0 -g -pg -Wall -std=c++0x")
cmake_minimum_required(VERSION 3.1)
project(clear)

set(REQ_CATKIN_PKGS
  roscpp
  geometry_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  tf2_eigen
  pcl_ros
  pcl_conversions
)

find_package(PkgConfig REQUIRED)

############################
## Manually find packages ##
############################
# Find Eigen3
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
message(STATUS "Found Eigen3, version: ${EIGEN3_VERSION_STRING}")
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}\n")
else()
message(FATAL_ERROR "Could not find Eigen!")
endif()


find_package(catkin REQUIRED COMPONENTS ${REQ_CATKIN_PKGS})
# find_package(
#   Boost
#   REQUIRED COMPONENTS serialization
# )


include_directories(include 
  SYSTEM 
  ${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR} 
  ${Boost_INCLUDE_DIRS})
link_directories(
  ${Boost_LIBRARY_DIRS} 
  )

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS src
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${REQ_CATKIN_PKGS}
  DEPENDS EIGEN3
)

###########
## Build ##
###########
# c++11 support required
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


add_executable(clear_ros
  src/node.cpp
  src/fuser.cpp
  src/clear/PairwiseMatcher.cpp
  src/clear/MultiwayMatcher.cpp
  src/clear/Hungarian.cpp
  src/clear/MyGraph.cpp
)

set_target_properties(clear_ros PROPERTIES OUTPUT_NAME fuser PREFIX "")

target_link_libraries(clear_ros 
  ${catkin_LIBRARIES} 
  ${Boost_LIBRARIES} 
)

# install(TARGETS clear_ros
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#   )

# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  # FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  # )
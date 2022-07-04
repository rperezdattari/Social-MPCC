cmake_minimum_required(VERSION 2.8.3)
project(lmpcc_tools)

add_compile_options(-std=c++11)
add_compile_options(-fstack-protector)
set(CMAKE_CXX_FLAGS "-O3 -Wall -DCMAKE_BUILD_TYPE=RELEASE")

find_package(catkin REQUIRED COMPONENTS
    eigen_conversions
    geometry_msgs
    nav_msgs
    roscpp
    roslint
    sensor_msgs
    std_msgs
    tf
    tf_conversions
    trajectory_msgs
    visualization_msgs
)
add_definitions(${EIGEN_DEFINITIONS})

catkin_package(
  CATKIN_DEPENDS eigen_conversions geometry_msgs nav_msgs roscpp sensor_msgs std_msgs tf tf_conversions visualization_msgs
  INCLUDE_DIRS include include/${PROJECT_NAME}
  LIBRARIES  lmpcc_tools lmpcc_tools
)

include_directories(include)

### BUILD ###
include_directories(include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    include/${PROJECT_NAME}
    )

add_library(lmpcc_tools
        src/ros_visuals.cpp
        )

add_dependencies(lmpcc_tools ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

### Test Case ####
install(
  DIRECTORY include/lmpcc_tools/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
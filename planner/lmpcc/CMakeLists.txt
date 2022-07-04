cmake_minimum_required(VERSION 2.8.3)
project(lmpcc)

add_compile_options(-std=c++11)
add_compile_options(-fstack-protector)
set(CMAKE_CXX_FLAGS "-O3 -Wall -DCMAKE_BUILD_TYPE=RELEASE")

add_definitions(${PCL_DEFINITIONS})

find_package(catkin REQUIRED COMPONENTS
    actionlib
    actionlib_msgs
    carla_msgs
    cmake_modules
    control_msgs
    dynamic_reconfigure
    eigen_conversions
    geometry_msgs
    kdl_conversions
    kdl_parser
    nav_msgs
    roscpp
    roslint
    sensor_msgs
    std_msgs
    tf
    tf_conversions
    trajectory_msgs
    urdf
    visualization_msgs
    shape_msgs
    lmpcc_msgs
    pcl_ros
    lmpcc_tools
        carla_ackermann_control
        derived_object_msgs
)

find_package(Boost REQUIRED COMPONENTS thread)
find_package(Eigen3 REQUIRED)
#find_package(OpenCV 3)

add_definitions(${EIGEN_DEFINITIONS})

find_package(orocos_kdl REQUIRED)

catkin_python_setup()

generate_dynamic_reconfigure_options(
        cfg/PredictiveController.cfg
)


# Generate messages in the 'msg' folder
add_message_files(
  FILES
  control_feedback.msg
)

## Generate services in the 'srv' folder
add_service_files(
        DIRECTORY
        srv
        FILES
        LMPCCReset.srv
)
# Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    std_msgs
    nav_msgs
    shape_msgs
    visualization_msgs
    lmpcc_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib_msgs dynamic_reconfigure eigen_conversions geometry_msgs kdl_conversions kdl_parser nav_msgs roscpp sensor_msgs std_msgs tf tf_conversions urdf visualization_msgs shape_msgs lmpcc_msgs lmpcc_tools carla_msgs carla_ackermann_control
  DEPENDS Boost
  INCLUDE_DIRS include include/${PROJECT_NAME}
  LIBRARIES  lmpcc_configuration lmpcc_controller
)

include_directories(
  include 
  python_forces_code
  )

### BUILD ###
include_directories(include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${orocos_kdl_INCLUDE_DIRS}
    include/${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/python_forces_code/generated_cpp
    ${PROJECT_SOURCE_DIR}/python_forces_code/PriusFORCESNLPsolver/include
    )


add_library(tkspline_lib
        src/spline.cpp
        )
add_dependencies(tkspline_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tkspline_lib ${CATKIN_LIBRARIES})

add_library(lmpcc_configuration src/lmpcc_configuration.cpp)
add_dependencies(lmpcc_configuration ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(lmpcc_configuration
    ${catkin_LIBRARIES}
    ${orocos_kdl_LIBRARIES}
    ${CASADI_LIBRARIES}
    )

if (EXISTS ${PROJECT_SOURCE_DIR}/python_forces_code/PriusFORCESNLPsolver/PriusFORCESNLPsolver_interface.c)
  add_library(lmpcc_controller
          src/lmpcc_controller.cpp
          src/lmpcc_configuration.cpp
          src/Clothoid.cpp
          src/rk4.cpp
          src/lmpcc_control_node.cpp
          src/reference_path.cpp
          src/scenario/static_obstacles.cpp
          src/scenario/dynamic_obstacle.cpp
          src/scenario/scenario_manager.cpp
          src/scenario/polygon_constructor.cpp
          src/scenario/sampler.cpp
          src/scenario/safety_certifier.cpp
          src/interfaces/carla_interface.cpp
          )
  add_dependencies(lmpcc_controller ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


  add_executable(lmpcc_node
    src/lmpcc_control_node.cpp
    ${PROJECT_SOURCE_DIR}/python_forces_code/PriusFORCESNLPsolver/PriusFORCESNLPsolver_interface.c
    ${PROJECT_SOURCE_DIR}/python_forces_code/PriusFORCESNLPsolver/PriusFORCESNLPsolver_model.c
  )

  target_link_libraries(lmpcc_node
          lmpcc_controller
          lmpcc_configuration
          ${OpenCV_LIBS}
          ${catkin_LIBRARIES}
          ${PCL_LIBRARIES}
          tkspline_lib
          ${PROJECT_SOURCE_DIR}/python_forces_code/PriusFORCESNLPsolver/lib/libPriusFORCESNLPsolver.so
  )

  add_dependencies(lmpcc_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

  install(
    TARGETS lmpcc_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  )
else()
  message(WARNING "LMPCC WILL NOT BE COMPILED (IGNORED) BECAUSE THE SOLVER CANNOT BE FOUND.")
endif()

### Test Case ####
install(
  DIRECTORY include/lmpcc/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)


catkin_install_python(PROGRAMS scripts/predictions_3d.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

cmake_minimum_required(VERSION 3.12)
project(tiny_vo)

set(CMAKE_BUILD_TYPE Debug)
set(ROS_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++17")

find_package(
    catkin REQUIRED COMPONENTS
    sensor_msgs
    std_msgs
    roscpp
    message_generation
)

generate_messages(DEPENDENCIES std_msgs)

find_package(OpenCV 3.4 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

####################### intrinsic calibration program ############################

set(
    CALIBRATION_INC
    ./src/calibration
)

set(
    CALIBRATION_SRC
    ./src/calibration/main.cpp
)

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CALIBRATION_INC})
add_executable(calibration ${CALIBRATION_SRC})
target_link_libraries (calibration ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} yaml-cpp)
add_dependencies(calibration ${PROJECT_NAME}_generate_messages_cpp)

########################## visual odemetry program ###############################

set(
    TINY_VO_INC
    ./core/src
)

set(
    TINY_VO_SRC
    ./src/core/feature_detector.cpp
    ./src/core/visual_odemetry.cpp
    ./src/core/main.cpp
)

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TINY_VO_INC})
add_executable(tiny_vo ${TINY_VO_SRC})
target_link_libraries (tiny_vo ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} Eigen3::Eigen)
add_dependencies(tiny_vo ${PROJECT_NAME}_generate_messages_cpp)

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

set(
    TINY_VO_INC
    ./src
)

set(
    TINY_VO_SRC
    ./src/feature_detector.cpp
    ./src/main.cpp
)

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TINY_VO_INC})
add_executable(tiny_vo ${TINY_VO_SRC})
target_link_libraries (tiny_vo ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
add_dependencies(tiny_vo ${PROJECT_NAME}_generate_messages_cpp)

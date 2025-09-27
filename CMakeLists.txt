cmake_minimum_required(VERSION 3.10)
project(TrajectoryFit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

add_executable(trajectory main.cpp)

target_link_libraries(trajectory
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    glog::glog
)


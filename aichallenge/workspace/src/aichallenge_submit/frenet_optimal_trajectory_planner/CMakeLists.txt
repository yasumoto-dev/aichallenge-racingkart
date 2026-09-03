cmake_minimum_required(VERSION 3.8)
project(frenet_optimal_trajectory_planner)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Eigen3 REQUIRED)

set(EFOT_SRC thirdparty/erdos_fot/src)

include_directories(
  ${EFOT_SRC}
  ${EFOT_SRC}/CubicSpline
  ${EFOT_SRC}/Polynomials
  ${EFOT_SRC}/FrenetOptimalTrajectory
  ${EFOT_SRC}/Obstacle
  ${EFOT_SRC}/Car
)

# Vendored from https://github.com/erdos-project/frenet_optimal_trajectory_planner
# (commit ba0cb5662a0e2ea668b1c2b2951c0f6b84f44f5b, Apache-2.0). Only the
# algorithm core is built here; the Qt-based visualizer (main.cpp/anytime.cpp)
# and the Anytime planner are not needed and are excluded.
add_library(FrenetOptimalTrajectory SHARED
  ${EFOT_SRC}/utils.h
  ${EFOT_SRC}/Polynomials/QuarticPolynomial.cpp
  ${EFOT_SRC}/Polynomials/QuarticPolynomial.h
  ${EFOT_SRC}/Polynomials/QuinticPolynomial.cpp
  ${EFOT_SRC}/Polynomials/QuinticPolynomial.h
  ${EFOT_SRC}/CubicSpline/CubicSpline1D.cpp
  ${EFOT_SRC}/CubicSpline/CubicSpline1D.h
  ${EFOT_SRC}/CubicSpline/CubicSpline2D.cpp
  ${EFOT_SRC}/CubicSpline/CubicSpline2D.h
  ${EFOT_SRC}/FrenetOptimalTrajectory/FrenetOptimalTrajectory.cpp
  ${EFOT_SRC}/FrenetOptimalTrajectory/FrenetOptimalTrajectory.h
  ${EFOT_SRC}/FrenetOptimalTrajectory/FrenetPath.cpp
  ${EFOT_SRC}/FrenetOptimalTrajectory/FrenetPath.h
  ${EFOT_SRC}/FrenetOptimalTrajectory/fot_wrapper.cpp
  ${EFOT_SRC}/FrenetOptimalTrajectory/py_cpp_struct.h
  ${EFOT_SRC}/Obstacle/Obstacle.cpp
  ${EFOT_SRC}/Obstacle/Obstacle.h
  ${EFOT_SRC}/Car/Car.cpp
  ${EFOT_SRC}/Car/Car.h
)
target_compile_options(FrenetOptimalTrajectory PRIVATE -O3 -pthread)
target_compile_features(FrenetOptimalTrajectory PRIVATE cxx_std_14)
target_link_libraries(FrenetOptimalTrajectory Eigen3::Eigen pthread)

install(TARGETS FrenetOptimalTrajectory
  LIBRARY DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  frenet_optimal_trajectory_planner/frenet_optimal_trajectory_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_run_fot test/test_run_fot.py)
endif()

ament_package()

cmake_minimum_required(VERSION 3.8)
project(frenet_optimal_trajectory_planner)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

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
  ament_add_pytest_test(test_frenet_frame test/test_frenet_frame.py)
  ament_add_pytest_test(test_trajectory_candidates test/test_trajectory_candidates.py)
endif()

ament_package()

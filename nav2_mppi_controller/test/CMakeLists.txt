set(TEST_NAMES
  optimizer_smoke_test
  controller_state_transition_test
  models_test
  noise_generator_test
  parameter_handler_test
  motion_model_tests
  trajectory_visualizer_tests
)

# These construct Costmap2DROS with the local_namespace argument that Kilted removed.
if(nav2_costmap_2d_VERSION VERSION_LESS 1.4.0)
  list(APPEND TEST_NAMES utils_test path_handler_test critic_manager_test optimizer_unit_tests)
endif()

foreach(name IN LISTS TEST_NAMES)
  ament_add_gtest(${name}
    ${name}.cpp
  )

  ament_target_dependencies(${name}
    ${dependencies_pkgs}
  )

  target_link_libraries(${name}
    mppi_controller
  )

  if(${TEST_DEBUG_INFO})
    target_compile_definitions(${name} PUBLIC -DTEST_DEBUG_INFO)
  endif()

endforeach()

# This is a special case requiring linking against the critics library, and constructs
# Costmap2DROS like the tests above.
if(nav2_costmap_2d_VERSION VERSION_LESS 1.4.0)
  ament_add_gtest(critics_tests critics_tests.cpp)
  ament_target_dependencies(critics_tests ${dependencies_pkgs})
  target_link_libraries(critics_tests mppi_controller mppi_critics)
  if(${TEST_DEBUG_INFO})
    target_compile_definitions(critics_tests PUBLIC -DTEST_DEBUG_INFO)
  endif()
endif()

cmake_minimum_required(VERSION 3.16)
project(luna_cpp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(VORONOI_SANITIZERS "Enable host address and undefined-behavior sanitizers" OFF)
if(VORONOI_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Here and not only in voronoi-cpp: imported targets are visible in the
# directory that finds them and below it.
find_package(Kokkos CONFIG REQUIRED)
find_package(HDF5 COMPONENTS C)
include(CTest)

# voronoi-cpp: its current main by default, fetched at configure time; -DLUNA_VORONOI_COMMIT=<sha> for exactly one commit,
# -DLUNA_VORONOI_SOURCE=<dir> for a local checkout (cmake/voronoi.cmake, README "Install and build").
include(cmake/voronoi.cmake)

# The commits built with (luna-cpp and voronoi-cpp, dirty flags), regenerated at every build: luna::build_info().
set(luna_build_info_args -DLUNA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DVORONOI_DIR=${luna_voronoi_dir}
    -DVORONOI_ORIGIN=${luna_voronoi_origin} -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} ${luna_build_info_args} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.cmake)
file(READ ${CMAKE_CURRENT_BINARY_DIR}/luna_build_info.txt luna_build_line)
string(STRIP "${luna_build_line}" luna_build_line)
message(STATUS "${luna_build_line}")
add_custom_target(luna_build_info_files ALL
                  COMMAND ${CMAKE_COMMAND} ${luna_build_info_args} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.cmake
                  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/luna_build_info.cpp ${CMAKE_CURRENT_BINARY_DIR}/luna_build_info.txt)
add_library(luna_build_info STATIC ${CMAKE_CURRENT_BINARY_DIR}/luna_build_info.cpp)
add_dependencies(luna_build_info luna_build_info_files)
target_include_directories(luna_build_info PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro)

# voronoi_library. The library's tests and programs stay out of this build.
set(luna_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF)
add_subdirectory(${luna_voronoi_dir} ${CMAKE_CURRENT_BINARY_DIR}/voronoi-cpp EXCLUDE_FROM_ALL)
set(BUILD_TESTING ${luna_build_testing})

# voronoi-cpp is the mesh library only: this repository owns the solver. The targets keep their luna_ prefix
# from the split (the programs keep their names); only voronoi_library is linked from voronoi-cpp.
add_library(luna_hydro_library INTERFACE)
target_link_libraries(luna_hydro_library INTERFACE voronoi_library luna_build_info)
target_include_directories(luna_hydro_library INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro)

function(hydro_program target output source)
    add_executable(${target} ${source})
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${output})
    target_link_libraries(${target} PRIVATE luna_hydro_library)
    # check.hpp and support.hpp are the library's.
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test
        ${luna_voronoi_dir}/src/test)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

hydro_program(luna_hydro_1d hydro_1d src/bench/hydro_1d.cpp)
hydro_program(luna_hydro_wave3d hydro_wave3d src/bench/hydro_wave3d.cpp)
hydro_program(luna_hydro_kh2d hydro_kh2d src/bench/hydro_kh2d.cpp)
# device_memory.hpp is the library's.
target_include_directories(luna_hydro_kh2d PRIVATE ${luna_voronoi_dir}/src/bench)
# Snapshots need HDF5.
if(HDF5_FOUND)
    hydro_program(luna_hydro_sedov hydro_sedov src/bench/hydro_sedov.cpp)
    target_link_libraries(luna_hydro_sedov PRIVATE HDF5::HDF5)
else()
    message(STATUS "HDF5 not found: hydro_sedov is not built")
endif()

if(BUILD_TESTING)
    set(hydro_test_timeout 600)
    if(VORONOI_SANITIZERS)
        set(hydro_test_timeout 3000)  # about 11x slower: `hydro` and `hydro_wave` passed 600 s on CI's 2 cores with the moving step on both meshes
    endif()
    foreach(test hydro hydro1d hydro1d_cases hydro_wave hydro2d)
        hydro_program(${test}_test ${test}_test src/test/${test}_test.cpp)
        add_test(NAME ${test} COMMAND ${test}_test)
        set_tests_properties(${test} PROPERTIES TIMEOUT ${hydro_test_timeout})
    endforeach()
    # luna-cpp #30 (points pile into a squeezed low-density slab): the reproducer's gate, off until #30 is fixed (ctest -R issue30 shows it).
    add_test(NAME hydro1d_issue30 COMMAND hydro1d_cases_test issue30)
    set_tests_properties(hydro1d_issue30 PROPERTIES TIMEOUT ${hydro_test_timeout} DISABLED TRUE)
endif()

# The hand-written tangents and adjoints (hydro_derivatives_test): opt-in while they are new; off, the build is as before.
option(LUNA_DERIVATIVE_TESTS "Build and run hydro_derivatives_test" OFF)
if(BUILD_TESTING AND LUNA_DERIVATIVE_TESTS)
    hydro_program(hydro_derivatives_test hydro_derivatives_test src/test/hydro_derivatives_test.cpp)
    add_test(NAME hydro_derivatives COMMAND hydro_derivatives_test)
    set_tests_properties(hydro_derivatives PROPERTIES TIMEOUT ${hydro_test_timeout})
endif()

# The derivative examples (research_notes/autodiff_examples_2026-11-23.md): the drivers scripts/examples calls.
option(LUNA_DERIVATIVE_EXAMPLES "Build the derivative examples" OFF)
if(LUNA_DERIVATIVE_EXAMPLES)
    hydro_program(luna_sod_theta sod_theta src/examples/sod_theta.cpp)
    hydro_program(luna_rollout rollout src/examples/rollout.cpp)
endif()

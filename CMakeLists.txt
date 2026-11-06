cmake_minimum_required(VERSION 3.16)
project(luna_cpp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(VORONOI_SANITIZERS "Enable host address and undefined-behavior sanitizers" OFF)
if(VORONOI_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Here and not only in the submodule: imported targets are visible in the
# directory that finds them and below it.
find_package(Kokkos CONFIG REQUIRED)
find_package(HDF5 COMPONENTS C)
include(CTest)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extern/voronoi-cpp/CMakeLists.txt)
    message(FATAL_ERROR "extern/voronoi-cpp is empty: git submodule update --init --recursive")
endif()

# The pin must be voronoi-cpp's newest main, at configure and at every build.
option(LUNA_ALLOW_OLD_PIN "Warn only when extern/voronoi-cpp is behind voronoi-cpp main" OFF)
set(LUNA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include(cmake/check_pin.cmake)
add_custom_target(luna_check_pin ALL COMMAND ${CMAKE_COMMAND} -DLUNA_ALLOW_OLD_PIN=${LUNA_ALLOW_OLD_PIN}
                  -DLUNA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_pin.cmake)

# voronoi_library. The submodule's tests and programs stay out of this build.
set(luna_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF)
add_subdirectory(extern/voronoi-cpp EXCLUDE_FROM_ALL)
set(BUILD_TESTING ${luna_build_testing})

# The pinned voronoi-cpp still defines hydro_library, hydro_1d and hydro_sedov,
# so the targets here carry a luna_ prefix; the programs keep their names. Never
# link the submodule's targets: they build its copy of the hydro. Until the hydro
# leaves voronoi-cpp, `--target hydro_1d` means that copy, `--target luna_hydro_1d`
# this one (research_notes/split_2026-11-05.md).
add_library(luna_hydro_library INTERFACE)
target_link_libraries(luna_hydro_library INTERFACE voronoi_library)
target_include_directories(luna_hydro_library INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro)

function(hydro_program target output source)
    add_executable(${target} ${source})
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${output})
    target_link_libraries(${target} PRIVATE luna_hydro_library)
    # check.hpp and support.hpp are the library's.
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/voronoi-cpp/src/test)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

hydro_program(luna_hydro_1d hydro_1d src/bench/hydro_1d.cpp)
hydro_program(luna_hydro_wave3d hydro_wave3d src/bench/hydro_wave3d.cpp)
hydro_program(luna_hydro_kh2d hydro_kh2d src/bench/hydro_kh2d.cpp)
# device_memory.hpp is the library's.
target_include_directories(luna_hydro_kh2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/voronoi-cpp/src/bench)
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
        set(hydro_test_timeout 3000)  # about 11x slower: `hydro` and `hydro_wave` passed 600 s on CI's 2 cores with two passes
    endif()
    foreach(test hydro hydro1d hydro_wave hydro2d)
        hydro_program(${test}_test ${test}_test src/test/${test}_test.cpp)
        add_test(NAME ${test} COMMAND ${test}_test)
        set_tests_properties(${test} PROPERTIES TIMEOUT ${hydro_test_timeout})
    endforeach()
endif()

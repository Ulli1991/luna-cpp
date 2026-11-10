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

# Warns, loudly, when the pinned voronoi-cpp is behind its main, at configure, at every build and when a program starts.
option(LUNA_STRICT_PIN "Stop the build when extern/voronoi-cpp is behind voronoi-cpp main" OFF)
set(LUNA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include(cmake/check_pin.cmake)
add_compile_definitions(LUNA_VORONOI_PIN="${pin}" LUNA_VORONOI_LATEST="${latest}" LUNA_VORONOI_BEHIND=${luna_pin_behind})
add_custom_target(luna_check_pin ALL COMMAND ${CMAKE_COMMAND} -DLUNA_STRICT_PIN=${LUNA_STRICT_PIN}
                  -DLUNA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_pin.cmake)

# voronoi_library. The submodule's tests and programs stay out of this build.
set(luna_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF)
add_subdirectory(extern/voronoi-cpp EXCLUDE_FROM_ALL)
set(BUILD_TESTING ${luna_build_testing})

# voronoi-cpp is the mesh library only: this repository owns the solver. The targets keep their luna_ prefix
# from the split (the programs keep their names); only voronoi_library is linked from the submodule.
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
        set(hydro_test_timeout 3000)  # about 11x slower: `hydro` and `hydro_wave` passed 600 s on CI's 2 cores with the moving step on both meshes
    endif()
    foreach(test hydro hydro1d hydro1d_cases hydro_wave hydro2d)
        hydro_program(${test}_test ${test}_test src/test/${test}_test.cpp)
        add_test(NAME ${test} COMMAND ${test}_test)
        set_tests_properties(${test} PROPERTIES TIMEOUT ${hydro_test_timeout})
    endforeach()
endif()

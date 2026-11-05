# Stops a build on an out-of-date extern/voronoi-cpp: the pin must be the head
# of voronoi-cpp main. -DLUNA_ALLOW_OLD_PIN=ON turns that into a warning, for
# reproducing an old result. With no network (Raven's compute nodes) it warns.
# Run with cmake -P, or include()d from the top CMakeLists.txt.
if(NOT DEFINED LUNA_SOURCE_DIR)
    set(LUNA_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()
execute_process(COMMAND git -C ${LUNA_SOURCE_DIR}/extern/voronoi-cpp rev-parse HEAD
                OUTPUT_VARIABLE pin OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE pin_status)
execute_process(COMMAND git ls-remote https://github.com/Ulli1991/voronoi-cpp.git refs/heads/main
                OUTPUT_VARIABLE remote ERROR_QUIET RESULT_VARIABLE remote_status TIMEOUT 15)
string(REGEX MATCH "^[0-9a-f]+" latest "${remote}")
if(NOT pin_status EQUAL 0 OR NOT remote_status EQUAL 0 OR latest STREQUAL "")
    message(WARNING "voronoi-cpp pin not checked: no git checkout or no network")
elseif(NOT pin STREQUAL latest)
    set(text "extern/voronoi-cpp is at ${pin}, voronoi-cpp main at ${latest}: run `git submodule update --remote extern/voronoi-cpp` and commit it")
    if(LUNA_ALLOW_OLD_PIN)
        message(WARNING "${text}")
    else()
        message(FATAL_ERROR "${text}")
    endif()
endif()

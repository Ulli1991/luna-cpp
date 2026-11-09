# Warns when extern/voronoi-cpp is behind voronoi-cpp main; the pin moves when
# luna-cpp needs a newer library. -DLUNA_STRICT_PIN=ON stops the build instead.
# With no network (Raven's compute nodes) it warns that it could not check.
# Run with cmake -P, or include()d from the top CMakeLists.txt.
if(NOT DEFINED LUNA_SOURCE_DIR)
    set(LUNA_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()
execute_process(COMMAND git -C ${LUNA_SOURCE_DIR}/extern/voronoi-cpp rev-parse HEAD
                OUTPUT_VARIABLE pin OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE pin_status)
execute_process(COMMAND git ls-remote https://github.com/Ulli1991/voronoi-cpp.git refs/heads/main
                OUTPUT_VARIABLE remote ERROR_QUIET RESULT_VARIABLE remote_status TIMEOUT 15)
string(REGEX MATCH "^[0-9a-f]+" latest "${remote}")
set(luna_pin_behind 0)
if(NOT pin_status EQUAL 0 OR NOT remote_status EQUAL 0 OR latest STREQUAL "")
    message(WARNING "voronoi-cpp pin not checked: no git checkout or no network")
elseif(NOT pin STREQUAL latest)
    set(text "this luna-cpp uses voronoi-cpp ${pin}, but a newer voronoi-cpp exists (${latest}). To build with it: git submodule update --remote extern/voronoi-cpp")
    set(luna_pin_behind 1)
    set(banner "\n==================================================================\n  OLD voronoi-cpp: ${text}\n==================================================================\n")
    if(LUNA_STRICT_PIN)
        message(FATAL_ERROR "${banner}")
    else()
        message(WARNING "${banner}")
    endif()
endif()

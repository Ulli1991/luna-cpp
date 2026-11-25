# Where voronoi-cpp comes from (README, "Install and build"); sets luna_voronoi_dir and luna_voronoi_origin.
#   default                       voronoi-cpp main from GitHub, fetched at configure time (needs the network)
#   -DLUNA_VORONOI_COMMIT=<sha>   exactly that commit, fetched (papers, bit gates)
#   -DLUNA_VORONOI_SOURCE=<dir>   a local checkout, used as it is (Raven's compute nodes have no network);
#                                 with LUNA_VORONOI_COMMIT as well, configure stops unless the checkout is at that commit
# A reconfigure without network keeps the commit fetched before, with a warning.
set(LUNA_VORONOI_REPOSITORY "https://github.com/Ulli1991/voronoi-cpp.git" CACHE STRING "voronoi-cpp repository to fetch")
set(LUNA_VORONOI_COMMIT "" CACHE STRING "Build exactly this voronoi-cpp commit (empty: the current main)")
set(LUNA_VORONOI_SOURCE "" CACHE PATH "A local voronoi-cpp checkout to build against instead of fetching")

if(LUNA_VORONOI_SOURCE)
    get_filename_component(luna_voronoi_dir "${LUNA_VORONOI_SOURCE}" ABSOLUTE)
    if(NOT EXISTS "${luna_voronoi_dir}/CMakeLists.txt")
        message(FATAL_ERROR "LUNA_VORONOI_SOURCE=${luna_voronoi_dir} is not a voronoi-cpp checkout")
    endif()
    set(luna_voronoi_origin "local")
    if(LUNA_VORONOI_COMMIT)
        execute_process(COMMAND git -C "${luna_voronoi_dir}" rev-parse HEAD
                        OUTPUT_VARIABLE head OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        execute_process(COMMAND git -C "${luna_voronoi_dir}" rev-parse "${LUNA_VORONOI_COMMIT}^{commit}"
                        OUTPUT_VARIABLE want OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(head STREQUAL "" OR NOT head STREQUAL want)
            message(FATAL_ERROR "LUNA_VORONOI_SOURCE is at '${head}', not at LUNA_VORONOI_COMMIT=${LUNA_VORONOI_COMMIT}")
        endif()
        set(luna_voronoi_origin "commit")
    endif()
else()
    if(LUNA_VORONOI_COMMIT)
        set(sha "${LUNA_VORONOI_COMMIT}")
        set(luna_voronoi_origin "commit")
    else()
        execute_process(COMMAND git ls-remote "${LUNA_VORONOI_REPOSITORY}" refs/heads/main
                        OUTPUT_VARIABLE remote ERROR_QUIET RESULT_VARIABLE status TIMEOUT 30)
        string(REGEX MATCH "^[0-9a-f]+" sha "${remote}")
        set(luna_voronoi_origin "main")
        if(NOT status EQUAL 0 OR sha STREQUAL "")
            if(LUNA_VORONOI_FETCHED)
                message(WARNING "voronoi-cpp main not reachable: keeping ${LUNA_VORONOI_FETCHED}, fetched before")
                set(sha "${LUNA_VORONOI_FETCHED}")
            else()
                message(FATAL_ERROR "voronoi-cpp main not reachable (${LUNA_VORONOI_REPOSITORY}). Configure where there is "
                                    "network (on Raven: the login node), or give -DLUNA_VORONOI_SOURCE=<local checkout>")
            endif()
        endif()
    endif()
    include(FetchContent)
    # SOURCE_SUBDIR without a CMakeLists.txt: populate only; the add_subdirectory below keeps the library's tests out.
    FetchContent_Declare(voronoi_cpp GIT_REPOSITORY "${LUNA_VORONOI_REPOSITORY}" GIT_TAG "${sha}" SOURCE_SUBDIR luna-populate-only)
    FetchContent_MakeAvailable(voronoi_cpp)
    set(luna_voronoi_dir "${voronoi_cpp_SOURCE_DIR}")
    set(LUNA_VORONOI_FETCHED "${sha}" CACHE INTERNAL "voronoi-cpp commit fetched at the last configure")
endif()

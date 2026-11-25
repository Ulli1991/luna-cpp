// The luna-cpp and voronoi-cpp commits a program was built with; build_info() is in the generated
// luna_build_info.cpp (cmake/build_info.cmake), so a new commit relinks the programs without recompiling them.
#pragma once

#include <cstdio>
#include <string>

namespace luna {

// Commits (full sha, or "unknown" outside git) and whether the trees had uncommitted changes. voronoi_origin: main
// (fetched voronoi-cpp main), commit (-DLUNA_VORONOI_COMMIT) or local (-DLUNA_VORONOI_SOURCE). Used by: build_line.
struct BuildInfo {
    const char* luna_commit;
    bool luna_dirty;
    const char* voronoi_commit;
    bool voronoi_dirty;
    const char* voronoi_origin;
};

BuildInfo build_info();

// "build: luna-cpp=<sha> luna_dirty=0 voronoi-cpp=<sha> voronoi_dirty=0 voronoi_origin=main", the line of
// luna_build_info.txt. Used by: print_build_info, the snapshot and output-file metadata.
inline std::string build_line() {
    const BuildInfo b = build_info();
    return std::string("build: luna-cpp=") + b.luna_commit + " luna_dirty=" + (b.luna_dirty ? "1" : "0") +
           " voronoi-cpp=" + b.voronoi_commit + " voronoi_dirty=" + (b.voronoi_dirty ? "1" : "0") +
           " voronoi_origin=" + b.voronoi_origin;
}

// Every program and test prints the build line first on stdout. Used by: the mains of src/bench and src/test.
inline void print_build_info() {
    std::printf("%s\n", build_line().c_str());
    std::fflush(stdout);
}

// Writes the build line as a JSON side file <path>.build.json next to an output that has no room for metadata (.npy,
// .bin). Used by: hydro_1d, hydro_wave3d and hydro_kh2d (--out).
inline void write_build_json(const std::string& path) {
    const BuildInfo b = build_info();
    if (FILE* f = std::fopen((path + ".build.json").c_str(), "w")) {
        std::fprintf(f, "{\"luna_cpp\": \"%s\", \"luna_dirty\": %d, \"voronoi_cpp\": \"%s\", \"voronoi_dirty\": %d, \"voronoi_origin\": \"%s\"}\n",
                     b.luna_commit, b.luna_dirty ? 1 : 0, b.voronoi_commit, b.voronoi_dirty ? 1 : 0, b.voronoi_origin);
        std::fclose(f);
    }
}

}  // namespace luna

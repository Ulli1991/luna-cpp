// Tells a user at program start that this build uses an older voronoi-cpp than
// its main.
#pragma once

#include <cstdio>

namespace luna {

// Prints a banner to stderr when the configure step found the pinned
// voronoi-cpp behind its main.
inline void pin_notice() {
#if defined(LUNA_VORONOI_BEHIND) && LUNA_VORONOI_BEHIND
  std::fprintf(
      stderr,
      "==================================================================\n"
      "  OLD voronoi-cpp: this luna-cpp uses voronoi-cpp %.12s, a newer one\n"
      "  exists (%.12s). git submodule update --remote extern/voronoi-cpp\n"
      "==================================================================\n",
      LUNA_VORONOI_PIN, LUNA_VORONOI_LATEST);
#endif
}

} // namespace luna

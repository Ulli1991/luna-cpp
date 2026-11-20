#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Reads a NumPy .npy file of little-endian float64 in C order (versions 1 to 3): its values and shape.
// Used by: grid_test (LUNA's references in src/test/data/luna_grid, written by scripts/luna_grid_bits.py).
struct Npy {
    std::vector<double> values;
    std::vector<long> shape;
};

inline Npy read_npy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("read_npy: cannot open " + path);
    char magic[8];
    in.read(magic, 8);
    if (std::string(magic + 1, 5) != "NUMPY") throw std::runtime_error("read_npy: not an .npy file: " + path);
    std::uint32_t length = 0;
    if (magic[6] == 1) {
        unsigned char b[2];
        in.read(reinterpret_cast<char*>(b), 2);
        length = b[0] | b[1] << 8;
    } else {
        unsigned char b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        length = b[0] | b[1] << 8 | b[2] << 16 | static_cast<std::uint32_t>(b[3]) << 24;
    }
    std::string header(length, ' ');
    in.read(header.data(), length);
    if (header.find("'descr': '<f8'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("read_npy: only C-order float64: " + path);
    Npy out;
    const std::size_t open = header.find('(', header.find("'shape'")), close = header.find(')', open);
    std::size_t at = open + 1;
    long count = 1;
    while (at < close) {
        const std::size_t comma = header.find(',', at);
        const std::string item = header.substr(at, std::min(comma, close) - at);
        if (item.find_first_of("0123456789") != std::string::npos) {
            out.shape.push_back(std::stol(item));
            count *= out.shape.back();
        }
        if (comma == std::string::npos || comma > close) break;
        at = comma + 1;
    }
    out.values.resize(count);
    in.read(reinterpret_cast<char*>(out.values.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if (!in) throw std::runtime_error("read_npy: short file: " + path);
    return out;
}

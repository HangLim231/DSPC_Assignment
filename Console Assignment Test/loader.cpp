// File: loader.cpp
#include "loader.h"
#include <fstream>
#include <iostream>

std::vector<Image> load_dataset(const std::string& binary_file_path) {
    std::ifstream file(binary_file_path, std::ios::binary);
    std::vector<Image> dataset;

    if (!file) {
        std::cerr << "Failed to open CIFAR-10 binary file." << std::endl;
        return dataset;
    }

    const int record_size = 1 + 3072;
    unsigned char buffer[record_size];

    while (file.read(reinterpret_cast<char*>(buffer), record_size)) {
        Image img;
        img.label = buffer[0];
        img.pixels.resize(IMAGE_PIXELS);

        for (int i = 0; i < IMAGE_PIXELS; ++i) {
            float r = static_cast<float>(buffer[1 + i]) / 255.0f;
            float g = static_cast<float>(buffer[1 + i + 1024]) / 255.0f;
            float b = static_cast<float>(buffer[1 + i + 2048]) / 255.0f;
            img.pixels[i] = (r + g + b) / 3.0f;
        }

        dataset.push_back(img);
        if (dataset.size() >= 1000) break;
    }

    return dataset;
}

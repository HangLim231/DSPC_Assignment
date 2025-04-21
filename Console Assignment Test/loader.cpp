// File: loader.cpp
#include "loader.h"
#include <fstream>
#include <iostream>

using namespace std;

//Function to load CIFAR-10 dataset
std::vector<Image> load_dataset(const std::vector<std::string>& batch_files) {
    std::vector<Image> dataset;

    for (const auto& file_path : batch_files) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << file_path << std::endl;
            continue;
        }

        // Read the first 4 bytes to skip the header
        const int record_size = 1 + 3072;
        unsigned char buffer[record_size];

        // Read the file in chunks of record_size
        while (file.read(reinterpret_cast<char*>(buffer), record_size)) {
            Image img;
            img.label = buffer[0];
            img.pixels.resize(IMAGE_PIXELS);

            // Convert RGB to grayscale
            for (int i = 0; i < IMAGE_PIXELS; ++i) {
                float r = static_cast<float>(buffer[1 + i]) / 255.0f;
                float g = static_cast<float>(buffer[1 + i + 1024]) / 255.0f;
                float b = static_cast<float>(buffer[1 + i + 2048]) / 255.0f;
                img.pixels[i] = (r + g + b) / 3.0f;
            }

            dataset.push_back(img);
        }
    }

    return dataset;
}
    

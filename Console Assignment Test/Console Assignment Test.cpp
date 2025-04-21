// File: main.cpp
#include "train_serial.h"
#include "train_openmp.h"
#include "train_cuda.h"
#include "visualization.h"
#include "loader.h"
#include <iostream>
#include <fstream>
#include "evaluate.h"

int main() {
    std::vector<std::string> train_files = {
        "data/data_batch_1.bin",
        "data/data_batch_2.bin",
        "data/data_batch_3.bin",
        "data/data_batch_4.bin",
        "data/data_batch_5.bin"
    };

    // Debug: Check file existence
    for (const auto& file : train_files) {
        std::ifstream f(file.c_str());
        if (!f.good()) {
            std::cerr << "Warning: Cannot open file: " << file << "\n";
        }
        else {
            std::cout << "Successfully opened file: " << file << "\n";
        }
    }

    std::cout << "Loading CIFAR-10 training dataset...\n";
    std::vector<Image> train_data = load_dataset(train_files);

    // Verify dataset loading
    std::cout << "=== Dataset Statistics ===\n";
    std::cout << "Total images loaded: " << train_data.size() << "\n";

    if (!train_data.empty()) {
        // Check label distribution
        std::vector<int> label_counts(NUM_CLASSES, 0);
        for (const auto& img : train_data) {
            if (img.label >= 0 && img.label < NUM_CLASSES) {
                label_counts[img.label]++;
            }
        }

        std::cout << "Label distribution:\n";
        for (int i = 0; i < NUM_CLASSES; ++i) {
            std::cout << "Class " << i << ": " << label_counts[i] << " images\n";
        }
    }
    else {
        std::cerr << "ERROR: No images were loaded. Please check the dataset path.\n";
        return 1;
    }

    // Display a sample image before training
    if (!train_data.empty()) {
        std::cout << "Sample image from training set:\n";
        Visualizer::displayImage(train_data[0]);

        std::cout << "Press Enter to start CNN training using CUDA...";
        std::cin.get();
    }

    // Train the model using CUDA
    train_cuda(train_data);

    return 0;
}
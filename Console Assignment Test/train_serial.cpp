// File: train_serial.cpp
#include "train_serial.h"
#include <iostream>
#include <chrono>

void train_serial(const std::vector<Image>& dataset) {
    std::cout << "Training using CPU Serial Mode...\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < 5; ++epoch) {
        for (const auto& img : dataset) {
            // placeholder logic
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Serial CPU training time: " << elapsed.count() << " seconds\n";
}

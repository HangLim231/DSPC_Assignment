// File: train_openmp.cpp
#include "train_openmp.h"
#include <iostream>
#include <omp.h>

void train_openmp(const std::vector<Image>& dataset) {
    std::cout << "Training using CPU Parallel Mode with OpenMP...\n";
    omp_set_num_threads(4);
    double start = omp_get_wtime();

    for (int epoch = 0; epoch < 5; ++epoch) {
#pragma omp parallel for
        for (size_t i = 0; i < dataset.size(); ++i) {
            const auto& img = dataset[i];
            // placeholder logic
        }
    }

    double end = omp_get_wtime();
    std::cout << "OpenMP CPU training time: " << (end - start) << " seconds\n";
}

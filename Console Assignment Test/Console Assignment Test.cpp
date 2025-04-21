// File: main.cpp
#include "train_serial.h"
#include "train_openmp.h"
#include "train_cuda.h"
#include "loader.h"
#include <iostream>

int main() {
    std::vector<std::string> train_files = {
    "data/data_batch_1.bin",
    "data/data_batch_2.bin",
    "data/data_batch_3.bin",
    "data/data_batch_4.bin",
    "data/data_batch_5.bin"
    };
    std::vector<Image> train_data = load_dataset(train_files);

    // Check if the dataset is loaded correctly
    std::cout << "Loaded " << train_data.size() << " images.\n";

    train_cuda(train_data);  // ? Call it like a normal function "CUDA not enabled. Rebuild with nvcc to run train_cuda.\n";

    return 0;
}

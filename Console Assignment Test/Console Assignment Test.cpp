// File: main.cpp
#include "train_serial.h"
#include "train_openmp.h"
#include "train_cuda.h"
#include "loader.h"

int main() {
    std::string cifar_path = "./cifar-10-batches-bin/data_batch_1.bin";
    std::vector<Image> dataset = load_dataset(cifar_path);

    train_serial(dataset);
    train_openmp(dataset);
#ifdef __CUDACC__
    train_cuda(dataset);
#endif

    return 0;
}

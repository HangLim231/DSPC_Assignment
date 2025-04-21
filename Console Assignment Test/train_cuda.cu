// File: train_cuda.cu
#ifdef __CUDACC__
#include "train_cuda.h"
#include <iostream>
#include <cuda_runtime.h>

__global__ void conv_layer_gpu(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < IMAGE_PIXELS) {
        output[idx] = input[idx] * 0.5f; // dummy operation
    }
}

void train_cuda(const std::vector<Image>& dataset) {
    std::cout << "Training using GPU with CUDA...\n";
    float* d_input, * d_output;
    cudaMalloc(&d_input, IMAGE_PIXELS * sizeof(float));
    cudaMalloc(&d_output, IMAGE_PIXELS * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int epoch = 0; epoch < 5; ++epoch) {
        for (const auto& img : dataset) {
            cudaMemcpy(d_input, img.pixels.data(), IMAGE_PIXELS * sizeof(float), cudaMemcpyHostToDevice);
            conv_layer_gpu << <(IMAGE_PIXELS + 255) / 256, 256 >> > (d_input, d_output);
            cudaDeviceSynchronize();
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA GPU training time: " << milliseconds / 1000.0f << " seconds\n";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
#endif

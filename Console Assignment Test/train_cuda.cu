#ifdef __CUDACC__
#include "train_cuda.h"
#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

// Debug helper functions
void print_image_sample(const Image& img) {
    std::cout << "Image Label: " << img.label << "\n";
    std::cout << "First 64 pixels (8x8 corner):\n";
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            std::cout << std::fixed << std::setprecision(2)
                << img.pixels[i * IMAGE_SIZE + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Error checking function
void check_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - "
            << cudaGetErrorString(error) << "\n";
        exit(-1);
    }
}

// Kernel function for convolution layer (dummy operation)
__global__ void conv_layer_gpu(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Store original value for debugging
        float original = input[idx];
        output[idx] = input[idx] * 0.5f; // dummy operation

        // Add debug printf (only works with compute capability 2.0+)
        if (idx < 5) {
            printf("Thread %d: Input=%.2f, Output=%.2f\n",
                idx, original, output[idx]);
        }
    }
}

// Function to train the model using CUDA
void train_cuda(const std::vector<Image>& dataset) {
    std::cout << "\n=== Starting CUDA Training ===\n";

    // Debug: Print dataset info
    std::cout << "Dataset size: " << dataset.size() << " images\n";
    if (!dataset.empty()) {
        std::cout << "\nFirst image sample:\n";
        print_image_sample(dataset[0]);
    }

    // Allocate GPU memory
    float* d_input, * d_output;
    cudaError_t error;  // Check for CUDA errors

    // Allocate memory for input and output on the GPU
    error = cudaMalloc(&d_input, IMAGE_PIXELS * sizeof(float));
    check_cuda_error(error, "Input allocation failed");

    // Allocate memory for output on the GPU
    error = cudaMalloc(&d_output, IMAGE_PIXELS * sizeof(float));
    check_cuda_error(error, "Output allocation failed");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Training loop with progress tracking
    int total_epochs = 5;
    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << total_epochs << "\n";

        int batch_size = 100; // Process images in batches
        int num_batches = dataset.size() / batch_size;

        for (int batch = 0; batch < num_batches; ++batch) {
            // Process one image as example for debugging
            const auto& img = dataset[batch * batch_size];

            error = cudaMemcpy(d_input, img.pixels.data(),
                IMAGE_PIXELS * sizeof(float),
                cudaMemcpyHostToDevice);
            check_cuda_error(error, "Input transfer failed");

            // Launch kernel
            int threadsPerBlock = 256;
            int blocks = (IMAGE_PIXELS + threadsPerBlock - 1) / threadsPerBlock;
            conv_layer_gpu << <blocks, threadsPerBlock >> > (d_input, d_output, IMAGE_PIXELS);

            // Check for kernel launch errors
            error = cudaGetLastError();
            check_cuda_error(error, "Kernel launch failed");

            // Kernel synchronization
            error = cudaDeviceSynchronize();
            check_cuda_error(error, "Kernel synchronization failed");

            // Debug: Verify output (for first batch of each epoch)
            if (batch == 0) {
                std::vector<float> output(IMAGE_PIXELS);
                error = cudaMemcpy(output.data(), d_output,
                    IMAGE_PIXELS * sizeof(float),
                    cudaMemcpyDeviceToHost);
                check_cuda_error(error, "Output transfer failed");

                std::cout << "Sample output values (first 5):\n";
                for (int i = 0; i < 5; ++i) {
                    std::cout << output[i] << " ";
                }
                std::cout << "\n";
            }

            // Show progress
            if (batch % 10 == 0) {
                std::cout << "\rProgress: " << batch * batch_size
                    << "/" << dataset.size() << " images" << std::flush;
            }
        }
        std::cout << "\n";
    }

    // Timing results
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\nCUDA GPU training completed in "
        << milliseconds / 1000.0f << " seconds\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
#endif
// File: train_cuda.cu
#include "train_cuda.h"
#include "evaluate.h"
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>

// Constants for CNN architecture
#define CONV_KERNEL_SIZE 5
#define CONV_OUT_CHANNELS 16
#define CONV_OUT_SIZE (IMAGE_SIZE - CONV_KERNEL_SIZE + 1)
#define POOL_OUT_SIZE (CONV_OUT_SIZE / 2)
#define FC_INPUT_SIZE (POOL_OUT_SIZE * POOL_OUT_SIZE * CONV_OUT_CHANNELS)
#define LEARNING_RATE 0.01f
#define BATCH_SIZE 100
#define NUM_EPOCHS 100

// CNN Model parameters
struct CNNParams
{
    float* conv_kernels; // [CONV_OUT_CHANNELS, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE]
    float* conv_bias;    // [CONV_OUT_CHANNELS]
    float* fc_weights;   // [FC_INPUT_SIZE, NUM_CLASSES]
    float* fc_bias;      // [NUM_CLASSES]
};

// Print Image Sample for debugging
void print_image_sample(const Image& img)
{
    std::cout << "Image Label: " << img.label << "\n";
    std::cout << "First 1024 pixels (32x32 corner):\n";
    for (int i = 0; i < 32; ++i)
    {
        for (int j = 0; j < 32; ++j)
        {
            std::cout << std::fixed << std::setprecision(2)
                << img.pixels[i * IMAGE_SIZE + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Error checking function
void check_cuda_error(cudaError_t error, const char* message)
{
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << message << " - "
            << cudaGetErrorString(error) << "\n";
        exit(-1);
    }
}

// Randomly initialize parameters
/*Purpose: Allocates and initializes all CNN parameters (convolution kernels, biases, fully connected weights, biases)
on the GPU using Xavier initialization. Copies initialized values from host to device.*/
void init_parameters(CNNParams* params)
{
    cudaError_t error;

    // Allocate memory for convolution kernels
    error = cudaMalloc(&params->conv_kernels,
        CONV_OUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * sizeof(float));
    check_cuda_error(error, "Conv kernels allocation failed");

    // Allocate memory for convolution bias
    error = cudaMalloc(&params->conv_bias, CONV_OUT_CHANNELS * sizeof(float));
    check_cuda_error(error, "Conv bias allocation failed");

    // Allocate memory for fully connected weights
    error = cudaMalloc(&params->fc_weights, FC_INPUT_SIZE * NUM_CLASSES * sizeof(float));
    check_cuda_error(error, "FC weights allocation failed");

    // Allocate memory for fully connected bias
    error = cudaMalloc(&params->fc_bias, NUM_CLASSES * sizeof(float));
    check_cuda_error(error, "FC bias allocation failed");

    // Initialize host memory for random weights
    float* h_conv_kernels = new float[CONV_OUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE];
    float* h_conv_bias = new float[CONV_OUT_CHANNELS];
    float* h_fc_weights = new float[FC_INPUT_SIZE * NUM_CLASSES];
    float* h_fc_bias = new float[NUM_CLASSES];

    // Use Xavier initialization
    float conv_scale = sqrtf(3.0f / (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE));
    float fc_scale = sqrtf(3.0f / FC_INPUT_SIZE);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> conv_dist(-conv_scale, conv_scale);
    std::uniform_real_distribution<float> fc_dist(-fc_scale, fc_scale);

    // Initialize conv kernels
    for (int i = 0; i < CONV_OUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE; ++i)
    {
        h_conv_kernels[i] = conv_dist(gen);
    }

    // Initialize conv bias
    for (int i = 0; i < CONV_OUT_CHANNELS; ++i)
    {
        h_conv_bias[i] = 0.0f; // Initialize bias to zero
    }

    // Initialize FC weights
    for (int i = 0; i < FC_INPUT_SIZE * NUM_CLASSES; ++i)
    {
        h_fc_weights[i] = fc_dist(gen);
    }

    // Initialize FC bias
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        h_fc_bias[i] = 0.0f; // Initialize bias to zero
    }

    // Copy to device
    error = cudaMemcpy(params->conv_kernels, h_conv_kernels,
        CONV_OUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * sizeof(float),
        cudaMemcpyHostToDevice);
    check_cuda_error(error, "Conv kernels memcpy failed");

    error = cudaMemcpy(params->conv_bias, h_conv_bias,
        CONV_OUT_CHANNELS * sizeof(float),
        cudaMemcpyHostToDevice);
    check_cuda_error(error, "Conv bias memcpy failed");

    error = cudaMemcpy(params->fc_weights, h_fc_weights,
        FC_INPUT_SIZE * NUM_CLASSES * sizeof(float),
        cudaMemcpyHostToDevice);
    check_cuda_error(error, "FC weights memcpy failed");

    error = cudaMemcpy(params->fc_bias, h_fc_bias,
        NUM_CLASSES * sizeof(float),
        cudaMemcpyHostToDevice);
    check_cuda_error(error, "FC bias memcpy failed");

    // Free host memory
    delete[] h_conv_kernels;
    delete[] h_conv_bias;
    delete[] h_fc_weights;
    delete[] h_fc_bias;
}

__global__ void fc_kernel_batched(
    float* input,   // [batch, in_features]
    float* output,  // [batch, out_features]
    float* weights, // [in_features, out_features]
    float* bias,    // [out_features]
    int batch_size,
    int in_features,
    int out_features)
{
    int b = blockIdx.y; // batch index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && out_idx < out_features)
    {
        float sum = bias[out_idx];
        for (int i = 0; i < in_features; ++i)
        {
            sum += input[b * in_features + i] * weights[i * out_features + out_idx];
        }
        output[b * out_features + out_idx] = sum;
    }
}

// CUDA kernel for convolution operation
__global__ void conv2d_kernel(
    float* input,  // [batch, in_size*in_size]
    float* output, // [batch, out_channels, out_size, out_size]
    float* kernels, float* bias,
    int in_size, int kernel_size, int out_channels, int batch_size)
{
    int out_size = in_size - kernel_size + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z % out_channels;
    int batch = blockIdx.z / out_channels;

    if (col < out_size && row < out_size && channel < out_channels && batch < batch_size)
    {
        float sum = bias[channel];
        for (int k_row = 0; k_row < kernel_size; ++k_row)
        {
            for (int k_col = 0; k_col < kernel_size; ++k_col)
            {
                int input_row = row + k_row;
                int input_col = col + k_col;
                float k_val = kernels[channel * kernel_size * kernel_size + k_row * kernel_size + k_col];
                float in_val = input[batch * in_size * in_size + input_row * in_size + input_col];
                sum += in_val * k_val;
            }
        }
        output[batch * out_channels * out_size * out_size +
            channel * out_size * out_size +
            row * out_size + col] = sum;
    }
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel(float* input, float* output, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// CUDA kernel for max pooling
__global__ void maxpool_kernel(
    float* input, float* output,
    int in_channels, int in_size, int pool_size, int batch_size)
{
    int out_size = in_size / pool_size;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z % in_channels;
    int batch = blockIdx.z / in_channels;

    if (out_col < out_size && out_row < out_size && channel < in_channels && batch < batch_size)
    {
        float max_val = -INFINITY;
        for (int p_row = 0; p_row < pool_size; ++p_row)
        {
            for (int p_col = 0; p_col < pool_size; ++p_col)
            {
                int in_row = out_row * pool_size + p_row;
                int in_col = out_col * pool_size + p_col;
                float val = input[batch * in_channels * in_size * in_size +
                    channel * in_size * in_size +
                    in_row * in_size +
                    in_col];
                max_val = fmaxf(max_val, val);
            }
        }
        output[batch * in_channels * out_size * out_size +
            channel * out_size * out_size +
            out_row * out_size +
            out_col] = max_val;
    }
}

// CUDA kernel for fully connected layer (matrix multiplication)
__global__ void fc_kernel(float* input, float* output, float* weights, float* bias,
    int in_features, int out_features)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < out_features)
    {
        float sum = bias[out_idx];

        for (int i = 0; i < in_features; ++i)
        {
            sum += input[i] * weights[i * out_features + out_idx];
        }

        output[out_idx] = sum;
    }
}

// CUDA kernel for softmax activation
__global__ void softmax_kernel(float* input, float* output, int size)
{
    // Find maximum value for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < size; ++i)
    {
        max_val = fmaxf(max_val, input[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < size; ++i)
    {
        output[i] /= sum;
    }
}

__global__ void softmax_kernel_batched(
    float* input,  // [batch, num_classes]
    float* output, // [batch, num_classes]
    int batch_size,
    int num_classes)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size)
    {
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; ++i)
        {
            float v = input[b * num_classes + i];
            if (v > max_val)
                max_val = v;
        }
        float sum = 0.0f;
        for (int i = 0; i < num_classes; ++i)
        {
            output[b * num_classes + i] = expf(input[b * num_classes + i] - max_val);
            sum += output[b * num_classes + i];
        }
        for (int i = 0; i < num_classes; ++i)
        {
            output[b * num_classes + i] /= sum;
        }
    }
}

// CUDA kernel for cross-entropy loss and softmax gradient
__global__ void softmax_cross_entropy_kernel(float* softmax_output, int* labels,
    float* loss, float* gradient, int batch_size)
{
    float batch_loss = 0.0f;

    for (int i = 0; i < batch_size; ++i)
    {
        int label = labels[i];
        float prob = softmax_output[i * NUM_CLASSES + label];
        batch_loss -= logf(fmaxf(prob, 1e-10f)); // Prevent log(0)

        // Calculate gradient: softmax - one_hot
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            float target = (j == label) ? 1.0f : 0.0f;
            gradient[i * NUM_CLASSES + j] =
                (softmax_output[i * NUM_CLASSES + j] - target) / batch_size;
        }
    }

    *loss = batch_loss / batch_size;
}

// CUDA kernel for backward pass of fully connected layer
__global__ void fc_backward_kernel(float* input, float* grad_output,
    float* grad_weights, float* grad_bias,
    int batch_size, int in_features, int out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < in_features * out_features)
    {
        int in_idx = idx / out_features;
        int out_idx = idx % out_features;

        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b)
        {
            grad_sum += input[b * in_features + in_idx] *
                grad_output[b * out_features + out_idx];
        }

        grad_weights[idx] = grad_sum / batch_size;
    }

    // Calculate bias gradients
    if (idx < out_features)
    {
        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b)
        {
            grad_sum += grad_output[b * out_features + idx];
        }

        grad_bias[idx] = grad_sum / batch_size;
    }
}

// CUDA kernel for backward pass of convolution layer
__global__ void conv_backward_kernel(
    float* input,        // [batch, in_size*in_size]
    float* grad_output,  // [batch, out_channels, out_size, out_size]
    float* grad_kernels, // [out_channels, kernel_size, kernel_size]
    float* grad_bias,    // [out_channels]
    int batch_size,
    int in_size,
    int kernel_size,
    int out_channels)
{
    int k = blockIdx.x;  // kernel (output channel)
    int i = threadIdx.y; // kernel row
    int j = threadIdx.x; // kernel col

    if (k < out_channels && i < kernel_size && j < kernel_size)
    {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; ++b)
        {
            for (int out_row = 0; out_row < in_size - kernel_size + 1; ++out_row)
            {
                for (int out_col = 0; out_col < in_size - kernel_size + 1; ++out_col)
                {
                    int input_row = out_row + i;
                    int input_col = out_col + j;
                    float inp = input[b * in_size * in_size + input_row * in_size + input_col];
                    float grad_out = grad_output[b * out_channels * (in_size - kernel_size + 1) * (in_size - kernel_size + 1) +
                        k * (in_size - kernel_size + 1) * (in_size - kernel_size + 1) +
                        out_row * (in_size - kernel_size + 1) + out_col];
                    grad += inp * grad_out;
                }
            }
        }
        grad_kernels[k * kernel_size * kernel_size + i * kernel_size + j] = grad / batch_size;
    }

    // Bias gradient (one thread per output channel)
    if (i == 0 && j == 0 && k < out_channels)
    {
        float grad_b = 0.0f;
        int out_size = in_size - kernel_size + 1;
        for (int b = 0; b < batch_size; ++b)
        {
            for (int out_row = 0; out_row < out_size; ++out_row)
            {
                for (int out_col = 0; out_col < out_size; ++out_col)
                {
                    grad_b += grad_output[b * out_channels * out_size * out_size +
                        k * out_size * out_size +
                        out_row * out_size + out_col];
                }
            }
        }
        grad_bias[k] = grad_b / batch_size;
    }
}

__global__ void maxpool_backward_kernel(
    float* input,       // [BATCH_SIZE, CH, IN, IN]
    float* grad_output, // [BATCH_SIZE, CH, OUT, OUT]
    float* grad_input,  // [BATCH_SIZE, CH, IN, IN]
    int channels, int in_size, int pool_size, int batch_size)
{
    int out_size = in_size / pool_size;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z % channels;
    int batch = blockIdx.z / channels;

    if (out_col < out_size && out_row < out_size && channel < channels && batch < batch_size)
    {
        // Find max location in the input window
        float max_val = -INFINITY;
        int max_idx = 0;
        for (int p_row = 0; p_row < pool_size; ++p_row)
        {
            for (int p_col = 0; p_col < pool_size; ++p_col)
            {
                int in_row = out_row * pool_size + p_row;
                int in_col = out_col * pool_size + p_col;
                int idx = batch * channels * in_size * in_size +
                    channel * in_size * in_size +
                    in_row * in_size + in_col;
                float val = input[idx];
                if (val > max_val)
                {
                    max_val = val;
                    max_idx = idx;
                }
            }
        }
        // Route gradient only to max location
        float grad = grad_output[batch * channels * out_size * out_size +
            channel * out_size * out_size +
            out_row * out_size + out_col];
        // Set all grads to 0, except max_idx
        for (int p_row = 0; p_row < pool_size; ++p_row)
        {
            for (int p_col = 0; p_col < pool_size; ++p_col)
            {
                int in_row = out_row * pool_size + p_row;
                int in_col = out_col * pool_size + p_col;
                int idx = batch * channels * in_size * in_size +
                    channel * in_size * in_size +
                    in_row * in_size + in_col;
                grad_input[idx] = (idx == max_idx) ? grad : 0.0f;
            }
        }
    }
}

__global__ void relu_backward_kernel(
    float* input,       // [BATCH_SIZE, CH, SIZE, SIZE] (pre-activation)
    float* grad_output, // [BATCH_SIZE, CH, SIZE, SIZE]
    float* grad_input,  // [BATCH_SIZE, CH, SIZE, SIZE]
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// Function to update parameters with gradients (SGD)
__global__ void sgd_update_kernel(float* param, float* grad, int size, float lr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        param[idx] -= lr * grad[idx];
    }
}

// Function to train the model using CUDA
void train_cuda(const std::vector<Image>& dataset)
{
    std::cout << "\n=== Starting CUDA CNN Training on CIFAR-10 ===\n";

    // Debug: Print dataset info
    std::cout << "Dataset size: " << dataset.size() << " images\n";
    if (!dataset.empty())
    {
        std::cout << "\nFirst image sample:\n";
        print_image_sample(dataset[0]);
    }

    // Initialize CNN parameters
    CNNParams params;
    init_parameters(&params);
    std::cout << "CNN parameters initialized\n";

    // Allocate GPU memory for intermediate results
    float* d_images, * d_conv_output, * d_relu_output, * d_pool_output, * d_fc_output, * d_softmax_output;
    int* d_labels;
    float* d_loss;

    cudaError_t error;

    // Allocate memory for batch inputs and labels
    error = cudaMalloc(&d_images, BATCH_SIZE * IMAGE_PIXELS * sizeof(float));
    check_cuda_error(error, "Batch images allocation failed");

    error = cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int));
    check_cuda_error(error, "Batch labels allocation failed");

    // Allocate memory for intermediate outputs
    error = cudaMalloc(&d_conv_output,
        BATCH_SIZE * CONV_OUT_CHANNELS * CONV_OUT_SIZE * CONV_OUT_SIZE * sizeof(float));
    check_cuda_error(error, "Conv output allocation failed");

    error = cudaMalloc(&d_relu_output,
        BATCH_SIZE * CONV_OUT_CHANNELS * CONV_OUT_SIZE * CONV_OUT_SIZE * sizeof(float));
    check_cuda_error(error, "ReLU output allocation failed");

    error = cudaMalloc(&d_pool_output,
        BATCH_SIZE * CONV_OUT_CHANNELS * POOL_OUT_SIZE * POOL_OUT_SIZE * sizeof(float));
    check_cuda_error(error, "Pool output allocation failed");

    error = cudaMalloc(&d_fc_output, BATCH_SIZE * NUM_CLASSES * sizeof(float));
    check_cuda_error(error, "FC output allocation failed");

    error = cudaMalloc(&d_softmax_output, BATCH_SIZE * NUM_CLASSES * sizeof(float));
    check_cuda_error(error, "Softmax output allocation failed");

    error = cudaMalloc(&d_loss, sizeof(float));
    check_cuda_error(error, "Loss allocation failed");

    // Allocate memory for gradients
    float* d_softmax_grad, * d_fc_weights_grad, * d_fc_bias_grad;
    float* d_conv_kernels_grad, * d_conv_bias_grad;

    error = cudaMalloc(&d_softmax_grad, BATCH_SIZE * NUM_CLASSES * sizeof(float));
    check_cuda_error(error, "Softmax gradient allocation failed");

    error = cudaMalloc(&d_fc_weights_grad, FC_INPUT_SIZE * NUM_CLASSES * sizeof(float));
    check_cuda_error(error, "FC weights gradient allocation failed");

    error = cudaMalloc(&d_fc_bias_grad, NUM_CLASSES * sizeof(float));
    check_cuda_error(error, "FC bias gradient allocation failed");

    cudaMalloc(&d_conv_kernels_grad, CONV_OUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_conv_bias_grad, CONV_OUT_CHANNELS * sizeof(float));

    float* d_pool_grad_output; // [BATCH_SIZE, CONV_OUT_CHANNELS, POOL_OUT_SIZE, POOL_OUT_SIZE]
    float* d_relu_grad_output; // [BATCH_SIZE, CONV_OUT_CHANNELS, CONV_OUT_SIZE, CONV_OUT_SIZE]
    float* d_conv_grad_output; // [BATCH_SIZE, CONV_OUT_CHANNELS, CONV_OUT_SIZE, CONV_OUT_SIZE]

    cudaMalloc(&d_pool_grad_output, BATCH_SIZE * CONV_OUT_CHANNELS * POOL_OUT_SIZE * POOL_OUT_SIZE * sizeof(float));
    cudaMalloc(&d_relu_grad_output, BATCH_SIZE * CONV_OUT_CHANNELS * CONV_OUT_SIZE * CONV_OUT_SIZE * sizeof(float));
    cudaMalloc(&d_conv_grad_output, BATCH_SIZE * CONV_OUT_CHANNELS * CONV_OUT_SIZE * CONV_OUT_SIZE * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Training loop
    int total_epochs = NUM_EPOCHS;
    for (int epoch = 0; epoch < total_epochs; ++epoch)
    {
        std::cout << "\nEpoch " << epoch + 1 << "/" << total_epochs << "\n";

        // Create a copy of the dataset that we can shuffle
        std::vector<Image> shuffled_data = dataset;
        std::random_shuffle(shuffled_data.begin(), shuffled_data.end());

        int num_batches = shuffled_data.size() / BATCH_SIZE;
        float epoch_loss = 0.0f;
        int correct_predictions = 0;

        for (int batch = 0; batch < num_batches; ++batch)
        {
            // Prepare batch data
            std::vector<float> batch_images(BATCH_SIZE * IMAGE_PIXELS);
            std::vector<int> batch_labels(BATCH_SIZE);

            for (int i = 0; i < BATCH_SIZE; ++i)
            {
                const auto& img = shuffled_data[batch * BATCH_SIZE + i];
                std::copy(img.pixels.begin(), img.pixels.end(),
                    batch_images.begin() + i * IMAGE_PIXELS);
                batch_labels[i] = img.label;
            }

            // Copy batch data to GPU
            error = cudaMemcpy(d_images, batch_images.data(),
                BATCH_SIZE * IMAGE_PIXELS * sizeof(float),
                cudaMemcpyHostToDevice);
            check_cuda_error(error, "Batch images transfer failed");

            error = cudaMemcpy(d_labels, batch_labels.data(),
                BATCH_SIZE * sizeof(int),
                cudaMemcpyHostToDevice);
            check_cuda_error(error, "Batch labels transfer failed");

            // Forward pass
            dim3 conv_blocks(
                (CONV_OUT_SIZE + 7) / 8,
                (CONV_OUT_SIZE + 7) / 8,
                CONV_OUT_CHANNELS * BATCH_SIZE);
            dim3 conv_threads(8, 8);

            conv2d_kernel << <conv_blocks, conv_threads >> > (
                d_images,
                d_conv_output,
                params.conv_kernels,
                params.conv_bias,
                IMAGE_SIZE,
                CONV_KERNEL_SIZE,
                CONV_OUT_CHANNELS,
                BATCH_SIZE);
            check_cuda_error(cudaGetLastError(), "Conv2D kernel launch");
            check_cuda_error(cudaDeviceSynchronize(), "Conv2D kernel execution");

            // After conv2d_kernel
            int relu_size = BATCH_SIZE * CONV_OUT_CHANNELS * CONV_OUT_SIZE * CONV_OUT_SIZE;
            int relu_blocks = (relu_size + 255) / 256;
            relu_kernel << <relu_blocks, 256 >> > (
                d_conv_output,
                d_relu_output,
                relu_size);
            check_cuda_error(cudaGetLastError(), "ReLU kernel launch");
            check_cuda_error(cudaDeviceSynchronize(), "ReLU kernel execution");

            // MaxPool
            dim3 pool_blocks(
                (POOL_OUT_SIZE + 7) / 8,
                (POOL_OUT_SIZE + 7) / 8,
                CONV_OUT_CHANNELS * BATCH_SIZE);
            dim3 pool_threads(8, 8);
            maxpool_kernel << <pool_blocks, pool_threads >> > (
                d_relu_output,
                d_pool_output,
                CONV_OUT_CHANNELS,
                CONV_OUT_SIZE,
                2, // pool size
                BATCH_SIZE);
            check_cuda_error(cudaGetLastError(), "MaxPool kernel launch");
            check_cuda_error(cudaDeviceSynchronize(), "MaxPool kernel execution");

            // FC layer: launch with (ceil(NUM_CLASSES/256), BATCH_SIZE) grid
            dim3 fc_blocks((NUM_CLASSES + 255) / 256, BATCH_SIZE);
            fc_kernel_batched << <fc_blocks, 256 >> > (
                d_pool_output, // [BATCH_SIZE, FC_INPUT_SIZE]
                d_fc_output,   // [BATCH_SIZE, NUM_CLASSES]
                params.fc_weights,
                params.fc_bias,
                BATCH_SIZE,
                FC_INPUT_SIZE,
                NUM_CLASSES);
            check_cuda_error(cudaGetLastError(), "FC kernel launch");
            check_cuda_error(cudaDeviceSynchronize(), "FC kernel execution");

            // Softmax: launch with (ceil(BATCH_SIZE/128)), 128
            int softmax_blocks = (BATCH_SIZE + 127) / 128;
            softmax_kernel_batched << <softmax_blocks, 128 >> > (
                d_fc_output,
                d_softmax_output,
                BATCH_SIZE,
                NUM_CLASSES);
            check_cuda_error(cudaGetLastError(), "Softmax kernel launch");
            check_cuda_error(cudaDeviceSynchronize(), "Softmax kernel execution");

            // Loss and gradient computation
            softmax_cross_entropy_kernel << <1, 1 >> > (
                d_softmax_output,
                d_labels,
                d_loss,
                d_softmax_grad,
                BATCH_SIZE);

            // Backward pass - fully connected layer
            int fc_grad_blocks = (FC_INPUT_SIZE * NUM_CLASSES + 255) / 256;
            fc_backward_kernel << <fc_grad_blocks, 256 >> > (
                d_pool_output,
                d_softmax_grad,
                d_fc_weights_grad,
                d_fc_bias_grad,
                BATCH_SIZE,
                FC_INPUT_SIZE,
                NUM_CLASSES);

            // Update parameters
            int fc_weights_blocks = (FC_INPUT_SIZE * NUM_CLASSES + 255) / 256;
            sgd_update_kernel << <fc_weights_blocks, 256 >> > (
                params.fc_weights,
                d_fc_weights_grad,
                FC_INPUT_SIZE * NUM_CLASSES,
                LEARNING_RATE);

            int fc_bias_blocks = (NUM_CLASSES + 255) / 256;
            sgd_update_kernel << <fc_bias_blocks, 256 >> > (
                params.fc_bias,
                d_fc_bias_grad,
                NUM_CLASSES,
                LEARNING_RATE);

            // 1. Compute gradient w.r.t. pool output (from FC backward)
            cudaMemset(d_pool_grad_output, 0, BATCH_SIZE * CONV_OUT_CHANNELS * POOL_OUT_SIZE * POOL_OUT_SIZE * sizeof(float));

            // 2. MaxPool backward
            int pool_bwd_x = std::max(1, (POOL_OUT_SIZE + 7) / 8);
            int pool_bwd_y = std::max(1, (POOL_OUT_SIZE + 7) / 8);
            int pool_bwd_z = std::max(1, CONV_OUT_CHANNELS * BATCH_SIZE);
            if (pool_bwd_z > 65535) {
                std::cerr << "Error: pool_bwd_z (" << pool_bwd_z << ") exceeds CUDA grid limit (65535).\n";
                exit(-1);
            }
            dim3 pool_bwd_blocks(pool_bwd_x, pool_bwd_y, pool_bwd_z);
            dim3 pool_bwd_threads(8, 8);
            // std::cout << "Launching maxpool_backward_kernel with grid (" 
            //           << pool_bwd_x << ", " << pool_bwd_y << ", " << pool_bwd_z << ")\n";
            maxpool_backward_kernel << <pool_bwd_blocks, pool_bwd_threads >> > (
                d_relu_output,
                d_pool_grad_output,
                d_relu_grad_output,
                CONV_OUT_CHANNELS,
                CONV_OUT_SIZE,
                2, // pool size
                BATCH_SIZE
                );
            check_cuda_error(cudaGetLastError(), "MaxPool backward kernel launch");
            check_cuda_error(cudaDeviceSynchronize(), "MaxPool backward kernel execution");

            // 3. ReLU backward
            int relu_bwd_size = BATCH_SIZE * CONV_OUT_CHANNELS * CONV_OUT_SIZE * CONV_OUT_SIZE;
            int relu_bwd_blocks = (relu_bwd_size + 255) / 256;
            relu_backward_kernel << <relu_bwd_blocks, 256 >> > (
                d_conv_output,      // input to relu (forward)
                d_relu_grad_output, // grad from above
                d_conv_grad_output, // grad to below (for conv backward)
                relu_bwd_size);
            check_cuda_error(cudaGetLastError(), "ReLU backward kernel launch");
            check_cuda_error(cudaDeviceSynchronize(), "ReLU backward kernel execution");

            // Launch conv_backward_kernel
            dim3 conv_grad_blocks(CONV_OUT_CHANNELS);
            dim3 conv_grad_threads(CONV_KERNEL_SIZE, CONV_KERNEL_SIZE);
            conv_backward_kernel << <conv_grad_blocks, conv_grad_threads >> > (
                d_images,
                d_conv_grad_output, // You need to compute this, see below!
                d_conv_kernels_grad,
                d_conv_bias_grad,
                BATCH_SIZE,
                IMAGE_SIZE,
                CONV_KERNEL_SIZE,
                CONV_OUT_CHANNELS);
            check_cuda_error(cudaGetLastError(), "Conv backward kernel launch");
            check_cuda_error(cudaDeviceSynchronize(), "Conv backward kernel execution");

            int conv_weights_blocks = (CONV_OUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE + 255) / 256;
            sgd_update_kernel << <conv_weights_blocks, 256 >> > (
                params.conv_kernels,
                d_conv_kernels_grad,
                CONV_OUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE,
                LEARNING_RATE);

            int conv_bias_blocks = (CONV_OUT_CHANNELS + 255) / 256;
            sgd_update_kernel << <conv_bias_blocks, 256 >> > (
                params.conv_bias,
                d_conv_bias_grad,
                CONV_OUT_CHANNELS,
                LEARNING_RATE);

            // std::cout << "d_relu_output: " << d_relu_output << "\n";
            // std::cout << "d_pool_grad_output: " << d_pool_grad_output << "\n";
            // std::cout << "d_relu_grad_output: " << d_relu_grad_output << "\n";

            // Compute accuracy for this batch
            std::vector<float> h_softmax_output(BATCH_SIZE * NUM_CLASSES);
            error = cudaMemcpy(h_softmax_output.data(), d_softmax_output,
                BATCH_SIZE * NUM_CLASSES * sizeof(float),
                cudaMemcpyDeviceToHost);
            check_cuda_error(error, "Softmax output transfer failed");

            float h_loss;
            error = cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            check_cuda_error(error, "Loss transfer failed");

            epoch_loss += h_loss;

            // Count correct predictions
            for (int i = 0; i < BATCH_SIZE; ++i)
            {
                int predicted_class = 0;
                float max_prob = h_softmax_output[i * NUM_CLASSES];

                for (int j = 1; j < NUM_CLASSES; ++j)
                {
                    if (h_softmax_output[i * NUM_CLASSES + j] > max_prob)
                    {
                        max_prob = h_softmax_output[i * NUM_CLASSES + j];
                        predicted_class = j;
                    }
                }

                if (predicted_class == batch_labels[i])
                {
                    correct_predictions++;
                }
            }

            // Show progress
            if (batch % 10 == 0 || batch == num_batches - 1)
            {
                float accuracy = static_cast<float>(correct_predictions) / ((batch + 1) * BATCH_SIZE);
                float avg_loss = epoch_loss / (batch + 1);

                std::cout << "\rBatch " << batch + 1 << "/" << num_batches
                    << " - Loss: " << std::fixed << std::setprecision(4) << avg_loss
                    << " - Accuracy: " << std::fixed << std::setprecision(2)
                    << (accuracy * 100.0f) << "%" << std::flush;
            }
        }

        // Epoch summary
        float epoch_accuracy = static_cast<float>(correct_predictions) / (num_batches * BATCH_SIZE);
        epoch_loss /= num_batches;

        std::cout << "\nEpoch " << epoch + 1 << " completed"
            << " - Loss: " << std::fixed << std::setprecision(4) << epoch_loss
            << " - Accuracy: " << std::fixed << std::setprecision(2)
            << (epoch_accuracy * 100.0f) << "%" << std::endl;
    }

    // Timing results
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\nCUDA CNN training completed in "
        << milliseconds / 1000.0f << " seconds\n";

    // Export model weights for evaluation
    std::vector<float> h_fc_weights(FC_INPUT_SIZE * NUM_CLASSES);
    std::vector<float> h_fc_bias(NUM_CLASSES); // Changed to vector
    std::vector<float> h_conv_kernels(CONV_OUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE);
    std::vector<float> h_conv_bias(CONV_OUT_CHANNELS);

    // Copy parameters from device to host
    cudaMemcpy(h_fc_weights.data(), params.fc_weights,
        FC_INPUT_SIZE * NUM_CLASSES * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fc_bias.data(), params.fc_bias,
        NUM_CLASSES * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_conv_kernels.data(), params.conv_kernels,
        CONV_OUT_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_conv_bias.data(), params.conv_bias,
        CONV_OUT_CHANNELS * sizeof(float),
        cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_conv_output);
    cudaFree(d_relu_output);
    cudaFree(d_pool_output);
    cudaFree(d_fc_output);
    cudaFree(d_softmax_output);
    cudaFree(d_loss);
    cudaFree(d_softmax_grad);
    cudaFree(d_fc_weights_grad);
    cudaFree(d_fc_bias_grad);
    cudaFree(d_conv_kernels_grad);
    cudaFree(d_conv_bias_grad);
    cudaFree(params.conv_kernels);
    cudaFree(params.conv_bias);
    cudaFree(params.fc_weights);
    cudaFree(params.fc_bias);
    cudaFree(d_pool_grad_output);
    cudaFree(d_relu_grad_output);
    cudaFree(d_conv_grad_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Run evaluation on test set if available
    std::cout << "\nEvaluating model on test data...\n";
    evaluate_model("data/test_batch.bin", h_conv_kernels, h_conv_bias, h_fc_weights, h_fc_bias);
}

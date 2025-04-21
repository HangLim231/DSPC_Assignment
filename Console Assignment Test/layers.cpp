// File: layers.cpp
#include "layers.h"
#include <algorithm>
#include <cmath>

// Perform 2D convolution on a single-channel image
Matrix conv2d(const Matrix& input, const Matrix& kernel) {
    int out_h = input.size() - kernel.size() + 1;
    int out_w = input[0].size() - kernel[0].size() + 1;
    Matrix output(out_h, std::vector<float>(out_w, 0.0f));

    for (int i = 0; i < out_h; ++i) {
        for (int j = 0; j < out_w; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel.size(); ++ki) {
                for (int kj = 0; kj < kernel[0].size(); ++kj) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

// Apply ReLU activation function
Matrix relu(const Matrix& input) {
    Matrix output = input;
    for (auto& row : output)
        for (auto& val : row)
            val = std::max(0.0f, val);
    return output;
}


// Perform 2x2 max pooling
Matrix maxpool2x2(const Matrix& input) {
    int out_h = input.size() / 2;
    int out_w = input[0].size() / 2;
    Matrix output(out_h, std::vector<float>(out_w, 0.0f));

    for (int i = 0; i < out_h; ++i) {
        for (int j = 0; j < out_w; ++j) {
            float max_val = 0.0f;
            for (int m = 0; m < 2; ++m)
                for (int n = 0; n < 2; ++n)
                    max_val = std::max(max_val, input[i * 2 + m][j * 2 + n]);
            output[i][j] = max_val;
        }
    }
    return output;
}

// Flatten the 2D matrix into a 1D vector
std::vector<float> flatten(const Matrix& input) {
    std::vector<float> flat;
    for (const auto& row : input)
        flat.insert(flat.end(), row.begin(), row.end());
    return flat;
}

// Fully connected layer
float fully_connected(const std::vector<float>& input, const std::vector<float>& weights, float bias) {
    float sum = bias;
    for (size_t i = 0; i < input.size(); ++i) {
        sum += input[i] * weights[i];
    }
    return sum;
}

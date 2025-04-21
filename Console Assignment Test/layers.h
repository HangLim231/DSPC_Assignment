// File: layers.h
#pragma once
#include <vector>

// Define matrix type for 2D operations
typedef std::vector<std::vector<float>> Matrix;

// Neural network layer operations
Matrix conv2d(const Matrix& input, const Matrix& kernel);
Matrix relu(const Matrix& input);
Matrix maxpool2x2(const Matrix& input);
std::vector<float> flatten(const Matrix& input);
float fully_connected(const std::vector<float>& input, const std::vector<float>& weights, float bias);

// New operations for CNN
std::vector<float> softmax(const std::vector<float>& input);
float cross_entropy_loss(const std::vector<float>& softmax_output, int true_class);
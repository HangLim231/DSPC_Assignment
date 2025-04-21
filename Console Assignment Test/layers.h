// File: layers.h
#pragma once
#include <vector>

// Perform 2D convolution on a single-channel image
typedef std::vector<std::vector<float>> Matrix;

Matrix conv2d(const Matrix& input, const Matrix& kernel);
Matrix relu(const Matrix& input);
Matrix maxpool2x2(const Matrix& input);
std::vector<float> flatten(const Matrix& input);

float fully_connected(const std::vector<float>& input, const std::vector<float>& weights, float bias);

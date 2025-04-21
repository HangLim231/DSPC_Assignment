// File: train_cuda.h
#pragma once
#include <vector>
#include "loader.h"

// Function to train the model using CUDA
void train_cuda(const std::vector<Image>& dataset);


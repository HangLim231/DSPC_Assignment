// File: evaluate.h
#pragma once
#include "loader.h"
#include <vector>
#include <string>

// Predict the class of an image using full CNN weights
int predict(const Image& img,
    const std::vector<float>& conv_kernels,
    const std::vector<float>& conv_bias,
    const std::vector<float>& fc_weights,
    const std::vector<float>& fc_bias);

// Evaluate a trained CNN model on test dataset
void evaluate_model(const std::string& test_path,
    const std::vector<float>& conv_kernels,
    const std::vector<float>& conv_bias,
    const std::vector<float>& fc_weights,
    const std::vector<float>& fc_bias);

void displayPredictionResults(const Image& img, int prediction, int actual_label);

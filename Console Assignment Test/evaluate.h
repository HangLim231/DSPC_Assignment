// File: evaluate.h
#pragma once
#include "loader.h"
#include <vector>
#include <string>

// Function declarations
void displayPredictionResults(const Image& img, int prediction, int actual_label);
int predict(const Image& img, const std::vector<float>& fc_weights, float bias);
void evaluate_model(const std::string& test_path, const std::vector<float>& fc_weights, float bias);
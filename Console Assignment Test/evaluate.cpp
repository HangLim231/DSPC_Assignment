// File: evaluate.cpp
#include "loader.h"
#include "layers.h"
#include <iostream>
#include <chrono>

// Run a simple CNN forward pass and predict label
int predict(const Image& img, const std::vector<float>& fc_weights, float bias) {
    // Convert flat image to 2D matrix
    Matrix input(IMAGE_SIZE, std::vector<float>(IMAGE_SIZE));
    for (int i = 0; i < IMAGE_SIZE; ++i)
        for (int j = 0; j < IMAGE_SIZE; ++j)
            input[i][j] = img.pixels[i * IMAGE_SIZE + j];

    // Apply conv, relu, pool, flatten, fc
    Matrix kernel = { {1, 0, -1}, {1, 0, -1}, {1, 0, -1} }; // edge detector
    Matrix conv_out = conv2d(input, kernel);
    Matrix activated = relu(conv_out);
    Matrix pooled = maxpool2x2(activated);
    std::vector<float> features = flatten(pooled);

    float output = fully_connected(features, fc_weights, bias);
    return output > 0.5f ? 1 : 0; // binary decision for test
}

// Evaluate model using test set
void evaluate_model(const std::string& test_path, const std::vector<float>& fc_weights, float bias) {
    std::vector<Image> test_set = load_dataset(test_path);
    int correct = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& img : test_set) {
        int prediction = predict(img, fc_weights, bias);
        int label_bin = (img.label == 0 ? 1 : 0); // just compare with label 0 for now
        if (prediction == label_bin) correct++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float accuracy = (float)correct / test_set.size();
    std::cout << "Test accuracy: " << accuracy * 100.0f << "%\n";
    std::cout << "Evaluation time: " << elapsed.count() << " seconds\n";
}

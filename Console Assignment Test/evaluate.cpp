// File: evaluate.cpp
#include "loader.h"
#include "layers.h"
#include <iostream>
#include <chrono>
#include "visualization.h"
#include "evaluate.h"


// Add these new functions before the existing predict function
void displayPredictionResults(const Image& img, int prediction, int actual_label) {
    std::cout << "\n=== Prediction Results ===\n";
    Visualizer::displayImage(img);
    std::cout << "Predicted class: " << prediction << "\n";
    std::cout << "Actual class: " << actual_label << "\n";
    std::cout << "Result: " << (prediction == actual_label ? "Correct!" : "Incorrect") << "\n\n";
}

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

// Modify the existing evaluate_model function
void evaluate_model(const std::string& test_path, const std::vector<float>& fc_weights, float bias) {
    std::vector<Image> test_set = load_dataset({ test_path });
    int correct = 0;

    std::cout << "\n=== Starting Model Evaluation ===\n";
    std::cout << "Test set size: " << test_set.size() << " images\n\n";

    auto start = std::chrono::high_resolution_clock::now();

    // Show first few predictions as examples
    const int num_examples = 3;
    for (int i = 0; i < std::min(num_examples, (int)test_set.size()); ++i) {
        int prediction = predict(test_set[i], fc_weights, bias);
        displayPredictionResults(test_set[i], prediction, test_set[i].label);
    }

    // Continue with remaining evaluations
    for (size_t i = num_examples; i < test_set.size(); ++i) {
        int prediction = predict(test_set[i], fc_weights, bias);
        int label_bin = (test_set[i].label == 0 ? 1 : 0);
        if (prediction == label_bin) correct++;

        // Show progress
        if (i % 100 == 0) {
            std::cout << "\rProcessing: " << i << "/" << test_set.size()
                << " images" << std::flush;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float accuracy = (float)correct / test_set.size();
    std::cout << "\n\n=== Evaluation Results ===\n";
    std::cout << "Test accuracy: " << std::fixed << std::setprecision(2)
        << accuracy * 100.0f << "%\n";
    std::cout << "Evaluation time: " << elapsed.count() << " seconds\n";
    std::cout << "Total images processed: " << test_set.size() << "\n";
}

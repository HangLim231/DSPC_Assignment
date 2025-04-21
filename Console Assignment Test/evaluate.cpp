// File: evaluate.cpp
#include "loader.h"
#include "layers.h"
#include <iostream>
#include <chrono>
#include "visualization.h"
#include "evaluate.h"

// Constants for CNN architecture (make sure they match train_cuda.cu)
#define CONV_KERNEL_SIZE 5
#define CONV_OUT_CHANNELS 16
#define CONV_OUT_SIZE (IMAGE_SIZE - CONV_KERNEL_SIZE + 1)
#define POOL_OUT_SIZE (CONV_OUT_SIZE / 2)
#define FC_INPUT_SIZE (POOL_OUT_SIZE * POOL_OUT_SIZE * CONV_OUT_CHANNELS)

// Function to display prediction results
void displayPredictionResults(const Image& img, int prediction, int actual_label) {
    std::cout << "\n=== Prediction Results ===\n";
    Visualizer::displayImage(img);
    std::cout << "Predicted class: " << prediction << "\n";
    std::cout << "Actual class: " << actual_label << "\n";
    std::cout << "Result: " << (prediction == actual_label ? "Correct!" : "Incorrect") << "\n\n";
}

// Function to predict the class of an image using our trained model
int predict(const Image& img, const std::vector<float>& fc_weights, float bias) {
    // Convert flat image to 2D matrix 
    Matrix input(IMAGE_SIZE, std::vector<float>(IMAGE_SIZE));
    for (int i = 0; i < IMAGE_SIZE; ++i)
        for (int j = 0; j < IMAGE_SIZE; ++j)
            input[i][j] = img.pixels[i * IMAGE_SIZE + j];

    // Apply predefined edge detection kernel for convolution layer
    Matrix kernel = { {1, 0, -1}, {1, 0, -1}, {1, 0, -1} }; // edge detector

    // Forward pass through CNN layers
    Matrix conv_out = conv2d(input, kernel);
    Matrix activated = relu(conv_out);
    Matrix pooled = maxpool2x2(activated);
    std::vector<float> features = flatten(pooled);

    // Apply fully connected layer with trained weights
    // For simplicity in this implementation, we're using a binary classifier
    // so we just need to determine if the output is > 0.5
    float output = fully_connected(features, fc_weights, bias);

    // For multi-class classification, we would need to compute the argmax
    // In this simplified example, we're just doing binary classification
    return output > 0.5f ? 1 : 0;
}

// Function to evaluate the model on a test set
void evaluate_model(const std::string& test_path, const std::vector<float>& fc_weights, float bias) {
    std::vector<Image> test_set = load_dataset({ test_path });
    if (test_set.empty()) {
        std::cout << "Warning: No test images loaded from " << test_path << "\n";
        return;
    }

    int correct = 0;
    int total = 0;

    std::cout << "\n=== Starting Model Evaluation ===\n";
    std::cout << "Test set size: " << test_set.size() << " images\n\n";

    auto start = std::chrono::high_resolution_clock::now();

    // Show first few predictions as examples
    const int num_examples = std::min(3, static_cast<int>(test_set.size()));
    for (int i = 0; i < num_examples; ++i) {
        int prediction = predict(test_set[i], fc_weights, bias);
        displayPredictionResults(test_set[i], prediction, test_set[i].label);
    }

    // Continue with remaining evaluations
    std::cout << "Evaluating remaining images...\n";
    for (size_t i = num_examples; i < test_set.size(); ++i) {
        int prediction = predict(test_set[i], fc_weights, bias);

        // For the CIFAR-10 dataset with multiple classes
        // In a full implementation, we would handle all 10 classes here
        // For this simplified example, we're treating it as a binary problem
        int binary_label = (test_set[i].label == 0) ? 0 : 1;

        if (prediction == binary_label) {
            correct++;
        }
        total++;

        // Show progress
        if (i % 100 == 0) {
            std::cout << "\rProcessing: " << i << "/" << test_set.size()
                << " images" << std::flush;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate accuracy
    float accuracy = static_cast<float>(correct) / total;
    std::cout << "\n\n=== Evaluation Results ===\n";
    std::cout << "Test accuracy: " << std::fixed << std::setprecision(2)
        << accuracy * 100.0f << "%\n";
    std::cout << "Evaluation time: " << elapsed.count() << " seconds\n";
    std::cout << "Total images processed: " << total << "\n";
    std::cout << "Correct predictions: " << correct << "\n";
}
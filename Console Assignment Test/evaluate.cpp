// File: evaluate.cpp
#include "loader.h"
#include "layers.h"
#include "visualization.h"
#include "evaluate.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <iomanip>

using namespace std;

#define CONV_KERNEL_SIZE 5
#define CONV_OUT_CHANNELS 16 // This simplified CPU-side version uses 1 filter only
#define CONV_OUT_SIZE (IMAGE_SIZE - CONV_KERNEL_SIZE + 1)
#define POOL_OUT_SIZE (CONV_OUT_SIZE / 2)
#define FC_INPUT_SIZE (POOL_OUT_SIZE * POOL_OUT_SIZE * CONV_OUT_CHANNELS)
#define NUM_CLASSES 10

// Predict label using CNN weights
int predict(const Image& img,
    const std::vector<float>& conv_kernels,
    const std::vector<float>& conv_bias,
    const std::vector<float>& fc_weights,
    const std::vector<float>& fc_bias)
{

    // Convert flat 1D image to 2D matrix
    Matrix input(IMAGE_SIZE, std::vector<float>(IMAGE_SIZE));
    for (int i = 0; i < IMAGE_SIZE; ++i)
        for (int j = 0; j < IMAGE_SIZE; ++j)
            input[i][j] = img.pixels[i * IMAGE_SIZE + j];

    // Multi-channel convolution
    std::vector<Matrix> conv_outputs(CONV_OUT_CHANNELS);
    for (int ch = 0; ch < CONV_OUT_CHANNELS; ++ch)
    {
        // Extract kernel for this channel
        Matrix kernel(CONV_KERNEL_SIZE, std::vector<float>(CONV_KERNEL_SIZE));
        for (int i = 0; i < CONV_KERNEL_SIZE; ++i)
            for (int j = 0; j < CONV_KERNEL_SIZE; ++j)
                kernel[i][j] = conv_kernels[ch * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE + i * CONV_KERNEL_SIZE + j];

        // Conv -> ReLU -> Pool
        Matrix conv = conv2d(input, kernel);
        Matrix activated = relu(conv);
        Matrix pooled = maxpool2x2(activated);
        conv_outputs[ch] = pooled;
    }

    // Flatten all channels
    std::vector<float> features;
    for (const auto& pooled : conv_outputs)
    {
        std::vector<float> flat = flatten(pooled);
        features.insert(features.end(), flat.begin(), flat.end());
    }

    // Fully Connected layer
    std::vector<float> logits(NUM_CLASSES);
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        float sum = fc_bias[c];
        for (int i = 0; i < features.size(); ++i)
            sum += features[i] * fc_weights[i * NUM_CLASSES + c];
        logits[c] = sum;
    }

    // Softmax + argmax
    std::vector<float> probs = softmax(logits);
    int predicted_class = std::max_element(probs.begin(), probs.end()) - probs.begin();
    return predicted_class;
}


// Display image + predicted vs actual label
void displayPredictionResults(const Image& img, int prediction, int actual_label)
{
    cout << "\n=== Prediction Result ===\n";
    Visualizer::displayImage(img);
    cout << "Predicted: " << prediction << " | Actual: " << actual_label
        << " -> " << (prediction == actual_label ? "Correct" : "Incorrect") << "\n";
}

// Run full evaluation on test dataset
void evaluate_model(const std::string& test_file,
    const std::vector<float>& conv_kernels,
    const std::vector<float>& conv_bias,
    const std::vector<float>& fc_weights,
    const std::vector<float>& fc_bias)
{

    vector<Image> test_set = load_dataset({ test_file });
    if (test_set.empty())
    {
        cerr << "Failed to load test data from: " << test_file << "\n";
        return;
    }

    int correct = 0;
    auto start = chrono::high_resolution_clock::now();

    // Display prediction for first few images
    const int show_count = min(3, static_cast<int>(test_set.size()));
    for (int i = 0; i < show_count; ++i)
    {
        int pred = predict(test_set[i], conv_kernels, conv_bias, fc_weights, fc_bias);
        displayPredictionResults(test_set[i], pred, test_set[i].label);
        if (pred == test_set[i].label)
            correct++;
    }

    // Batch evaluate the rest
    for (size_t i = show_count; i < test_set.size(); ++i)
    {
        int pred = predict(test_set[i], conv_kernels, conv_bias, fc_weights, fc_bias);
        if (pred == test_set[i].label)
            correct++;
        if (i % 100 == 0)
        {
            cout << "\rEvaluated: " << i << " / " << test_set.size() << flush;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    float accuracy = static_cast<float>(correct) / test_set.size();

    // Final stats
    cout << "\n\n=== Evaluation Complete ===\n";
    cout << "Accuracy: " << fixed << setprecision(2) << (accuracy * 100.0f) << "%\n";
    cout << "Processed: " << test_set.size() << " images\n";
    cout << "Correct predictions: " << correct << "\n";
    cout << "Evaluation Time: " << duration.count() << " seconds\n";
}

//File name: visualization.h
#pragma once
#include "loader.h"
#include <iostream>
#include <iomanip>
#include <string>

// Class to visualize images and training progress
class Visualizer {
public:
    static void displayImage(const Image& img) {
        std::cout << "\n=== Image Visualization ===\n";
        std::cout << "Label: " << img.label << "\n";
        std::cout << "32x32 Image Preview:\n";

        // Display the image as ASCII art
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                float pixel = img.pixels[i * IMAGE_SIZE + j];
                char c = pixelToAscii(pixel);
                std::cout << c << c;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // Function to display training progress
    static void displayProgress(int epoch, int total_epochs, float accuracy, float loss) {
        std::cout << "\rEpoch " << epoch << "/" << total_epochs
            << " - Accuracy: " << std::fixed << std::setprecision(2)
            << (accuracy * 100) << "% - Loss: " << loss << std::flush;
    }

private:
    static char pixelToAscii(float pixel) {
        const char* ascii_chars = " .:-=+*#%@";
        int index = static_cast<int>(pixel * 9);
        return ascii_chars[std::min(9, std::max(0, index))];
    }
};
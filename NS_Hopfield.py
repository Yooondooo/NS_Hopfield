import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        """Train the Hopfield network using Hebbian learning."""
        num_patterns = len(patterns)
        for p in patterns:
            p = p.reshape(-1, 1)  # Convert pattern to column vector
            self.weights += np.dot(p, p.T)
        # Remove self-connections
        np.fill_diagonal(self.weights, 0)

    # def predict(self, input_pattern, max_iterations=10):
    #     """Predict the output for a given input pattern."""
    #     input_pattern = input_pattern.copy()
    #     for _ in range(max_iterations):
    #         prev_pattern = input_pattern.copy()
    #         for i in range(self.num_neurons):
    #             input_sum = np.dot(self.weights[i], input_pattern)
    #             input_pattern[i] = 1 if input_sum > 0 else -1
    #         # If the pattern has stabilized, stop
    #         if np.array_equal(input_pattern, prev_pattern):
    #             break
    #     return input_pattern
    def predict(self, input_pattern, max_iterations=10):
        """Predict the output for a given input pattern."""
        input_pattern = input_pattern.copy()

        print("Initial pattern:")
        self.display_pattern(input_pattern)  # Показываем начальное состояние

        for iteration in range(max_iterations):
            prev_pattern = input_pattern.copy()
            for i in range(self.num_neurons):
                input_sum = np.dot(self.weights[i], input_pattern)
                input_pattern[i] = 1 if input_sum > 0 else -1
                if (( i + 1) % 100 ==0):
                    print(f"\nIteration {i + 1}:")
                    self.display_pattern(input_pattern)  # Показываем состояние после итерации

            # Если паттерн стабилизировался, выходим
            if np.array_equal(input_pattern, prev_pattern):
                print(f"\nPattern stabilized at iteration {iteration + 1}")
                break

        return input_pattern

    def convert_image_to_pattern(self, image_path):
        """Convert an image to a binary pattern."""
        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))  # Ensure the image size is 28x28
        pattern = np.array(image).flatten()
        # Normalize to +1 and -1
        pattern = np.where(pattern > 128, 1, -1)
        return pattern

    def display_pattern(self, pattern):
        """Display the pattern as an image."""
        pattern = pattern.reshape(28, 28)
        plt.imshow(pattern, cmap='gray')
        plt.show()

# Load sample images (ensure they are in the correct format and path)
letter_images = {
    'A': ['data/data10/A0.png'],
    'B': ['data/data10/B0.png'],
    'C': ['data/data10/C0.png'],
}

# Initialize the Hopfield network with enough neurons
num_neurons = 28 * 28  # 784 neurons for 28x28 image
hopfield = HopfieldNetwork(num_neurons)

# Train the network on the patterns
patterns = []
for letter, images in letter_images.items():
    for img_path in images:
        pattern = hopfield.convert_image_to_pattern(img_path)
        patterns.append(pattern)
hopfield.train(np.array(patterns))

# Test the network with a noisy image
test_image = 'data/data11/А1.png'
noisy_pattern = hopfield.convert_image_to_pattern(test_image)

# Introduce noise (randomly flip some bits)
noise_level = 0
num_noise = int(noise_level * len(noisy_pattern))
noisy_indices = np.random.choice(len(noisy_pattern), num_noise, replace=False)
noisy_pattern[noisy_indices] = -noisy_pattern[noisy_indices]

# Predict the output pattern
output_pattern = hopfield.predict(noisy_pattern)

# Display the predicted letter
hopfield.display_pattern(noisy_pattern)
hopfield.display_pattern(output_pattern)


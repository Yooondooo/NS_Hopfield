import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from PIL import Image
import os
import cv2

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        patterns = np.array(patterns)
        # Векторизованное обучение: W = (1/N) * Σ(p_i * p_i^T)
        self.weights = np.dot(patterns.T, patterns) / patterns.shape[0]
        np.fill_diagonal(self.weights, 0)

    def predict(self, input_pattern, max_iterations=40, async_update=False, show_progress=False):
        input_pattern = input_pattern.copy()
        pattern = input_pattern
        if show_progress:
            print("Initial pattern:")
            self.display_pattern(pattern)
        for iteration in range(max_iterations):
            prev_pattern = pattern.copy()
            if async_update:
                indices = np.arange(self.num_neurons)
                np.random.shuffle(indices)
                for i in indices:
                    pattern[i] = 1 if np.dot(self.weights[i], pattern) > 0 else -1
            else:
                net_input = np.dot(self.weights, pattern)
                pattern = np.where(net_input > 0, 1, -1)
            if show_progress:
                print(f"\nIteration {iteration + 1}:")
                self.display_pattern(pattern)

            if np.array_equal(pattern, prev_pattern):
                if show_progress:
                    print(f"\nPattern stabilized at iteration {iteration + 1}")
                break

        return pattern

    @staticmethod
    def convert_image_to_pattern(image_path):
        image = Image.open(image_path).convert('L')
        image = image.resize((100,100))
        pattern = np.array(image).flatten()
        pattern = np.where(pattern > 128, 1, -1)
        return pattern

    def display_pattern(self, pattern):
        pattern = pattern.reshape(100,100)
        plt.imshow(pattern, cmap='gray')
        plt.show()

    # def recognize_letter(self, output_pattern, letter_patterns):
    #     min_distance = float('inf')
    #     os.makedirs(f'patterns', exist_ok=True)
    #     recognized_letter = None
    #     for letter, patterns in letter_patterns.items():
    #         for pattern in patterns:
    #             cv2.imwrite(f'patterns/{letter}.png',pattern)
    #             distance = euclidean(output_pattern, pattern)
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 recognized_letter = letter
    #     return recognized_letter

    def recognize_letter(self, output_pattern, letter_patterns):
        min_distance = float('inf')
        os.makedirs('patterns', exist_ok=True)
        recognized_letter = None

        for letter, patterns in letter_patterns.items():
            for idx, pattern in enumerate(patterns):
                # Убедимся, что pattern имеет правильный тип
                pattern_img = pattern.reshape((int(np.sqrt(pattern.size)), -1)) if pattern.ndim == 1 else pattern

                # Нормализация в диапазон [0, 255] и приведение к uint8, если нужно
                if pattern_img.dtype != np.uint8:
                    pattern_img = np.clip(pattern_img, 0, 1) * 255
                    pattern_img = pattern_img.astype(np.uint8)

                # Сохраняем с уникальным именем
                cv2.imwrite(f'patterns/{letter}_{idx}.png', pattern_img)

                # Вычисление расстояния
                distance = euclidean(output_pattern.flatten(), pattern.flatten())
                if distance < min_distance:
                    min_distance = distance
                    recognized_letter = letter

        return recognized_letter
    def recognize_pattern(self, output_pattern, letter_patterns):
        min_distance = float('inf')
        patterni = output_pattern
        for letter, patterns in letter_patterns.items():
            for pattern in patterns:
                distance = euclidean(output_pattern, pattern)
                if distance < min_distance:
                    min_distance = distance
                    patterni = pattern
        return patterni

letter_images = {
    'A': ['data/data10/A0.png'],
    'B': ['data/data10/B0.png'],
    'M': ['data/data10/M0.png'],
    'Q': ['data/data10/Q0.png'],
    'X': ['data/data10/X0.png'],
    'Z': ['data/data10/Z0.png']
}
letter_patterns = {}
for letter, images in letter_images.items():
    patterns = [HopfieldNetwork.convert_image_to_pattern(img_path) for img_path in images]
    letter_patterns[letter] = patterns

num_neurons = 100 * 100
hopfield = HopfieldNetwork(num_neurons)

patterns = []
for letter, images in letter_images.items():
    for img_path in images:
        pattern = hopfield.convert_image_to_pattern(img_path)
        patterns.append(pattern)
hopfield.train(np.array(patterns))

test_image = 'data/data10/B0.png'
noisy_pattern = hopfield.convert_image_to_pattern(test_image)

noise_level = 0.2
num_noise = int(noise_level * len(noisy_pattern))
noisy_indices = np.random.choice(len(noisy_pattern), num_noise, replace=False)
noisy_pattern[noisy_indices] = -noisy_pattern[noisy_indices]
output_pattern = hopfield.predict(noisy_pattern,async_update=False)

hopfield.display_pattern(noisy_pattern)
hopfield.display_pattern(output_pattern)

recognized_letter = hopfield.recognize_letter(output_pattern, letter_patterns)
hopfield.display_pattern(hopfield. recognize_pattern(output_pattern, letter_patterns))
print(f"Recognized letter: {recognized_letter}")

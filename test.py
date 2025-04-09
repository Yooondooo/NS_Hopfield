import json
import numpy as np
import matplotlib.pyplot as plt
from model.NS import HopfieldNetwork
import tensorflow as tf


def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        return json.load(file)


# Загрузка данных
X_test = np.load('data/data/X.npy') / 255.0  # Нормализуем данные
y_test = np.load('data/data/y.npy')

# Загружаем конфигурацию
config = load_config()
path = config['path']

# Восстанавливаем модель Хопфилда
input_dim = X_test.shape[1] * X_test.shape[2]  # 28x28 пикселей
num_classes = 78  # Должно совпадать с обучением

model = HopfieldNetwork(input_shape=(28, 28, 1), num_classes=num_classes)
model.build((None, input_dim))
model.load_weights(f'{path}/hopfield_model_weights.weights.h5')

# Тестирование модели
correct_predictions = 0
for i in range(len(X_test)):
    input_pattern = np.sign(X_test[i].flatten())
    output_pattern = np.sign(model.W.numpy() @ input_pattern)

    # Определяем класс по наибольшему сходству
    predicted_class = np.argmax(np.dot(model.W.numpy(), input_pattern))
    actual_class = y_test[i]

    if predicted_class == actual_class:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Визуализация результата
plt.bar(['Correct', 'Incorrect'], [correct_predictions, len(X_test) - correct_predictions], color=['green', 'red'])
plt.xlabel('Prediction Type')
plt.ylabel('Count')
plt.title('Hopfield Network Test Results')
plt.savefig(f'{path}/test_results.png')
plt.show()
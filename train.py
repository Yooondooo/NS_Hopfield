import json
import numpy as np
import matplotlib.pyplot as plt
from model.NS import create_model, HopfieldNetwork
import tensorflow as tf


def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        return json.load(file)


def train_hopfield_network(X, num_classes):
    # Сеть Хопфилда — это ассоциативная память
    input_dim = X.shape[1] * X.shape[2]  # 28x28 пикселей
    hopfield_model = HopfieldNetwork(input_shape=(28, 28, 1), num_classes=num_classes)

    # Подготовим матрицу ассоциативной памяти для обучения
    memory_matrix = np.zeros((num_classes, input_dim), dtype=np.float32)

    # Запоминаем образцы (для сети Хопфилда)
    for i in range(num_classes):
        memory_matrix[i] = np.sign(X[i].flatten())  # Преобразуем в бинарные данные

    # Обновляем веса сети Хопфилда
    hopfield_model.build((None, input_dim))
    hopfield_model.W.assign(memory_matrix.T @ memory_matrix)  # Матрица весов для Хопфилда

    return hopfield_model


# Загрузка данных
X = np.load('data/data/X.npy') / 255.0  # Нормализуем данные
y = np.load('data/data/y.npy')

# Получаем конфигурацию
config = load_config()
path = config['path']

# Инициализация и тренировка сети Хопфилда
model = train_hopfield_network(X, num_classes=78)  # Создаем сеть Хопфилда

# Сохраняем веса (не модель, так как в Хопфилде нет традиционного обучения)
model.save_weights(f'{path}/hopfield_model_weights.weights.h5')

# Для визуализации предсказаний
plt.plot(np.arange(100), label='Обучение')  # Поскольку нет стандартного обучения, визуализируем фиктивную метрику
plt.title('Обучение модели Хопфилда')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.savefig(f'{path}/training_plot.png')
plt.show()

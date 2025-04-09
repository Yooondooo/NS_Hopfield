import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.activations import softmax
import json

def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        return json.load(file)

class HopfieldNetwork(Model):
    def __init__(self, input_shape, num_classes, **kwargs):
        super(HopfieldNetwork, self).__init__(**kwargs)
        self.input_shape_ = input_shape
        self.num_classes = num_classes

        config = load_config()
        self.input_dim = np.prod(input_shape)
        self.num_classes = num_classes

    def build(self, input_shape):
        # Создаем матрицу связей для Хопфилда
        self.W = self.add_weight(shape=(self.input_dim, self.input_dim), initializer="zeros", trainable=False)
        super(HopfieldNetwork, self).build(input_shape)

    def call(self, inputs):
        inputs_flat = tf.reshape(inputs, (-1, self.input_dim))
        # Преобразуем вход в бинарный вид (для Хопфилда)
        inputs_flat = tf.sign(inputs_flat)
        outputs = tf.matmul(inputs_flat, self.W)
        outputs = tf.sign(outputs)  # Деактивация через знаки
        return outputs

    def get_config(self):
        return {"input_shape": self.input_shape_, "num_classes": self.num_classes}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_model(input_shape, num_classes):
    model = HopfieldNetwork(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
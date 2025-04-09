import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import string
from model.NS import HopfieldNetwork

def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        return json.load(file)

LETTERS = string.ascii_uppercase[:26]

X = np.load('data/data_test/X.npy') / 255.0
y = np.load('data/data_test/y.npy')
X = X.reshape((-1, 28, 28, 1))
config = load_config()
path = config['path']
model = tf.keras.models.load_model(f'{path}/saved_model.keras', custom_objects = {"MyModel": HopfieldNetwork})
y_pred = np.argmax(model.predict(X), axis=1)
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=LETTERS, yticklabels=LETTERS,cbar=False)
plt.title('Матрица предсказания')
plt.xlabel('Предсказано')
plt.ylabel('Правильный ответ')
plt.savefig(f'{path}/confusion_matrix.png')
plt.show()

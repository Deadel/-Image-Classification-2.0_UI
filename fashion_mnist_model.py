import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Funkcja do trenowania modelu
def train_and_save_model():
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    model.save('static/fashion_mnist_model.h5')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nDokładność na danych testowych:', test_acc)

# Funkcja do ładowania modelu
def load_model():
    return tf.keras.models.load_model('static/fashion_mnist_model.h5')

# Funkcja do predykcji
def predict_image(model, img):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    img = np.expand_dims(img, axis=0)
    predictions = probability_model.predict(img)
    return predictions

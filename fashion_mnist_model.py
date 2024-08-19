import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import os

def train_and_save_model():
    # Sprawdź i utwórz folder, jeśli nie istnieje
    if not os.path.exists('static'):
        os.makedirs('static')

    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    
    # Dodaj wymiar kanału do obrazów
    train_images = train_images[..., np.newaxis] / 255.0
    test_images = test_images[..., np.newaxis] / 255.0
    
    # Definiowanie modelu
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

    # Trening modelu
    try:
        model.fit(train_images, train_labels, epochs=10)
    except Exception as e:
        print(f'Error during training: {e}')
        return

    # Zapisz model
    try:
        model.save('static/fashion_mnist_model.h5')
        print('Model saved as fashion_mnist_model.h5')
    except Exception as e:
        print(f'Error saving model: {e}')
        return

    # Ocena modelu
    try:
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print('\nDokładność na danych testowych:', test_acc)
    except Exception as e:
        print(f'Error evaluating model: {e}')

if __name__ == "__main__":
    train_and_save_model()

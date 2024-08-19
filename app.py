from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from fashion_mnist_model import load_model, class_names, predict_image

# Inicjalizacja Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Wczytanie modelu
model = load_model()

# Funkcja do przygotowania obrazu
def prepare_image(image_path):
    img = Image.open(image_path).convert('L')  # Konwertuj na skalę szarości
    img = img.resize((28, 28))  # Zmień rozmiar na 28x28 pikseli
    img = np.array(img) / 255.0  # Normalizacja
    img = img.reshape(28, 28, 1)  # Przekształć na odpowiedni kształt dla modelu
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Przygotowanie obrazu i predykcja
            img = prepare_image(file_path)
            predictions = predict_image(model, img)
            predicted_class = class_names[np.argmax(predictions)]

            return render_template('index.html', image=file.filename, prediction=predicted_class)
    
    return render_template('index.html', image=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

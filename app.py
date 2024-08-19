import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from PIL import Image
import io

app = Flask(__name__)

def load_model():
    try:
        model = tf.keras.models.load_model('static/fashion_mnist_model.h5')
        return model
    except Exception as e:
        print(f'Error loading model: {e}')
        return None

model = load_model()

def prepare_image(image):
    try:
        image = image.convert('L')  # Convert image to grayscale
        image = image.resize((28, 28))  # Resize image to 28x28 pixels
        image = np.array(image)  # Convert image to numpy array
        image = image / 255.0  # Normalize image
        image = image[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        return image
    except Exception as e:
        print(f'Error preparing image: {e}')
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                img = Image.open(io.BytesIO(file.read()))
                img = prepare_image(img)
                if img is not None:
                    predictions = model.predict(img)
                    predicted_class = np.argmax(predictions[0])
                    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                    predicted_label = class_names[predicted_class]
                    return render_template('index.html', prediction=predicted_label)
                else:
                    return render_template('index.html', prediction='Error processing image')
            except Exception as e:
                print(f'Error during prediction: {e}')
                return render_template('index.html', prediction='Error during prediction')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

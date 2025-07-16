from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/fabric_pattern_classifier.keras')  # Replace with your model path
classes = ['class1', 'class2', 'class3']  # Replace with your actual class names

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(224, 224))  # adjust size as per model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    return render_template('result.html', image=file.filename, label=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
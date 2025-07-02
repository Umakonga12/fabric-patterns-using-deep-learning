from flask import Flask, render_template, request
import tensorflow._api.v2 as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
app = Flask(__name__)
# Load model
model = load_model("Fabric_Model_cnn.h5")
# Fabric pattern labels (modify as needed)
labels = ['chequered', 'paisley', 'plain', 'polka-dotted', 'striped', 'zigzagged']

def get_model_prediction(image_path):
    img = load_img(image_path, target_size=(255, 255))
    img = img_to_array(img)
    x = np.expand_dims(img, axis=0)
    predictions = model.predict(x, verbose=0)
    return labels[predictions.argmax()]
@app.route('/')
def home():
    return render_template("home.html")
@app.route('/predict_page')
def predict():
    return render_template("predictpage.html")

@app.route('/predict', methods=['POST'])
def prediction():
    img = request.files['ump_image']
    upload_dir = 'static/assets/uploads/'
    os.makedirs(upload_dir, exist_ok=True)
    img_path = upload_dir + img.filename
    img.save(img_path)
    pred = get_model_prediction(img_path)
    return render_template("predictpage.html", img_path=img_path, prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)

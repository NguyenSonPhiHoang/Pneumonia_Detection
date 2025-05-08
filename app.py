import os
from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
from model import build_model
from data_loader import preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    try:
        model = tf.keras.models.load_model('models/cnn_model.keras')
        print("Model loaded successfully")
    except:
        print("Could not load model, building a new one")
        model = build_model()
    return model

def load_threshold():
    threshold_file = 'outputs/result/optimal_threshold.txt'
    if os.path.exists(threshold_file):
        with open(threshold_file, 'r') as f:
            return float(f.read())
    return 0.3  # Đồng bộ ngưỡng với evaluate.py

model = load_model()
threshold = load_threshold()
print(f"Using threshold: {threshold}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            img = preprocess_image(file_path)
            prediction = model.predict(np.expand_dims(img, axis=0))
            probability = float(prediction[0][0])
            print(f"Raw probability: {probability}")

            if probability >= threshold:
                result = "PNEUMONIA"
                confidence = probability * 100
            else:
                result = "NORMAL"
                confidence = (1 - probability) * 100

            return jsonify({
                'result': result,
                'confidence': f"{confidence:.2f}%",
                'image_path': file_path
            })
        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
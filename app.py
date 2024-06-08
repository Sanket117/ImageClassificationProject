from flask import Flask, request, jsonify, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
model = load_model('../models/cifar10_model.h5')

# Mapping of class indices to class labels
class_labels = {
    0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 
    4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'
}

@app.route('/')
def home():
    return render_template('index.html', img=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    try:
        img = Image.open(io.BytesIO(file.read()))
    except:
        return render_template('index.html', error_message="Invalid image file. Please upload a valid image.")

    img = img.resize((32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    predicted_label = class_labels[class_idx]
    
    return render_template('result.html', img=img, predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)

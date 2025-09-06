from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from rembg import remove
import io
import base64

# Flask app
app = Flask(__name__)

# Load models
name_model = tf.keras.models.load_model("models/name.h5", compile=False)
shape_model = tf.keras.models.load_model("models/gem_shape_model_v3.h5", compile=False)

# Constants
IMG_SIZE = (128, 128)
GEM_CLASSES = ['Blue_sapphire_stone', 'Diomond', 'Emerald_stone', 'Ruby_stone']
GEM_SHAPES = ['Marquise_shape', 'Oval_shape', 'Cusion_shape', 'Pear_shape']

# Image processors
def preprocess_color(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_grayscale(image):
    image = image.convert("L").resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return np.expand_dims(arr, axis=-1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Load and remove background
        image = Image.open(file.stream).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        output = remove(buf.getvalue())
        clean_img = Image.open(io.BytesIO(output)).convert("RGBA")

        # Resize to 128x128
        resized_img = clean_img.resize(IMG_SIZE)

        # Convert to base64
        out_buf = io.BytesIO()
        resized_img.save(out_buf, format='PNG')
        base64_img = base64.b64encode(out_buf.getvalue()).decode("utf-8")

        return jsonify({
            "image": f"data:image/png;base64,{base64_img}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Load resized + cleaned image
        clean_img = Image.open(file.stream).convert("RGBA")

        # Predict name
        processed_name = preprocess_color(clean_img)
        name_pred = name_model.predict(processed_name)
        gem_name = GEM_CLASSES[np.argmax(name_pred[0])]

        # Predict shape
        processed_shape = preprocess_grayscale(clean_img)
        shape_pred = shape_model.predict(processed_shape)
        gem_shape = GEM_SHAPES[np.argmax(shape_pred[0])]

        return jsonify({
            "gem_name": gem_name,
            "gem_shape": gem_shape
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

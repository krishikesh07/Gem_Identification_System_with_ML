import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from rembg import remove
import io
import base64
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM

# Flask app
app = Flask(__name__)

# Load models
name_model = keras.models.load_model("models/name.h5", compile=False)
shape_model = keras.models.load_model("models/shape.h5", compile=False)

# Load LLM (assuming it's a Hugging Face compatible model downloaded locally in models/llm_model)
llm_path = "models/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(llm_path)
llm_model = AutoModelForCausalLM.from_pretrained(llm_path)

# Load embedding model (local, assuming pre-downloaded)
embedder = SentenceTransformer('models/all-MiniLM-L6-v2')

# Constants
IMG_SIZE = (128, 128)
GEM_CLASSES = ['Blue_sapphire_stone', 'Diomond', 'Emerald_stone', 'Ruby_stone']
GEM_SHAPES = ['Marquise_shape', 'Oval_shape', 'Cusion_shape', 'Pear_shape']
DATABASE_FOLDER = "database"
VECTOR_DB_FOLDER = "vector_dbs"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# Function to extract and split text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    # Simple splitter (chunks of ~500 characters)
    chunks = []
    for i in range(0, len(text), 500):
        chunks.append(text[i:i+500])
    return chunks

# Function to get or create embeddings and vector store (simple pickle-based, using list of embeddings)
def get_vector_store(gem_name):
    pdf_path = os.path.join(DATABASE_FOLDER, f"{gem_name}.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF for {gem_name} not found.")
    
    db_path = os.path.join(VECTOR_DB_FOLDER, f"{gem_name}_embeddings.pkl")
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            return pickle.load(f)  # Returns {'chunks': [...], 'embeddings': [...]}
    
    chunks = extract_text_from_pdf(pdf_path)
    embeddings = embedder.encode(chunks)
    db = {'chunks': chunks, 'embeddings': embeddings}
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)
    return db

def clean_and_resize_image(file_stream):
    """Remove background, resize, and return both PIL image and base64 string."""
    # Load original image
    image = Image.open(file_stream).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    # Remove background
    output = remove(buf.getvalue())
    clean_img = Image.open(io.BytesIO(output)).convert("RGBA")

    # Resize
    resized_img = clean_img.resize(IMG_SIZE)

    # Convert to base64 (for frontend preview)
    out_buf = io.BytesIO()
    resized_img.save(out_buf, format='PNG')
    base64_img = base64.b64encode(out_buf.getvalue()).decode("utf-8")

    return resized_img, base64_img


# Simple retriever using cosine similarity
def retrieve_docs(vector_store, query, top_k=3):
    query_emb = embedder.encode([query])
    similarities = cosine_similarity(query_emb, vector_store['embeddings'])[0]
    top_indices = np.argsort(similarities)[-top_k:]
    return [vector_store['chunks'][i] for i in top_indices]

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
        resized_img, base64_img = clean_and_resize_image(file.stream)
        return jsonify({"image": f"data:image/png;base64,{base64_img}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        clean_img, _ = clean_and_resize_image(file.stream)

        # Predict name
        processed_name = preprocess_color(clean_img)
        name_pred = name_model.predict(processed_name)
        gem_name = GEM_CLASSES[np.argmax(name_pred[0])]

        # Predict shape
        processed_shape = preprocess_grayscale(clean_img)
        shape_pred = shape_model.predict(processed_shape)
        gem_shape = GEM_SHAPES[np.argmax(shape_pred[0])]

        # Ensure vector store is preloaded
        get_vector_store(gem_name)

        return jsonify({
            "gem_name": gem_name,
            "gem_shape": gem_shape
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get('message')
    gem_name = data.get('gem_name')
    if not message or not gem_name:
        return jsonify({"reply": "Invalid request."}), 400

    try:
        # Check if question is related to the gem
        prompt_check = f"Is the following question related to {gem_name}? Question: {message}\nAnswer only 'yes' or 'no'."
        inputs = tokenizer(prompt_check, return_tensors="pt")
        outputs = llm_model.generate(**inputs, max_new_tokens=10)
        check_response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        if 'yes' not in check_response:
            return jsonify({"reply": "Please ask questions related to the predicted gem only."})

        # Retrieve relevant docs
        vector_store = get_vector_store(gem_name)
        docs = retrieve_docs(vector_store, message)
        context = "\n".join(docs)

        # Generate response
        prompt = f"Based on the following information about {gem_name}, answer the question: {message}\nContext: {context}\nAnswer concisely:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = llm_model.generate(**inputs, max_new_tokens=200)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer concisely:")[-1].strip()

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
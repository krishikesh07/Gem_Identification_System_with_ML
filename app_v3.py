import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from PIL import Image
from rembg import remove
import io
import base64
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import pickle
from llama_cpp import Llama
from flask_session import Session  # For server-side session storage

# ------------------ Flask setup ------------------ #
app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)  # store session on server side

# ------------------ Load models ------------------ #
name_model = keras.models.load_model("models/name.h5", compile=False)
shape_model = keras.models.load_model("models/shape.h5", compile=False)

# GGUF LLaMA model
llm_model = Llama(model_path="models/llama-3.2-1b-instruct-q4_k_m.gguf", n_ctx=2048, n_threads=4)

# Sentence embedding model
embedder = SentenceTransformer('models/all-MiniLM-L6-v2')

# ------------------ Constants ------------------ #
IMG_SIZE = (128, 128)
GEM_CLASSES = ['Blue_sapphire', 'Diomond', 'Emerald', 'Ruby']
GEM_SHAPES = [ 'Cusion_shape', 'Oval_shape','Marquise_shape', 'Pear_shape']
DATABASE_FOLDER = "database"
VECTOR_DB_FOLDER = "vector_dbs"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# ------------------ Helper Functions ------------------ #
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() + "\n" for page in reader.pages)
    return [text[i:i+500] for i in range(0, len(text), 500)]

def get_vector_store(gem_name):
    pdf_path = os.path.join(DATABASE_FOLDER, f"{gem_name}.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF for {gem_name} not found.")
    
    db_path = os.path.join(VECTOR_DB_FOLDER, f"{gem_name}_embeddings.pkl")
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            return pickle.load(f)
    
    chunks = extract_text_from_pdf(pdf_path)
    embeddings = embedder.encode(chunks)
    db = {'chunks': chunks, 'embeddings': embeddings}
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)
    return db

def clean_and_resize_image(file_stream):
    image = Image.open(file_stream).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    # Remove background
    output = remove(buf.getvalue())
    clean_img = Image.open(io.BytesIO(output)).convert("RGBA")

    # Resize
    resized_img = clean_img.resize(IMG_SIZE)

    # Convert to base64
    out_buf = io.BytesIO()
    resized_img.save(out_buf, format='PNG')
    base64_img = base64.b64encode(out_buf.getvalue()).decode("utf-8")
    return resized_img, base64_img

def preprocess_color(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_grayscale(image):
    image = image.convert("L").resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return np.expand_dims(arr, axis=-1)

def retrieve_docs(vector_store, query, top_k=3):
    query_emb = embedder.encode([query])
    similarities = cosine_similarity(query_emb, vector_store['embeddings'])[0]
    top_indices = np.argsort(similarities)[-top_k:]
    return [vector_store['chunks'][i] for i in top_indices]

# ------------------ Routes ------------------ #
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        _, base64_img = clean_and_resize_image(file.stream)
        return jsonify({"image": f"data:image/png;base64,{base64_img}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get('file')
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

        # Preload vector store
        get_vector_store(gem_name)

        # Initialize conversation memory in session
        session['conversation'] = []
        session['current_gem'] = gem_name

        return jsonify({"gem_name": gem_name, "gem_shape": gem_shape})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    gem_name = session.get('current_gem', None)
    if not user_message or not gem_name:
        return jsonify({"reply": "Please upload and predict a gem first."}), 400

    # Load conversation history
    conversation = session.get('conversation', [])

    # Add user message to history
    conversation.append(f"User: {user_message}")

    # Build prompt with context
    context_prompt = f"The following is a conversation about {gem_name}:\n"
    context_prompt += "\n".join(conversation)
    context_prompt += f"\nBot (answer concisely):"

    # Retrieve relevant docs
    try:
        vector_store = get_vector_store(gem_name)
        docs = retrieve_docs(vector_store, user_message)
        doc_context = "\n".join(docs)
        full_prompt = f"{context_prompt}\nContext:\n{doc_context}\nAnswer concisely:"
        
        # Generate response
        output = llm_model(full_prompt, max_tokens=200)
        bot_reply = output['choices'][0]['text'].strip()

        # Update conversation memory
        conversation.append(f"Bot: {bot_reply}")
        session['conversation'] = conversation

        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

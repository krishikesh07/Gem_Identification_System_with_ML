# Gem_Identification_System_with_ML
This project is an AI Gem Identification System with an integrated chatbot assistant.
It combines custom Convolutional Neural Networks (CNNs) for gemstone classification and a Flask-based chatbot interface to identify gems, analyze their shapes and answer user queries.

features

Gem Identification with CNNs
name.h5 → Classifies gemstone type (e.g., Ruby, Emerald, Sapphire, Diamond).
shape.h5 → Classifies gemstone shape.

Knowledge Base with Embeddings

Pre-computed embeddings from gem-related documents (.pkl files).
Uses all-MiniLM-L6-v2 SentenceTransformer for semantic similarity search.

Chatbot Assistant

Powered by LLaMA model (llama.cpp) for natural language responses.
Retrieves gem knowledge from PDF-based embeddings and answers user questions.

Image Preprocessing

Removes image backgrounds using rembg.
Accepts user-uploaded gemstone images for analysis.

Web Application

Built with Flask (Python web framework).
Includes HTML templates (index.html) and CSS (style.css).


Project Structure

UI/
│── app.py                 # Main Flask app  
│── app_v3.py              # Extended version with chatbot integration  
│── app_miniLM-L6-V2.py    # Embedding + NLP model integration  
│
├── database/              # Reference documents
│   ├── Blue_sapphire.pdf
│   ├── Diamond.pdf
│   ├── Emerald.pdf
│   └── RUBY.pdf
│
├── models/                # Models used
│   ├── name.h5            # CNN model for gemstone classification
│   ├── shape.h5           # CNN model for gemstone shape classification
│   ├── all-MiniLM-L6-v2   # Pre-trained sentence transformer (embeddings)
│   └── llama-3.2-1b-instruct-q4_k_m.gguf   # LLaMA model for chatbot
│
├── vector_dbs/            # Pre-computed embeddings
│   ├── Blue_sapphire_embeddings.pkl
│   ├── Emerald_embeddings.pkl
│   └── Ruby_embeddings.pkl
│
├── static/                # Frontend assets
│   └── style.css
│
├── templates/             # HTML templates
│   └── index.html
│
└── flask_session/         # Session storage

Installation

1. Clone the Repository

git clone https://github.com/yourusername/gem-identification-chatbot.git
cd gem-identification-chatbot/UI

2. Run the App

python app_v3.py

Then open:
=http://127.0.0.1:5000/

Requirements

Flask
flask-session
keras
tensorflow
numpy
Pillow
rembg
sentence-transformers
scikit-learn
pypdf
llama-cpp-python


Models Used

Custom CNN Models (Keras/TensorFlow)
name.h5 → Gemstone classification.
shape.h5 → Gemstone shape detection.
https://drive.google.com/file/d/1ewKwi8HsfW9iZaUQ3mk_J2k_3UnjcnfN/view?usp=drive_link - Modelels have been upload to the follwing google drive 

Pre-trained Embedding Model
all-MiniLM-L6-v2 (Sentence Transformers) for semantic similarity search.

Chatbot Model
llama.cpp model (llama-3.2-1b-instruct-q4_k_m.gguf) for conversational responses.

Usage

Upload a gemstone image → CNN predicts gem type & shape.
Ask questions about gems → Chatbot retrieves answers from embedding database.
Get both AI-powered classification + knowledge-based chatbot replies.

Example Queries

"In which country can found this typeof stone.
"whats the price range of the stone"
"what is the chemical composition of the stone"

Future Improvements

Train CNNs with a larger gem dataset for better accuracy.
Enhance chatbot with multimodal input (image + text queries).
Deploy on cloud for real-time API access.

Author
Developed by S.Krishikesh 
2025


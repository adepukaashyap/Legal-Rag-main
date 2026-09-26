import os
import numpy as np
import faiss

from flask import Flask, request, jsonify
from flask_cors import CORS

from groq import Groq
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# Environment variables
# -----------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

if not MONGO_URI:
    raise ValueError("MONGO_URI is not set")


# -----------------------------
# Groq
# -----------------------------

groq_client = Groq(api_key=GROQ_API_KEY)


# -----------------------------
# MongoDB
# -----------------------------

mongo_client = MongoClient(MONGO_URI)

db = mongo_client["legal_rag"]  # CHANGE THIS if your Legal RAG uses another database name

users_collection = db["users"]

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CSV_PATH = os.path.join(os.path.dirname(__file__), "constitution.csv")

df = pd.read_csv(CSV_PATH)

documents = []

for _, row in df.iterrows():
    documents.append(
        Document(
            page_content=str(row["article_desc"]),
            metadata={
                "article_id": str(row["article_id"])
            }
        )
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128
)

documents = text_splitter.split_documents(documents)

# -----------------------------
# Create embeddings
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

document_texts = [doc.page_content for doc in documents]

document_embeddings = embedding_model.embed_documents(document_texts)

embeddings_np = np.array(document_embeddings).astype("float32")

# Normalize embeddings
faiss.normalize_L2(embeddings_np)


# -----------------------------
# Create FAISS index
# -----------------------------

index = faiss.IndexFlatIP(embeddings_np.shape[1])

index.add(embeddings_np)

print("FAISS index created successfully")
print("Documents:", len(documents))

# -----------------------------
# RAG Query Function
# -----------------------------

def process_rag_query(user_query):

    # Create query embedding
    query_embedding = embedding_model.embed_query(user_query)

    query_embedding_np = np.array(
        [query_embedding]
    ).astype("float32")

    faiss.normalize_L2(query_embedding_np)

    # Search FAISS
    D, I = index.search(
        query_embedding_np,
        k=5
    )

    # Collect relevant chunks
    relevant_chunks = []

    for score, idx in zip(D[0], I[0]):

        if score >= 0.4:
            doc = documents[idx]
            relevant_chunks.append(doc.page_content)

    # No relevant legal context
    if not relevant_chunks:
        return {
            "answer": "No, this does not come under legal queries."
        }

    # Combine context
    context = "\n".join(relevant_chunks)

    # System prompt
    system_prompt = (
        "You are a legal assistant. Answer the user's question based only "
        "on the following context from the Constitution of India. "
        "Do not add introductory or filler phrases. "
        "Respond only with the direct legal answer, in a formal and concise manner.\n\n"
        f"Context:\n{context}"
    )

    # Groq Qwen
    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        temperature=0.4,
        max_completion_tokens=200,
        reasoning_effort="none"
    )

    generated_text = response.choices[0].message.content

    return {
        "answer": generated_text.strip()
    }
# -----------------------------
# Flask App
# -----------------------------

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)

# -----------------------------
# Generate API
# -----------------------------

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({
                "error": "Query is required"
            }), 400

        result = process_rag_query(query)

        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500
        
# -----------------------------
# Signup API
# -----------------------------

@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()

        name = data.get("name", "")
        email = data.get("email", "")
        password = data.get("password", "")

        if not email or not password:
            return jsonify({
                "error": "Email and password are required"
            }), 400

        existing_user = users_collection.find_one({
            "email": email
        })

        if existing_user:
            return jsonify({
                "error": "User already exists"
            }), 409

        hashed_password = generate_password_hash(password)

        users_collection.insert_one({
            "name": name,
            "email": email,
            "password": hashed_password
        })

        return jsonify({
            "message": "Signup successful"
        }), 201

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500
        
# -----------------------------
# Login API
# -----------------------------

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()

        email = data.get("email", "")
        password = data.get("password", "")

        if not email or not password:
            return jsonify({
                "error": "Email and password are required"
            }), 400

        user = users_collection.find_one({
            "email": email
        })

        if not user:
            return jsonify({
                "error": "Invalid email or password"
            }), 401

        if not check_password_hash(
            user["password"],
            password
        ):
            return jsonify({
                "error": "Invalid email or password"
            }), 401

        return jsonify({
            "message": "Login successful",
            "user": {
                "name": user.get("name", ""),
                "email": user["email"]
            }
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500
        
# -----------------------------
# Run Flask App
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
    
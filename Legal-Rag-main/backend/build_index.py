import os
import numpy as np
import faiss
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


BASE_DIR = os.path.dirname(__file__)

CSV_PATH = os.path.join(BASE_DIR, "constitution.csv")
INDEX_PATH = os.path.join(BASE_DIR, "constitution.index")
DOCUMENTS_PATH = os.path.join(BASE_DIR, "documents.npy")


# Load CSV
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


# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128
)

documents = text_splitter.split_documents(documents)

print("Documents:", len(documents))


# Load the SAME embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create embeddings
document_texts = [doc.page_content for doc in documents]

document_embeddings = embedding_model.embed_documents(
    document_texts
)

embeddings_np = np.array(
    document_embeddings
).astype("float32")


# Normalize
faiss.normalize_L2(embeddings_np)


# Create FAISS index
index = faiss.IndexFlatIP(
    embeddings_np.shape[1]
)

index.add(embeddings_np)


# Save FAISS index
faiss.write_index(
    index,
    INDEX_PATH
)


# Save document texts + metadata
document_data = np.array(
    [
        {
            "page_content": doc.page_content,
            "article_id": doc.metadata["article_id"]
        }
        for doc in documents
    ],
    dtype=object
)

np.save(
    DOCUMENTS_PATH,
    document_data,
    allow_pickle=True
)


print("FAISS index saved successfully")
print("Index:", INDEX_PATH)
print("Documents:", DOCUMENTS_PATH)
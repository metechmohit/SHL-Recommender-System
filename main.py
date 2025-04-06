import os
from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env (optional locally)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
client = OpenAI(api_key=api_key)

# --- Load data and build FAISS index ---
df = pd.read_csv("data/shl_with_embeddings.csv")
df["openai_embedding"] = df["openai_embedding"].apply(eval)
embedding_matrix = np.vstack(df["openai_embedding"].values).astype("float32")
faiss.normalize_L2(embedding_matrix)
index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

# --- Embedding helper ---
def get_openai_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

# --- API Endpoint ---
@app.get("/recommend")
def recommend_assessments(query: str = Query(...), top_k: int = 5):
    query_vec = get_openai_embedding(query)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)

    results = df.iloc[indices[0]][[
        "Assessment Name", "Assessment URL",
        "Remote Testing Support", "Adaptive/IRT Support",
        "Time", "Test Type Keys"
    ]].copy()

    return results.to_dict(orient="records")

@app.get("/")
def root():
    return {"message": "SHL Assessment API is running. Use /recommend endpoint."}


# --- For local run or Render start command ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

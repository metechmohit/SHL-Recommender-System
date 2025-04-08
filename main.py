import os
from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import faiss
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env (optional locally)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
client = OpenAI(api_key=api_key)

# --- Load data and build FAISS index ---
df = pd.read_csv("data/shl_with_embeddings.csv")
# Safely convert string representations of lists to actual Python lists
df["openai_embedding"] = df["openai_embedding"].apply(eval)
embedding_matrix = np.vstack(df["openai_embedding"].values).astype("float32")
faiss.normalize_L2(embedding_matrix)
index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

# --- Embedding helper ---
def get_openai_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    # Return a 2D numpy array as required by FAISS
    return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

# --- API Endpoint ---
@app.get("/recommend")
def recommend_assessments(query: str = Query(...), top_k: int = 5):
    # Get the query embedding and ensure it's 2D.
    query_vec = get_openai_embedding(query)
    faiss.normalize_L2(query_vec)
    
    # Search the FAISS index
    scores, indices = index.search(query_vec, top_k)
    
    # Convert indices to a list for safe pandas indexing.
    selected_indices = indices[0].tolist()
    
    # If there are no valid indices, return an empty response.
    if not selected_indices or all(idx == -1 for idx in selected_indices):
        return JSONResponse(content=[])
    
    # Extract the results from the DataFrame.
    results = df.iloc[selected_indices][[
        "Assessment Name", "Assessment URL",
        "Remote Testing Support", "Adaptive/IRT Support",
        "Time", "Test Type Keys"
    ]].copy()
    
    return JSONResponse(content=results.to_dict(orient="records"))

@app.get("/")
def root():
    return JSONResponse(content={"message": "SHL Assessment API is running. Use /recommend endpoint."})

# --- For local run or Render start command ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

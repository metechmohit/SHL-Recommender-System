import os
from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow all origins for endpoint testing in SHL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=api_key)

df = pd.read_csv("data/shl_with_embeddings.csv")
df["openai_embedding"] = df["openai_embedding"].apply(eval)
embedding_matrix = np.vstack(df["openai_embedding"].values).astype("float32")
faiss.normalize_L2(embedding_matrix)
index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

def get_openai_embedding(text, model="text-embedding-3-small"):
    # Pre-process the text by replacing newlines
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

# @app.get("/recommend")
# def recommend_assessments(query: str = Query(...), top_k: int = 5):
#     query_vec = get_openai_embedding(query)
#     faiss.normalize_L2(query_vec)
#     scores, indices = index.search(query_vec, top_k)
    
#     results = df.iloc[indices[0]][[
#         "Assessment Name", "Assessment URL",
#         "Remote Testing Support", "Adaptive/IRT Support",
#         "Time", "Test Type Keys"
#     ]].copy()

#     return results.to_dict(orient="records")

@app.get("/recommend")
def recommend_assessments(
    query: str = Query(...),
    min_score: float = Query(0.7, ge=0.0, le=1.0)
):
    query_vec = get_openai_embedding(query)
    faiss.normalize_L2(query_vec)
    
    # Search top 20 for buffer, we’ll filter manually
    scores, indices = index.search(query_vec, 20)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= min_score:
            item = df.iloc[idx][[
                "Assessment Name", "Assessment URL",
                "Remote Testing Support", "Adaptive/IRT Support",
                "Time", "Test Type Keys"
            ]].to_dict()
            item["Score"] = round(float(score), 3)
            results.append(item)

    results = results[:10]

    if not results:
        raise HTTPException(status_code=404, detail="No relevant assessments found. Try broadening your query.")

    return results



@app.get("/")
def root():
    return {"message": "SHL Assessment API is running. Use /recommend endpoint."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

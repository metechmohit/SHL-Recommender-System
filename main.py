import os
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=api_key)

# Load embeddings
df = pd.read_csv("data/shl_with_embeddings.csv")
df["openai_embedding"] = df["openai_embedding"].apply(eval)
embedding_matrix = np.vstack(df["openai_embedding"].values).astype("float32")
faiss.normalize_L2(embedding_matrix)
index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Helper to generate embedding
def get_openai_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

# Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Request model for /recommend
class QueryRequest(BaseModel):
    query: str

#  Response model
class AssessmentResponse(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: list[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: list[AssessmentResponse]

def parse_duration(duration_str):
    match = re.search(r"\d+", str(duration_str)) #to extract only integer from df["Time"]
    return int(match.group()) if match else 0
    
# recommend endpoint (for testing and frontend)
@app.post("/recommend")
def recommend_assessments(
    payload: QueryRequest,
    frontend: bool = Query(False, description="Set true to return frontend-friendly format")
):
    query = payload.query
    query_vec = get_openai_embedding(query)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, 20)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if score >= 0.35:
            row = df.iloc[idx]

            if frontend:
                results.append({
                    "Assessment Name": row["Assessment Name"],
                    "Assessment URL": row["Assessment URL"],
                    "Description": row["Description"].split("|")[0].strip(),
                    "Time": row["Time"],  
                    "Remote Testing Support": row["Remote Testing Support"],
                    "Adaptive/IRT Support": row["Adaptive/IRT Support"],
                    "Test Type Keys": row["Test Type Keys"]
                })
            else:
                results.append({
                    "url": row["Assessment URL"],
                    "adaptive_support": row["Adaptive/IRT Support"],
                    "description": row["Description"].split("|")[0].strip(),
                    "duration": parse_duration(row["Time"]),  #for int only
                    "remote_support": row["Remote Testing Support"],
                    "test_type": [t.strip() for t in row["Test Type Keys"].split(",")]
                })

    if not results:
        raise HTTPException(status_code=404, detail="No relevant assessments found.")

    if frontend:
        return results[:10]
    else:
        return {"recommended_assessments": results[:10]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key)
df = pd.read_csv("data/shl_products_catalog.csv")  #scraped product_catalog

# Create text for embedding generation of correspondiong assessments
df["embedding_text"] = (
    df["Assessment Name"] + ". " +
    df["Description"] + ". " +
    "Test Type: " + df["Test Type Keys"] + ". " +
    "Duration: " + df["Time"] + ". " +
    "Remote Testing: " + df["Remote Testing Support"] + ". " +
    "Adaptive Support: " + df["Adaptive/IRT Support"]
)

# Generate embeddings
def get_openai_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

df["openai_embedding"] = df["embedding_text"].apply(get_openai_embedding)

df.to_csv("data/shl_with_embeddings.csv", index=False)

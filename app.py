import streamlit as st
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI

# --- Load data and FAISS index ---
@st.cache_resource
def load_data():
    df = pd.read_csv("data/shl_with_embeddings.csv")
    
    df["openai_embedding"] = df["openai_embedding"].apply(eval)  # Convert stringified lists to actual lists
    embedding_matrix = np.vstack(df["openai_embedding"].values).astype("float32")
    faiss.normalize_L2(embedding_matrix)
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    return df, index

df, index = load_data()

# --- OpenAI Setup ---

OPENAI_API_KEY = st.secrets["key"]
client = OpenAI(api_key=OPENAI_API_KEY)

def get_openai_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

# --- Streamlit UI ---
st.title("🔍 SHL Assessment Recommender")

query = st.text_input("Enter a job description or test requirement:")

if st.button("Find Assessments") and query:
    query_vec = get_openai_embedding(query)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, 5)

    top_results = df.iloc[indices[0]][[
        "Assessment Name", 
        "Assessment URL", 
        "Remote Testing Support", 
        "Adaptive/IRT Support", 
        "Time", 
        "Test Type Keys"
    ]].copy()

    for i, row in top_results.iterrows():
        st.markdown(f"### [{row['Assessment Name']}]({row['Assessment URL']})")
        st.write(f"- **Remote Testing**: {row['Remote Testing Support']}")
        st.write(f"- **Adaptive Support**: {row['Adaptive/IRT Support']}")
        st.write(f"- **Duration**: {row['Time']}")
        st.write(f"- **Test Type**: {row['Test Type Keys']}")
        st.markdown("---")

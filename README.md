# SHL Assessment Recommendation System

A semantic search-powered recommendation engine for SHL assessments. Users can enter natural language queries (like job descriptions), and the system returns relevant assessments from SHL’s catalog based on OpenAI embeddings and FAISS vector search.

---

##  Live Demo Links

- **Streamlit App** (UI): [ Try on Streamlit](https://shl-assessments-recommender.streamlit.app/)
- **FastAPI Backend** (API): [ Try on Render](https://shl-recommender-system.onrender.com/docs)

---

##  Features

-  Real-time search using OpenAI’s `text-embedding-3-small`
-  Instant top-k assessment recommendations via FAISS
-  API-first architecture with FastAPI
-  Streamlit-based UI for demo/testing
-  Publicly deployed with Render (API) & Streamlit Cloud (frontend)

---

##  Tech Stack

| Layer            | Tools Used                           |
|------------------|----------------------------------------|
| Scraping         | Selenium, BeautifulSoup                |
| Data Handling    | Pandas, NumPy                          |
| Embeddings       | OpenAI (`text-embedding-3-small`)      |
| Vector Search    | FAISS (`faiss-cpu`)                    |
| API Server       | FastAPI, Uvicorn                       |
| Frontend         | Streamlit                              |
| Deployment       | Render (API), Streamlit Cloud (UI)     |

---

## ⚙ Setup Instructions (Local)

### 1. Clone the repository

```bash
git clone https://github.com/metechmohit/SHL-Recommender-System.git
cd SHL-Recommender-System
pip install -r requirements.txt
```
## Create a .env file with:
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxx
```
## Run FastAPI server
```bash
uvicorn main:app --reload
```
### Run Streamlit app
```bash
streamlit run app.py
```
### Example Query:
```bash
GET /recommend?query=aptitude test under 30 minutes

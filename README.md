# SHL Assessment Recommendation System

A semantic search-powered recommendation engine for SHL assessments. Users can enter natural language queries (like job descriptions), and the system returns relevant assessments from SHL’s catalog based on OpenAI embeddings and FAISS vector search.

---
## Code for Scraping/ Collection of SHL product catalog with Selenum and bs4 is in [catalog_scraping.py](https://metechmohit.github.io/shl-recommendation-frontened/)
## Code for generating embeddings of Assessments Description is in [preprocess_embeddings.py](https://metechmohit.github.io/shl-recommendation-frontened/)
## Code for Endpoint creation is in [main.py](https://metechmohit.github.io/shl-recommendation-frontened/)
## Code for streamlit app is in [app.py](https://metechmohit.github.io/shl-recommendation-frontened/)
---

##  Live Demo Links

- **Deployed_frontend** (UI): [ For frontend](https://metechmohit.github.io/shl-recommendation-frontened/)
- **Streamlit App** (UI): [ Try on Streamlit](https://shl-assessments-recommender.streamlit.app/) ## Needs to wake up
- **FastAPI Swagger** (API): https://shl-recommender-system.onrender.com/docs    (Pass query param as search requirement)
- **Endpoint_eg.** https://shl-recommender-system.onrender.com/recommend?query=Assessment%20for%20entry%20level%20managers
- Change the param  query

---

##  Features

-  Real-time search using OpenAI’s `text-embedding-3-small`
-  Instant top-k assessment recommendations via FAISS
-  API-first architecture with FastAPI
-  Publicly deployed endpoint with Render (API) & Streamlit or frotnend deployment

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

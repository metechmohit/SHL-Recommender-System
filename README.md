#  SHL Assessment Recommendation System

A semantic search-powered recommendation engine for SHL assessments. Users can enter natural language queries (e.g., job descriptions), and the system returns the most relevant individual assessments using OpenAI embeddings and FAISS similarity search — delivered via a FastAPI endpoint.

---

##  Approach (In Brief)

1. **Scraping**: Collected structured data from SHL’s product catalog using `Selenium` and `BeautifulSoup`.
2. **Embedding Text Construction**: Combined assessment name, description, type, and support details for contextual embedding.
3. **Embedding Generation**: Used OpenAI’s `text-embedding-3-small` model to generate vector representations.
4. **Similarity Search**: Indexed vectors with FAISS and performed cosine similarity search.
5. **API Creation**: Built `/recommend` endpoint with FastAPI that returns 1–10 relevant assessments based on a score threshold.
6. **Frontend**: Built both Streamlit and pure HTML/JS frontend for interaction with the API.
7. **Deployment**: Backend on Render, frontend on GitHub Pages and Streamlit Cloud.

---

## Code Overview

| File | Purpose |
|------|---------|
| [`catalog_scraping.py`](https://github.com/metechmohit/SHL-Recommender-System/blob/master/catalog_scraping.py) | Scrapes SHL product catalog across multiple pages |
| [`preprocess_embeddings.py`](https://github.com/metechmohit/SHL-Recommender-System/blob/master/preprocess_embeddings.py) | Constructs embedding text and generates OpenAI embeddings |
| [`main.py`](https://github.com/metechmohit/SHL-Recommender-System/blob/master/main.py) | FastAPI backend with FAISS search and `/recommend` endpoint |
| [`app.py`](https://github.com/metechmohit/SHL-Recommender-System/blob/master/app.py) | Optional Streamlit-based UI for testing the recommendation system |

---

## Live Demo Links

- **Frontend (HTML/CSS)**: [Try Frontend](https://metechmohit.github.io/shl-recommendation-frontened/)
- **Streamlit UI**: [Launch Streamlit](https://huggingface.co/spaces/mohitsingheng/SHL-Assessment-Recommender) *(click to wake up if idle)*
- **FastAPI Docs (Swagger UI)**: [View Swagger](https://shl-recommender-system.onrender.com/docs)
- **Sample API Endpoint**:  
  - Health Check Endpoint(GET): [https://shl-recommender-system.onrender.com/health](https://shl-recommender-system.onrender.com/health)
  - Assessment Recommendation Endpoint(POST): https://shl-recommender-system.onrender.com/recommend

> You can customize the `query` parameter for different use cases.

---

## Features

- Semantic vector search using OpenAI’s `text-embedding-3-small`
- FAISS-powered fast top-k assessment retrieval
- Relevance filtering based on score threshold (min 1, max 10 results)
- REST API architecture (JSON output)
- CORS-enabled and publicly accessible (for assessment testing)
- Deployed backend (Render) + two frontend options (Streamlit & Static)

---

## Tech Stack

| Layer         | Tools/Libraries                     |
|---------------|-------------------------------------|
| Scraping      | `Selenium`, `BeautifulSoup`         |
| Processing    | `pandas`, `numpy`                   |
| Embeddings    | `OpenAI API` (`text-embedding-3-small`) |
| Vector Search | `FAISS` (`faiss-cpu`)               |
| API Backend   | `FastAPI`, `Uvicorn`                |
| Frontend      | `Streamlit`, `HTML/CSS/JS`          |
| Deployment    | `Render`, `Huggingface`, `GitHub Pages` |

---

## Local Setup Instructions

### 1. Clone the Repository

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

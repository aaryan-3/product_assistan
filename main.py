import torch
import pandas as pd
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import re

# ---------- LOAD DATA ----------
df = pd.read_pickle("products.pkl")
product_embeddings = torch.load("product_embeddings.pt", map_location="cpu")
product_embeddings = F.normalize(product_embeddings, p=2, dim=1)

# Normalize rating & price
df["norm_rating"] = df["rating"] / 5.0
df["norm_price"] = 1 - (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# ---------- SIMPLE TEXT EMBEDDING (LIGHTWEIGHT) ----------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(df["description"] + " " + df["tags"])

def embed_query_light(query):
    vec = vectorizer.transform([query]).toarray()
    return torch.tensor(vec, dtype=torch.float32)

# ---------- INTENT HELPERS ----------
def extract_budget(q):
    q = q.lower()
    m = re.search(r'(under|below)\s*(\d+)', q)
    if m: return (0, int(m.group(2)))
    return None

def budget_score(row, budget):
    if not budget: return 0
    low, high = budget
    return 1 if row["price"] <= high else -0.5

def detect_interest(q):
    q = q.lower()
    if "music" in q: return "music"
    if "gaming" in q: return "gaming"
    if "beauty" in q or "skincare" in q: return "beauty"
    return None

def interest_score(row, interest):
    if not interest: return 0
    text = (row["description"] + " " + row["tags"]).lower()
    if interest == "music" and any(w in text for w in ["audio","headphone","speaker"]): return 1
    if interest == "gaming" and "gaming" in text: return 1
    if interest == "beauty" and any(w in text for w in ["skin","cosmetic","beauty"]): return 1
    return 0

# ---------- SEARCH ----------
def search_products(query, top_k=3):
    query_vec = embed_query_light(query)

    # cosine similarity using tfidf
    product_vecs = torch.tensor(vectorizer.transform(df["description"] + " " + df["tags"]).toarray())
    semantic_scores = F.cosine_similarity(query_vec, product_vecs)

    budget = extract_budget(query)
    interest = detect_interest(query)

    budget_scores = df.apply(lambda r: budget_score(r, budget), axis=1).values
    interest_scores = df.apply(lambda r: interest_score(r, interest), axis=1).values

    final_scores = (
        0.5 * semantic_scores.numpy() +
        0.2 * df["norm_rating"].values +
        0.2 * df["norm_price"].values +
        0.1 * budget_scores +
        0.1 * interest_scores
    )

    top_indices = final_scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

# ---------- API ----------
@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    results = search_products(request.query)

    response = []
    for _, row in results.iterrows():
        response.append({
            "product_name": row["product_name"],
            "price": float(row["price"]),
            "rating": float(row["rating"]),
            "category": row["category"]
        })

    return {"recommendations": response}

import torch
import pandas as pd
import torch.nn.functional as F
import re
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

# ---------- LOAD DATA ----------
df = pd.read_pickle("products.pkl")
product_embeddings = torch.load("product_embeddings.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

product_embeddings = F.normalize(product_embeddings, p=2, dim=1)

# Normalize rating & price
df["norm_rating"] = df["rating"] / 5.0
df["norm_price"] = 1 - (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# ---------- EMBEDDING ----------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def embed_query(query):
    encoded = tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**encoded)
    return F.normalize(mean_pooling(output, encoded["attention_mask"]), p=2, dim=1)

# ---------- INTENT HELPERS ----------
def extract_budget(q):
    q = q.lower()
    m = re.search(r'(under|below)\s*(\d+)', q)
    if m: return (0, int(m.group(2)))
    m = re.search(r'between\s*(\d+)\s*and\s*(\d+)', q)
    if m: return (int(m.group(1)), int(m.group(2)))
    return None

def budget_score(row, budget):
    if not budget: return 0
    low, high = budget
    if low <= row["price"] <= high: return 1
    if row["price"] > high: return -0.6
    return 0.2

def extract_gender(q):
    q = q.lower()
    if any(w in q for w in ["men","male","boy"]): return "men"
    if any(w in q for w in ["women","female","girl"]): return "women"
    return None

def gender_score(row, g):
    if not g: return 0
    return 1 if g in row["target_audience"].lower() else 0

def detect_interest(q):
    q = q.lower()
    if "music" in q: return "music"
    if "gaming" in q: return "gaming"
    if any(w in q for w in ["beauty","skincare","cosmetic"]): return "beauty"
    return None

def interest_score(row, interest):
    if not interest: return 0
    text = (row["description"] + " " + row["tags"]).lower()
    if interest == "music" and any(w in text for w in ["audio","headphone","speaker"]): return 1
    if interest == "gaming" and "gaming" in text: return 1
    if interest == "beauty" and any(w in text for w in ["skin","cosmetic","beauty"]): return 1
    return -0.3

def detect_category(query):
    q = query.lower()
    if any(w in q for w in ["skincare","beauty","cosmetic","skin"]): return "beauty"
    if any(w in q for w in ["shirt","jacket","jeans","shoes"]): return "fashion"
    if any(w in q for w in ["headphone","speaker","laptop","earbud"]): return "electronics"
    return None

def category_score(row, cat):
    if not cat: return 0
    return 1 if cat in row["category"].lower() else -0.5

# ---------- SEARCH ----------
def search_products(query, top_k=3):
    query_emb = embed_query(query)
    semantic_scores = torch.matmul(query_emb, product_embeddings.T).squeeze(0).cpu().numpy()

    budget = extract_budget(query)
    gender = extract_gender(query)
    interest = detect_interest(query)
    category = detect_category(query)

    budget_scores = df.apply(lambda r: budget_score(r, budget), axis=1).values
    gender_scores = df.apply(lambda r: gender_score(r, gender), axis=1).values
    interest_scores = df.apply(lambda r: interest_score(r, interest), axis=1).values
    category_scores = df.apply(lambda r: category_score(r, category), axis=1).values

    final_scores = (
        0.45 * semantic_scores +
        0.20 * df["norm_rating"].values +
        0.15 * df["norm_price"].values +
        0.10 * budget_scores +
        0.05 * gender_scores +
        0.05 * interest_scores +
        0.05 * category_scores
    )

    top_indices = final_scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

# ---------- API ----------
@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    results = search_products(request.query, top_k=3)

    response = []
    for _, row in results.iterrows():
        response.append({
            "product_name": row["product_name"],
            "price": float(row["price"]),
            "rating": float(row["rating"]),
            "category": row["category"]
        })

    return {"recommendations": response}



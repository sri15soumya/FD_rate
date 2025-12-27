import os
import re
import math
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================================
# 🔹 Load environment variables
# ==========================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 🔹 Connect to MySQL database
# (make sure db has numeric min_ten/max_ten in DAYS)
# ==========================================
engine = create_engine("mysql+mysqlconnector://root:soumya%40150905@localhost/fd_rates_db")

def load_fd_data():
    query = "SELECT id, bank, tenure, min_ten, max_ten, senior_citizen_rate FROM fd_rates"
    df = pd.read_sql(query, engine)
    # normalize column names if needed
    df.rename(columns={
        'min_ten': 'min_ten',
        'max_ten': 'max_ten',
        'senior_citizen_rate': 'rate'
    }, inplace=True)
    # replace NaN with None for safety, but keep numeric dtype where present
    df['min_ten'] = df['min_ten'].apply(lambda x: None if (pd.isna(x)) else float(x))
    df['max_ten'] = df['max_ten'].apply(lambda x: None if (pd.isna(x)) else float(x))
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    return df

# ==========================================
# 🔹 Query-to-days parser (user input normalization)
# ==========================================
def clean_metadata(meta_list):
    cleaned = []
    for m in meta_list:
        if isinstance(m, dict):
            cleaned.append({k: ("" if v is None else v) for k, v in m.items()})
        else:
            cleaned.append({})
    return cleaned

def unit_to_days(value: float, unit: str) -> float:
    unit = (unit or "").lower()
    if 'year' in unit or unit == 'y':
        return value * 365.0
    if 'month' in unit:
        return value * 30.0
    if 'day' in unit:
        return value
    return float(value)  # fallback assume days

def parse_query_to_days(q: str):
    """
    Parse user textual query and return (q_min_days, q_max_days).
    Handles:
      - "5 to 6 years"
      - "1-3 months"
      - "between 4 and 45 days"
      - "best for 6 months" -> single value (treated as exact range +/- small buffer)
      - "less than 1 year" -> (0, 365)
      - "more than 3 years" -> (3*365, large cap)
    Returns (None, None) if cannot parse numeric info.
    """
    s = str(q).lower()
    # normalize common words
    s = s.replace('between', ' ').replace('and', ' ').replace('–', '-').replace('—', '-')
    s = s.replace('upto', 'up to')
    # regex to find patterns "X to Y unit" or "X - Y unit"
    # Search for explicit range: e.g. "5 to 6 years", "4-45 days", "5 months to 10 months"
    r = re.search(r'(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(years?|months?|days?|y)?', s)
    if r:
        a, b, unit = r.groups()
        # if unit None, try to infer unit by searching near tokens
        if not unit:
            # try separate units: "5 years to 6 years" handled earlier, else search for "year" anywhere
            if 'year' in s:
                unit = 'year'
            elif 'month' in s:
                unit = 'month'
            elif 'day' in s:
                unit = 'day'
            else:
                unit = 'day'
        a_d = unit_to_days(float(a), unit)
        b_d = unit_to_days(float(b), unit)
        return (min(a_d, b_d), max(a_d, b_d))

    # patterns like "5 years" or "6 months" (single value)
    r2 = re.search(r'(\d+(?:\.\d+)?)\s*(years?|months?|days?|y)', s)
    if r2:
        val, unit = r2.groups()
        d = unit_to_days(float(val), unit)
        # Treat single value as a small window around it (e.g., +/- 0)
        return (d, d)

    # "less than X unit"
    r3 = re.search(r'less than\s*(\d+(?:\.\d+)?)\s*(years?|months?|days?|y)', s)
    if r3:
        val, unit = r3.groups()
        max_d = unit_to_days(float(val), unit)
        return (0.0, max_d)

    # "more than / greater than X unit"
    r4 = re.search(r'(more than|greater than|>|\babove\b)\s*(\d+(?:\.\d+)?)\s*(years?|months?|days?|y)', s)
    if r4:
        val = float(r4.group(2)); unit = r4.group(3)
        min_d = unit_to_days(val, unit)
        # choose an upper cap (10 years = 3650 days)
        return (min_d, 3650.0)

    # try to find two separate numeric+unit tokens and pick nearest pair
    tokens = re.findall(r'(\d+(?:\.\d+)?)\s*(years?|months?|days?|y)?', s)
    nums = [(float(n), u) for n, u in tokens if n.strip() != '']
    if len(nums) >= 2:
        # pick first two by appearance
        a, au = nums[0]
        b, bu = nums[1]
        # if units missing, infer from surrounding text
        if not au:
            au = 'day' if 'day' in s else ('month' if 'month' in s else ('year' if 'year' in s else 'day'))
        if not bu:
            bu = au
        a_d = unit_to_days(a, au)
        b_d = unit_to_days(b, bu)
        return (min(a_d, b_d), max(a_d, b_d))

    return (None, None)

# ==========================================
# 🔹 Numeric overlap retrieval (preferred)
# ==========================================
def numeric_retrieval(df, q_min, q_max, top_k=5):
    """
    Return top_k rows from df where (row.max_ten >= q_min) and (row.min_ten <= q_max)
    ranked by rate desc.
    """
    if q_min is None or q_max is None:
        return pd.DataFrame()

    # filter rows that have numeric min/max
    mask = df['min_ten'].notnull() & df['max_ten'].notnull()
    subset = df[mask].copy()
    # numeric overlap condition
    overlap_mask = (subset['max_ten'] >= q_min) & (subset['min_ten'] <= q_max)
    candidates = subset[overlap_mask].copy()
    if candidates.empty:
        return candidates
    # sort by rate desc (higher rates first)
    candidates = candidates.sort_values(by='rate', ascending=False)
    return candidates.head(top_k)

# ==========================================
# 🔹 Prepare text chunks for embedding + store metadata
# ==========================================
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

def prepare_chunks_and_metadata(df):
    texts = []
    metadatas = []
    ids = []
    for i, row in df.iterrows():
        rate_display = f"{row['rate']}%" if not pd.isna(row['rate']) else "N/A"
        min_t = "" if row['min_ten'] is None else f"{float(row['min_ten'])} days"
        max_t = "" if row['max_ten'] is None else f"{float(row['max_ten'])} days"
        text = f"Bank: {row['bank']}, Tenure: {row['tenure']} ({min_t} - {max_t}), Interest Rate: {rate_display}"
        texts.append(text)
        metadatas.append({
            "id": int(row.get('id', i)),
            "bank": row['bank'],
            "tenure": row['tenure'],
            "min_ten": row['min_ten'],
            "max_ten": row['max_ten'],
            "rate": float(row['rate']) if not pd.isna(row['rate']) else None
        })
        ids.append(f"chunk_{i}")
    return texts, metadatas, ids

# ==========================================
# 🔹 Chroma init + store
# ==========================================
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="fd_rates")

def store_in_vector_db_if_empty(texts, metadatas, ids):
    # if collection empty -> add; if already present, skip (avoid duplicates)
    existing = collection.count()
    if existing == 0:
        embeddings = model_embed.encode(texts, show_progress_bar=True).tolist()
        metadatas = clean_metadata(metadatas)
        collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        return True
    return False

def semantic_retrieval(query, top_k=5):
    q_emb = model_embed.encode([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    if results and len(results["documents"]) > 0:
        # return list of document texts
        docs = results["documents"][0]
        md = results.get("metadatas", [])
        return docs, md
    return [], []

# ==========================================
# 🔹 Gemini answer generation (context + query)
# ==========================================
def generate_answer_from_context(context_text, query):
    prompt = f"""You are an intelligent assistant that helps users compare Indian Fixed Deposit (FD) rates.
Use the provided context to answer the user query accurately and clearly.

Context:
{context_text}

Question:
{query}

Provide a clear, factual, and helpful answer."""
    model_gemini = genai.GenerativeModel("gemini-flash-latest")
    response = model_gemini.generate_content(prompt)
    return response.text

# ==========================================
# 🔹 Top-level query handling (numeric-first, fallback to semantic)
# ==========================================
def answer_query(df, query, top_k=5):
    # parse user query for numeric days
    q_min, q_max = parse_query_to_days(query)
    # if single exact value, consider small tolerance? here treat exact range
    if q_min is not None and q_max is not None:
        numeric_candidates = numeric_retrieval(df, q_min, q_max, top_k=top_k)
        if not numeric_candidates.empty:
            # build concise answer using numeric matches (ranked by rate)
            rows = []
            for _, r in numeric_candidates.iterrows():
                rate = f"{r['rate']}%" if not pd.isna(r['rate']) else "N/A"
                rows.append(f"{r['bank']}: {rate} for {r['tenure']}")
            context_text = "Top numeric matches:\n" + "\n".join(rows)
            # optionally still ask Gemini to write the final answer but it's often unnecessary. We'll call Gemini to produce nicer language.
            return generate_answer_from_context(context_text, query)

    # Fallback: semantic retrieval from embeddings
    docs, metadatas = semantic_retrieval(query, top_k=top_k)
    if docs:
        context_text = "\n".join(docs)
        return generate_answer_from_context(context_text, query)

    # Nothing found
    return "No matching FD rates found for your query."

def initailize_rag():
    df = load_fd_data()
    print(f"✅ Loaded {len(df)} FD records from database.")

    texts, metadatas, ids = prepare_chunks_and_metadata(df)
    print(f"✅ Prepared {len(texts)} text chunks with metadata.")

    stored = store_in_vector_db_if_empty(texts, metadatas, ids)
    if stored:
        print("✅ Generated embeddings and stored them in Chroma.")
    else:
        print("ℹ️ Chroma collection already had data — skipping store.")
    
    return df

def get_fd_answer(df,query):
    
    """
    Main callable function for backend use.
    Takes a query and returns the generated answer text.
    """
    try:
        return answer_query(df, query, top_k=5)
    except Exception as e:
        return f"Error while answering: {e}"
        

# ==========================================
# 🔹 Main Flow
# ==========================================
if __name__ == "__main__":
    print("🔍 Building FD Rate RAG Agent...")

    

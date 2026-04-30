from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ============================================================
# STEP 1: Load a neural embedding model (runs locally, free)
# ============================================================

print("Loading neural embedding model (first time takes a minute)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")

# ============================================================
# STEP 2: Test documents — pairs that mean the same thing
# but use DIFFERENT words
# ============================================================

test_pairs = [
    {
        "name": "Revenue synonyms",
        "doc_a": "Q4 revenue was $2.3 million, up 15% year over year",
        "doc_b": "Sales reached $2.3M in the fourth quarter, a 15% increase",
    },
    {
        "name": "Customer synonyms",
        "doc_a": "Amit Patel is the highest spending customer at $90,000",
        "doc_b": "Our top client by total purchase value spent ninety thousand dollars",
    },
    {
        "name": "Region performance",
        "doc_a": "The south region had the weakest performance",
        "doc_b": "Southern territory underperformed compared to other areas",
    },
    {
        "name": "Completely different topics",
        "doc_a": "Q4 revenue was $2.3 million",
        "doc_b": "The cat sat on the mat in the garden",
    },
    {
        "name": "Same topic different detail",
        "doc_a": "Enterprise Plan is the top selling product",
        "doc_b": "Dashboard Pro has the most orders but lowest revenue",
    },
]

# ============================================================
# STEP 3: Compare TF-IDF vs Neural on each pair
# ============================================================

print("=" * 65)
print("  TF-IDF vs NEURAL EMBEDDINGS — Side by Side")
print("=" * 65)

for pair in test_pairs:
    # TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_vectors = vectorizer.fit_transform([pair["doc_a"], pair["doc_b"]])
    tfidf_score = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]

    # Neural similarity
    neural_vectors = model.encode([pair["doc_a"], pair["doc_b"]])
    neural_score = cosine_similarity([neural_vectors[0]], [neural_vectors[1]])[0][0]

    # Show comparison
    diff = neural_score - tfidf_score
    winner = "NEURAL WINS" if diff > 0.1 else "SIMILAR" if abs(diff) <= 0.1 else "TFIDF WINS"

    print(f"\n  {pair['name']}:")
    print(f"    A: \"{pair['doc_a'][:60]}\"")
    print(f"    B: \"{pair['doc_b'][:60]}\"")
    print(f"    TF-IDF:  {tfidf_score:.3f}")
    print(f"    Neural:  {neural_score:.3f}")
    print(f"    Gap: {diff:+.3f}  [{winner}]")

# ============================================================
# STEP 4: Full semantic search comparison
# ============================================================

documents = [
    "Q4 2025 revenue was $2.3 million, up 15% year over year",
    "The north region generated $920K in Q4, leading all regions",
    "Customer churn rate dropped to 3.2% in Q4 2025",
    "Enterprise Plan is the top selling product at $208K total revenue",
    "Amit Patel is the highest spending customer at $90,000",
    "Average order value across all products is $24,125",
    "The south region had the weakest performance at $310K revenue",
    "Dashboard Pro has the most orders but lowest revenue per unit",
    "10 new customers signed up in 2024 across all regions",
    "Analytics Suite generated $163K in total revenue",
]

# Embed all documents with neural model
doc_embeddings = model.encode(documents)

# Also create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_doc_vectors = tfidf_vectorizer.fit_transform(documents)

queries = [
    "How much money did we make last quarter?",
    "Who spends the most with us?",
    "Which area is underperforming?",
]

print(f"\n{'='*65}")
print("  SEARCH COMPARISON: Same queries, different methods")
print(f"{'='*65}")

for query in queries:
    print(f"\n  Query: \"{query}\"")

    # Neural search
    query_embedding = model.encode([query])
    neural_sims = cosine_similarity(query_embedding, doc_embeddings)[0]
    neural_top = neural_sims.argsort()[-3:][::-1]

    # TF-IDF search
    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_sims = cosine_similarity(query_tfidf, tfidf_doc_vectors)[0]
    tfidf_top = tfidf_sims.argsort()[-3:][::-1]

    print(f"\n    TF-IDF top 3:")
    for rank, idx in enumerate(tfidf_top, 1):
        print(f"      {rank}. [{tfidf_sims[idx]:.3f}] {documents[idx][:65]}")

    print(f"\n    Neural top 3:")
    for rank, idx in enumerate(neural_top, 1):
        print(f"      {rank}. [{neural_sims[idx]:.3f}] {documents[idx][:65]}")
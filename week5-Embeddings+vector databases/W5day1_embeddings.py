import anthropic
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# STEP 1: Use Claude to generate embeddings via Anthropic's API
# We'll use a simple approach — get embeddings using a free model
# ============================================================

# Our document store — a simple Python list
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
    "The west region contributed $740K in Q4 revenue",
    "Monthly revenue peaked in April at $64,000",
    "Rajesh Kumar from the north region placed 3 orders totaling $64,000",
    "The company has 4 sales regions: north, south, east, and west",
    "Q3 to Q4 revenue growth for north region was 15%",
]

# ============================================================
# STEP 2: Create simple embeddings using word overlap + TF-IDF
# (In Week 6, we'll use proper embedding models via APIs)
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF converts text into number vectors based on word importance
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(documents)

print(f"Created embeddings for {len(documents)} documents")
print(f"Each document is now a vector of {doc_vectors.shape[1]} numbers")

# ============================================================
# STEP 3: Semantic search — find most similar documents
# ============================================================

def search(query, n_results=3):
    # Convert query to same vector space
    query_vector = vectorizer.transform([query])

    # Calculate similarity between query and all documents
    similarities = cosine_similarity(query_vector, doc_vectors)[0]

    # Get top N most similar
    top_indices = similarities.argsort()[-n_results:][::-1]

    print(f"\nQuery: \"{query}\"")
    print(f"Top {n_results} results:")
    for rank, idx in enumerate(top_indices, 1):
        score = round(similarities[idx], 3)
        print(f"  {rank}. [{score}] {documents[idx]}")
    return top_indices

# ============================================================
# STEP 4: Test with different queries
# ============================================================

print("=" * 60)
print("  SEMANTIC SEARCH WITH EMBEDDINGS (TF-IDF)")
print("=" * 60)

search("What was the total revenue in Q4?")

search("Which region performed the worst?")

search("Who is our biggest customer?")

search("Which product sells the most?")

# Same meaning, different words
search("How much money did we make last quarter?")

# Unrelated question — should get low scores
search("What is the weather in Delhi?")
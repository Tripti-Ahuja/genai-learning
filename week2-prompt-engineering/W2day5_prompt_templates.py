import anthropic
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

def clean_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ============================================================
# TEMPLATE 1: Review Classifier
# ============================================================

def classify_review(review):
    prompt = f"""Classify this customer review. Return ONLY valid JSON, no markdown.

Review: "{review}"

{{"sentiment": "positive/negative/neutral", "confidence": "high/medium/low", "topic": "main topic in 2-3 words"}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(clean_json(response.content[0].text))

# ============================================================
# TEMPLATE 2: SQL Explainer
# ============================================================

def explain_sql(query):
    prompt = f"""Explain this SQL query in plain English. Return ONLY valid JSON, no markdown.

Query: {query}

{{"explanation": "what it does in one sentence", "tables_used": ["list", "of", "tables"], "complexity": "simple/medium/complex"}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(clean_json(response.content[0].text))

# ============================================================
# TEMPLATE 3: Data Summary Generator
# ============================================================

def summarize_data(data_description, audience):
    prompt = f"""Summarize this data for a {audience}. Return ONLY valid JSON, no markdown.

Data: {data_description}

{{"summary": "2-3 sentence summary", "key_metric": "most important number", "action": "recommended next step"}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(clean_json(response.content[0].text))

# ============================================================
# USE THE TEMPLATES
# ============================================================

print("=== REVIEW CLASSIFIER ===")
reviews = [
    "Love the new dashboard feature, saves me hours!",
    "App crashed again. Third time this week.",
    "It works. Does what it says.",
]
for review in reviews:
    result = classify_review(review)
    print(f"  {result['sentiment']:<10} ({result['confidence']}) - {result['topic']:<20} | {review[:50]}")

print("\n=== SQL EXPLAINER ===")
queries = [
    "SELECT name, COUNT(*) FROM customers GROUP BY name",
    "SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.date > '2024-01-01' GROUP BY c.name HAVING SUM(o.amount) > 5000",
]
for query in queries:
    result = explain_sql(query)
    print(f"  [{result['complexity']}] {result['explanation']}")
    print(f"  Tables: {', '.join(result['tables_used'])}\n")

print("=== DATA SUMMARIZER ===")
result = summarize_data(
    "Q4 revenue: $2.3M (up 15% YoY). Top region: North ($900K). Weakest: South ($300K, down 8%). New customers: 145.",
    "C-level executive"
)
print(f"  Summary:    {result['summary']}")
print(f"  Key Metric: {result['key_metric']}")
print(f"  Action:     {result['action']}")

print()
result2 = summarize_data(
    "Q4 revenue: $2.3M (up 15% YoY). Top region: North ($900K). Weakest: South ($300K, down 8%). New customers: 145.",
    "data analyst"
)
print(f"  Summary:    {result2['summary']}")
print(f"  Key Metric: {result2['key_metric']}")
print(f"  Action:     {result2['action']}")
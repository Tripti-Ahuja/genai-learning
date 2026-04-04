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
# TEST 1: Force Claude to return valid JSON
# ============================================================

prompt = """Analyze this customer review and respond with ONLY valid JSON, no other text:

Review: "Switched from Tableau to your platform. SQL integration is great, saves hours weekly. But dashboards are slow and exports crash on large datasets."

Return this exact JSON structure:
{
    "sentiment": "positive/negative/neutral",
    "score": 1-10,
    "pros": ["list", "of", "pros"],
    "cons": ["list", "of", "cons"],
    "priority_fix": "single most urgent issue",
    "would_recommend": true/false
}"""

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    temperature=0,
    messages=[{"role": "user", "content": prompt}]
)

raw_text = response.content[0].text.strip()
print("--- RAW RESPONSE ---")
print(raw_text)

# Parse JSON and use it like a normal Python dictionary
try:
    data = json.loads(clean_json(raw_text))
    print("\n--- PARSED AS PYTHON DICT ---")
    print(f"Sentiment:      {data['sentiment']}")
    print(f"Score:          {data['score']}/10")
    print(f"Pros:           {', '.join(data['pros'])}")
    print(f"Cons:           {', '.join(data['cons'])}")
    print(f"Priority Fix:   {data['priority_fix']}")
    print(f"Would Recommend: {data['would_recommend']}")
except json.JSONDecodeError as e:
    print(f"\nJSON PARSE ERROR: {e}")
    print("Claude didn't return valid JSON. This is why we need validation!")

# ============================================================
# TEST 2: Batch processing — analyze 5 reviews in one call
# ============================================================

batch_prompt = """Analyze each review and return ONLY a JSON array. No other text.

Reviews:
1. "Best analytics tool I've ever used. Worth every penny."
2. "Crashed 3 times today. Losing patience."
3. "It's fine for basic reports. Nothing special."
4. "The AI features are mind-blowing. Game changer for our team."
5. "Terrible customer support. Waited 2 weeks for a response."

Return this exact format:
[
    {"review_number": 1, "sentiment": "positive/negative/neutral", "score": 1-10},
    ...
]"""

response2 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    temperature=0,
    messages=[{"role": "user", "content": batch_prompt}]
)

raw_text2 = response2.content[0].text.strip()
print("\n--- BATCH RESULTS ---")

try:
    reviews = json.loads(clean_json(raw_text2))
    print(f"{'#':<4} {'Sentiment':<12} {'Score'}")
    print("-" * 28)
    for r in reviews:
        print(f"{r['review_number']:<4} {r['sentiment']:<12} {r['score']}/10")
except json.JSONDecodeError as e:
    print(f"JSON PARSE ERROR: {e}")
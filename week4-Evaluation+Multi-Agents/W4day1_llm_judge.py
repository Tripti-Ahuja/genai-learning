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
# STEP 1: Generate responses to evaluate
# ============================================================

def generate_response(question):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0.5,
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

# ============================================================
# STEP 2: Judge the response on a rubric
# ============================================================

def judge_response(question, response):
    judge_prompt = f"""You are a strict quality evaluator. Rate this AI response on a 1-5 scale for each criterion. Return ONLY valid JSON, no markdown.

Question asked: "{question}"
AI Response: "{response}"

Rate each criterion:
- accuracy: Is the information correct? (1=wrong, 5=perfectly accurate)
- completeness: Does it fully answer the question? (1=barely, 5=comprehensive)
- clarity: Is it easy to understand? (1=confusing, 5=crystal clear)
- conciseness: Is it the right length? (1=too long/short, 5=perfect length)

{{"accuracy": 1-5, "completeness": 1-5, "clarity": 1-5, "conciseness": 1-5, "overall": 1-5, "feedback": "one sentence explaining the score"}}"""

    judge = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    return json.loads(clean_json(judge.content[0].text))

# ============================================================
# TEST: Generate and judge responses
# ============================================================

test_questions = [
    "What is a LEFT JOIN in SQL?",
    "Explain what an API is to a 10-year-old",
    "What is the capital of France?",
    "How do you calculate revenue growth percentage?",
    "Explain the difference between a data analyst and a data engineer",
]

print("=" * 70)
print("  LLM-AS-JUDGE: Claude evaluates its own responses")
print("=" * 70)

all_scores = []

for q in test_questions:
    print(f"\nQ: {q}")

    # Generate the response
    response = generate_response(q)
    print(f"A: {response[:100]}...")

    # Judge it
    scores = judge_response(q, response)
    all_scores.append(scores)

    print(f"   Accuracy: {scores['accuracy']}/5 | Completeness: {scores['completeness']}/5 | Clarity: {scores['clarity']}/5 | Conciseness: {scores['conciseness']}/5 | Overall: {scores['overall']}/5")
    print(f"   Feedback: {scores['feedback']}")

# Summary
print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")
avg_overall = sum(s['overall'] for s in all_scores) / len(all_scores)
avg_accuracy = sum(s['accuracy'] for s in all_scores) / len(all_scores)
print(f"  Average Overall Score: {avg_overall:.1f}/5")
print(f"  Average Accuracy:      {avg_accuracy:.1f}/5")
print(f"  Total Questions:       {len(test_questions)}")

lowest = min(all_scores, key=lambda x: x['overall'])
print(f"  Weakest Response:      {lowest['feedback']}")
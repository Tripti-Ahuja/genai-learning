import anthropic
import json
from dotenv import load_dotenv
from rouge_score import rouge_scorer

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# STEP 1: Define test cases with expected answers
# ============================================================

test_cases = [
    {
        "question": "What is a LEFT JOIN in SQL?",
        "reference": "A LEFT JOIN returns all rows from the left table and matching rows from the right table. If there is no match, NULL values are returned for the right table columns."
    },
    {
        "question": "What does GROUP BY do in SQL?",
        "reference": "GROUP BY groups rows that have the same values in specified columns into summary rows. It is typically used with aggregate functions like COUNT, SUM, AVG, MAX, and MIN."
    },
    {
        "question": "What is an API?",
        "reference": "An API is an Application Programming Interface. It allows two software systems to communicate with each other by sending requests and receiving responses."
    },
    {
        "question": "What is the difference between a primary key and a foreign key?",
        "reference": "A primary key uniquely identifies each row in a table. A foreign key is a column that references the primary key of another table, creating a relationship between the two tables."
    },
    {
        "question": "What is data normalization?",
        "reference": "Data normalization is the process of organizing a database to reduce redundancy and improve data integrity. It involves dividing tables into smaller tables and defining relationships between them."
    },
]

# ============================================================
# STEP 2: Generate Claude's answers
# ============================================================

def get_claude_answer(question):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": f"{question} Answer in 2 sentences max."}]
    )
    return response.content[0].text

# ============================================================
# STEP 3: Score with ROUGE
# ============================================================

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

print("=" * 70)
print("  ROUGE SCORING: Comparing Claude's answers vs reference answers")
print("=" * 70)

all_scores = []

for tc in test_cases:
    claude_answer = get_claude_answer(tc["question"])
    scores = scorer.score(tc["reference"], claude_answer)

    result = {
        "question": tc["question"],
        "rouge1": round(scores["rouge1"].fmeasure, 3),
        "rouge2": round(scores["rouge2"].fmeasure, 3),
        "rougeL": round(scores["rougeL"].fmeasure, 3),
    }
    all_scores.append(result)

    print(f"\nQ: {tc['question']}")
    print(f"  Reference: {tc['reference'][:80]}...")
    print(f"  Claude:    {claude_answer[:80]}...")
    print(f"  ROUGE-1: {result['rouge1']}  |  ROUGE-2: {result['rouge2']}  |  ROUGE-L: {result['rougeL']}")

# ============================================================
# STEP 4: Summary
# ============================================================

print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")

avg_r1 = sum(s["rouge1"] for s in all_scores) / len(all_scores)
avg_r2 = sum(s["rouge2"] for s in all_scores) / len(all_scores)
avg_rL = sum(s["rougeL"] for s in all_scores) / len(all_scores)

print(f"  Average ROUGE-1: {avg_r1:.3f}")
print(f"  Average ROUGE-2: {avg_r2:.3f}")
print(f"  Average ROUGE-L: {avg_rL:.3f}")

best = max(all_scores, key=lambda x: x["rougeL"])
worst = min(all_scores, key=lambda x: x["rougeL"])
print(f"\n  Best match:  {best['question'][:50]}  (ROUGE-L: {best['rougeL']})")
print(f"  Worst match: {worst['question'][:50]}  (ROUGE-L: {worst['rougeL']})")

print(f"\n  Interpretation:")
print(f"  0.0-0.3 = Low overlap (Claude used very different wording)")
print(f"  0.3-0.6 = Moderate overlap (similar content, different phrasing)")
print(f"  0.6-1.0 = High overlap (very similar to reference)")
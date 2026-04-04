import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

def ask_claude(prompt, label):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")
    print(response.content[0].text)
    return response

# --- ZERO-SHOT: No examples, just the instruction ---
zero_shot_prompt = """Classify the following customer review as 'positive', 'negative', or 'neutral'.

Review: "The dashboard loads fast but the filters keep breaking when I select multiple regions."

Classification:"""

# --- FEW-SHOT: Give examples first, then ask ---
few_shot_prompt = """Classify customer reviews as 'positive', 'negative', or 'neutral'.

Review: "Absolutely love the new reporting feature, saves me hours every week!"
Classification: positive

Review: "The app crashes every time I try to export data. Completely unusable."
Classification: negative

Review: "It works fine. Nothing special but gets the job done."
Classification: neutral

Review: "The dashboard loads fast but the filters keep breaking when I select multiple regions."
Classification:"""

ask_claude(zero_shot_prompt, "ZERO-SHOT")
ask_claude(few_shot_prompt, "FEW-SHOT")

# --- Now let's test both on 5 tricky reviews ---
test_reviews = [
    "Great tool but the learning curve is steep",
    "I cancelled my subscription after 2 days",
    "The update fixed some bugs but introduced new ones",
    "My team uses it daily for all our reporting needs",
    "It's okay I guess, nothing compared to Tableau though",
]

print(f"\n{'='*60}")
print("COMPARISON: Zero-shot vs Few-shot on 5 reviews")
print(f"{'='*60}")
print(f"{'Review':<55} {'Zero-shot':<12} {'Few-shot'}")
print("-" * 80)

for review in test_reviews:
    zs_prompt = f"Classify this review as 'positive', 'negative', or 'neutral'. Reply with ONLY the classification word.\n\nReview: \"{review}\"\nClassification:"

    fs_prompt = f"""Classify customer reviews as 'positive', 'negative', or 'neutral'. Reply with ONLY the classification word.

Review: "Absolutely love the new reporting feature!"
Classification: positive

Review: "The app crashes every time I export data."
Classification: negative

Review: "It works fine. Nothing special."
Classification: neutral

Review: "{review}"
Classification:"""

    zs_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        temperature=0,
        messages=[{"role": "user", "content": zs_prompt}]
    )

    fs_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        temperature=0,
        messages=[{"role": "user", "content": fs_prompt}]
    )

    zs_answer = zs_response.content[0].text.strip().lower()
    fs_answer = fs_response.content[0].text.strip().lower()

    match = "✓" if zs_answer == fs_answer else "✗ DIFF"
    print(f"{review:<55} {zs_answer:<12} {fs_answer:<12} {match}")
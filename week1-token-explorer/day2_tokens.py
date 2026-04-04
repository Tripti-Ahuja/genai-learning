import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

sentences = [
    "Hello world",
    "SELECT * FROM customers WHERE revenue > 1000000",
    "Retrieval Augmented Generation is a technique for grounding LLM outputs",
    "मैं एक डेटा एनालिस्ट हूँ",
    "{'name': 'Reeshabh', 'role': 'analyst', 'tools': ['SQL', 'Tableau', 'Python']}"
]

print(f"{'#':<4} {'Tokens':<8} {'Chars':<8} {'Ratio':<8} Sentence")
print("-" * 80)

for i, sentence in enumerate(sentences, 1):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1,
        messages=[{"role": "user", "content": sentence}]
    )

    tokens = response.usage.input_tokens
    chars = len(sentence)
    ratio = round(chars / tokens, 1)

    print(f"{i:<4} {tokens:<8} {chars:<8} {ratio:<8} {sentence[:60]}")

print("\n--- What to notice ---")
print("1. English text: ~4 characters per token")
print("2. SQL/code: often more tokens (symbols are separate tokens)")
print("3. Hindi text: way more tokens per character (non-Latin scripts are expensive)")
print("4. JSON: lots of tokens for punctuation and structure")
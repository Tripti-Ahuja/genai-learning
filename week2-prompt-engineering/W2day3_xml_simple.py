import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

def extract_tag(text, tag):
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start != -1 and end != -1:
        return text[start + len(f"<{tag}>"):end].strip()
    return "NOT FOUND"

prompt = """Analyze this customer review:

"We use your tool daily for Salesforce reporting. Love the SQL integration but the dashboard is too slow."

Respond in EXACTLY this format:

<sentiment>positive, negative, or neutral</sentiment>
<pros>List positives</pros>
<cons>List negatives</cons>
<priority>Single most urgent fix</priority>"""

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=200,
    temperature=0,
    messages=[{"role": "user", "content": prompt}]
)

full_text = response.content[0].text
print("--- FULL RESPONSE ---")
print(full_text)

print("\n--- EXTRACTED ---")
print(f"Sentiment: {extract_tag(full_text, 'sentiment')}")
print(f"Pros:      {extract_tag(full_text, 'pros')}")
print(f"Cons:      {extract_tag(full_text, 'cons')}")
print(f"Priority:  {extract_tag(full_text, 'priority')}")
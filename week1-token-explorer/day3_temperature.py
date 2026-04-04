import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

prompt = "Suggest a name for a data analytics startup"

for temp in [0, 0, 0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=30,
        temperature=temp,
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Temp {temp:<4} → {response.content[0].text.strip()}")
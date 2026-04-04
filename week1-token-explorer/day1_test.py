import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one sentence!"}]
)

print(message.content[0].text)
print(f"\nTokens used — Input: {message.usage.input_tokens}, Output: {message.usage.output_tokens}")
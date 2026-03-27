import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

conversation_history = []
total_input_tokens = 0
total_output_tokens = 0

print("=" * 50)
print("  Claude Chatbot (type 'quit' to exit)")
print("  Model: Haiku 4.5 | Temperature: 0.5")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "quit":
        print(f"\n--- Session Stats ---")
        print(f"Total input tokens:  {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        total_cost = (total_input_tokens * 1.00 / 1_000_000) + (total_output_tokens * 5.00 / 1_000_000)
        print(f"Total cost:          ${total_cost:.5f}")
        print("Goodbye!")
        break

    if not user_input:
        continue

    conversation_history.append({"role": "user", "content": user_input})

    print("\nClaude: ", end="", flush=True)

    full_response = ""

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=0.5,
        system="You are a helpful data analytics mentor. Keep responses concise — 2-3 sentences max unless asked for detail.",
        messages=conversation_history
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    response = stream.get_final_message()
    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens

    conversation_history.append({"role": "assistant", "content": full_response})

    print(f"\n  [tokens: in={response.usage.input_tokens}, out={response.usage.output_tokens}]")
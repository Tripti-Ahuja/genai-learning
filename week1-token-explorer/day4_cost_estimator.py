import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# Pricing per million tokens (as of 2026)
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6-20250415": {"input": 15.00, "output": 75.00},
}

# Test prompts of different sizes
prompts = [
    ("Short question", "What is a JOIN in SQL?"),
    ("Medium prompt", "You are a senior data analyst. I have a PostgreSQL database with tables: customers (id, name, region, signup_date), orders (id, customer_id, amount, order_date), and products (id, name, category, price). Write a SQL query to find the top 10 customers by total order value in the last 6 months, broken down by product category."),
    ("Long prompt", open("day4_cost_estimator.py").read()),
]

print(f"{'Prompt':<18} {'Input Tok':<12} {'Output Tok':<12} {'Haiku $':<12} {'Sonnet $':<12} {'Opus $'}")
print("-" * 78)

for label, prompt_text in prompts:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt_text}]
    )

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens

    costs = {}
    for model_name, prices in PRICING.items():
        short_name = model_name.split("-")[1]
        cost = (in_tok * prices["input"] / 1_000_000) + (out_tok * prices["output"] / 1_000_000)
        costs[short_name] = f"${cost:.5f}"

    print(f"{label:<18} {in_tok:<12} {out_tok:<12} {costs['haiku']:<12} {costs['sonnet']:<12} {costs['opus']}")

print("\n--- Quick math ---")
print("If you send 100 similar medium prompts per day:")
for model_name, prices in PRICING.items():
    short = model_name.split("-")[1]
    daily = 100 * (80 * prices["input"] + 200 * prices["output"]) / 1_000_000
    monthly = daily * 30
    print(f"  {short:<8} → ${daily:.3f}/day → ${monthly:.2f}/month")
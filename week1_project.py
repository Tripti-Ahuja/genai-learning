import anthropic
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

load_dotenv()
client = anthropic.Anthropic()

# 5 different input types to compare
inputs = [
    ("English", "What is the total revenue by region for the last quarter?"),
    ("Hindi", "पिछली तिमाही के लिए क्षेत्र के अनुसार कुल राजस्व क्या है?"),
    ("SQL Query", "SELECT region, SUM(revenue) AS total_revenue FROM sales WHERE order_date >= '2025-10-01' GROUP BY region ORDER BY total_revenue DESC;"),
    ("JSON Data", '{"report": "Q4 Sales", "regions": ["North", "South", "East", "West"], "metric": "revenue", "year": 2025}'),
    ("Python Code", "import pandas as pd\ndf = pd.read_csv('sales.csv')\nresult = df.groupby('region')['revenue'].sum().sort_values(ascending=False)\nprint(result)"),
]

# Pricing
HAIKU_INPUT = 1.00
HAIKU_OUTPUT = 5.00

# Collect results
results = []

print("=" * 70)
print("  WEEK 1 PROJECT: Token Explorer")
print("=" * 70)
print(f"\n{'Type':<15} {'Input Tok':<12} {'Output Tok':<12} {'Chars':<8} {'Cost ($)':<10}")
print("-" * 60)

for label, text in inputs:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": text}]
    )

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    chars = len(text)
    cost = (in_tok * HAIKU_INPUT / 1_000_000) + (out_tok * HAIKU_OUTPUT / 1_000_000)

    results.append({
        "label": label,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "chars": chars,
        "cost": cost,
        "ratio": round(chars / in_tok, 1)
    })

    print(f"{label:<15} {in_tok:<12} {out_tok:<12} {chars:<8} ${cost:.5f}")

print("-" * 60)

# Key insights
most_expensive = max(results, key=lambda x: x["cost"])
most_efficient = max(results, key=lambda x: x["ratio"])
print(f"\nMost expensive input:  {most_expensive['label']} (${most_expensive['cost']:.5f})")
print(f"Most token-efficient:  {most_efficient['label']} ({most_efficient['ratio']} chars/token)")

# Generate bar chart
labels = [r["label"] for r in results]
in_tokens = [r["input_tokens"] for r in results]
out_tokens = [r["output_tokens"] for r in results]

x = range(len(labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: Input vs Output tokens
bar_width = 0.35
bars1 = ax1.bar([i - bar_width/2 for i in x], in_tokens, bar_width, label="Input Tokens", color="#2E75B6")
bars2 = ax1.bar([i + bar_width/2 for i in x], out_tokens, bar_width, label="Output Tokens", color="#ED7D31")
ax1.set_xlabel("Input Type")
ax1.set_ylabel("Token Count")
ax1.set_title("Input vs Output Tokens by Type")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, rotation=15)
ax1.legend()
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(int(bar.get_height())), ha="center", fontsize=9)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(int(bar.get_height())), ha="center", fontsize=9)

# Chart 2: Cost comparison
costs = [r["cost"] * 1000 for r in results]  # convert to millicents for readability
bars3 = ax2.bar(labels, costs, color=["#2E75B6", "#ED7D31", "#548235", "#BF8F00", "#7F77DD"])
ax2.set_xlabel("Input Type")
ax2.set_ylabel("Cost (x $0.001)")
ax2.set_title("Cost per Request (in thousandths of $)")
ax2.set_xticklabels(labels, rotation=15)
for bar in bars3:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("week1_token_comparison.png", dpi=150, bbox_inches="tight")
print("\nChart saved as: week1_token_comparison.png")
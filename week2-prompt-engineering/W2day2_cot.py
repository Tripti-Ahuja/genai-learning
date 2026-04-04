import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

def ask_claude(prompt, label, max_tokens=500):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")
    print(response.content[0].text)
    return response.content[0].text

# ============================================================
# TEST 1: Data analytics word problem
# ============================================================

problem = """A company has 3 sales regions. North had $500K revenue with 20% growth.
South had $800K revenue with -5% growth. West had $300K revenue with 40% growth.
If these growth rates continue for 2 more years, which region will have the highest revenue?"""

# WITHOUT Chain-of-Thought
direct_prompt = f"""{problem}

Give me the answer in one sentence."""

# WITH Chain-of-Thought
cot_prompt = f"""{problem}

Think step by step:
1. Calculate Year 1 revenue for each region
2. Calculate Year 2 revenue for each region
3. Compare and give the final answer"""

ask_claude(direct_prompt, "DIRECT ANSWER (no CoT)")
ask_claude(cot_prompt, "CHAIN-OF-THOUGHT")

# ============================================================
# TEST 2: SQL debugging problem
# ============================================================

sql_problem = """This SQL query is supposed to find customers who placed more than 5 orders
in the last 30 days but it returns wrong results. Find the bug.

SELECT customer_name, COUNT(order_id)
FROM customers c, orders o
WHERE c.id = o.customer_id
AND order_date > '2025-01-01'
HAVING COUNT(order_id) > 5"""

direct_sql = f"""{sql_problem}

What's the bug? Answer in one sentence."""

cot_sql = f"""{sql_problem}

Think step by step:
1. Read the query and understand what each clause does
2. Check if the JOIN is correct
3. Check if the WHERE filter is correct
4. Check if GROUP BY is present (required for HAVING)
5. Check if HAVING condition is correct
6. List all bugs found"""

ask_claude(direct_sql, "SQL BUG - DIRECT ANSWER")
ask_claude(cot_sql, "SQL BUG - CHAIN-OF-THOUGHT")

# ============================================================
# TEST 3: Salesforce funnel math
# ============================================================

funnel_problem = """A Salesforce dashboard shows:
- Total Leads: 5,000
- Qualified Leads: 1,200
- Opportunities Created: 800
- Closed Won: 200
- Average Deal Size: $15,000

What is the overall conversion rate and total revenue from closed deals?"""

direct_funnel = f"""{funnel_problem}

Answer in one sentence."""

cot_funnel = f"""{funnel_problem}

Think step by step:
1. Calculate each stage conversion rate
2. Calculate overall conversion rate (Leads to Closed Won)
3. Calculate total revenue (Closed Won x Average Deal Size)
4. Summarize all metrics"""

ask_claude(direct_funnel, "FUNNEL - DIRECT ANSWER")
ask_claude(cot_funnel, "FUNNEL - CHAIN-OF-THOUGHT")
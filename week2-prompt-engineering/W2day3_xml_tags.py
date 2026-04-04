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

def extract_tag(text, tag):
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start != -1 and end != -1:
        return text[start + len(f"<{tag}>"):end].strip()
    return "NOT FOUND"

# ============================================================
# TEST 1: SQL Analysis with XML structure
# ============================================================

sql_prompt = """Analyze this SQL query for potential issues:

SELECT c.name, COUNT(o.id), SUM(o.amount)
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.order_date > '2024-01-01'
GROUP BY c.name
HAVING SUM(o.amount) > 1000

Respond using EXACTLY this format:

<query_summary>
What this query does in plain English
</query_summary>

<issues>
List each issue found, one per line
</issues>

<fixed_query>
The corrected SQL query
</fixed_query>

<confidence>
high, medium, or low
</confidence>"""

response1 = ask_claude(sql_prompt, "SQL ANALYSIS - XML OUTPUT")

print(f"\n{'='*60}")
print("[EXTRACTED FROM XML TAGS]")
print(f"{'='*60}")
for tag in ["query_summary", "issues", "fixed_query", "confidence"]:
    print(f"\n--- {tag.upper()} ---")
    print(extract_tag(response1, tag))

# ============================================================
# TEST 2: Customer review analysis with XML
# ============================================================

review_prompt = """Analyze this customer review and respond in EXACTLY this format:

Review: "We switched from Tableau to your platform 6 months ago. The SQL integration
is amazing and saves us hours weekly. However, the dashboard loading time is painfully
slow and the export feature crashes on large datasets. Support team was helpful though."

<sentiment>positive, negative, or neutral</sentiment>

<pros>
List each positive point, one per line
</pros>

<cons>
List each negative point, one per line
</cons>

<action_items>
List specific things the product team should fix, one per line
</action_items>

<priority>
The single most urgent issue to fix
</priority>

<response_draft>
A short professional reply to this customer (2-3 sentences)
</response_draft>"""

response2 = ask_claude(review_prompt, "REVIEW ANALYSIS - XML OUTPUT")

print(f"\n{'='*60}")
print("[EXTRACTED SECTIONS]")
print(f"{'='*60}")
print(f"\nSentiment: {extract_tag(response2, 'sentiment')}")
print(f"\nPriority:  {extract_tag(response2, 'priority')}")
print(f"\nDraft Response:\n{extract_tag(response2, 'response_draft')}")

# ============================================================
# TEST 3: CoT + XML combined (the power pattern)
# ============================================================

combined_prompt = """A Salesforce report shows these pipeline numbers:
- Total Leads: 5,000
- Qualified Leads: 1,200
- Opportunities: 800
- Closed Won: 200
- Average Deal Size: $15,000

Analyze the sales funnel. Think step by step inside the thinking tags,
then give structured output.

<thinking>
Show your step-by-step calculations here
</thinking>

<metrics>
Lead-to-Qualified Rate: X%
Qualified-to-Opportunity Rate: X%
Opportunity-to-Close Rate: X%
Overall Conversion Rate: X%
Total Revenue: $X
</metrics>

<weakest_stage>
Which stage has the biggest drop-off and why
</weakest_stage>

<recommendation>
One specific action to improve the weakest stage
</recommendation>"""

response3 = ask_claude(combined_prompt, "CoT + XML COMBINED")

print(f"\n{'='*60}")
print("[QUICK EXTRACT]")
print(f"{'='*60}")
print(f"\nWeakest Stage: {extract_tag(response3, 'weakest_stage')}")
print(f"\nRecommendation: {extract_tag(response3, 'recommendation')}")
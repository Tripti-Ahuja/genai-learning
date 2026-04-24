import anthropic
import sqlite3
import json
import time
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# TOOLS (simplified for testing)
# ============================================================

tools = [
    {
        "name": "run_sql_query",
        "description": "Executes a READ-ONLY SQL SELECT query. Tables: 'customers' (id, name, region, signup_date) and 'orders' (id, customer_id, amount, product, order_date). Regions are lowercase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluates a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]

def run_sql_query(query):
    if not query.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT allowed"})
    try:
        conn = sqlite3.connect("sales.db")
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return json.dumps({"columns": columns, "row_count": len(rows), "data": [dict(zip(columns, row)) for row in rows]})
    except Exception as e:
        return json.dumps({"error": str(e)})

def calculator(expression):
    try:
        return json.dumps({"result": eval(expression)})
    except Exception as e:
        return json.dumps({"error": str(e)})

tool_functions = {"run_sql_query": run_sql_query, "calculator": calculator}

# ============================================================
# AGENT RUNNER (returns answer as string)
# ============================================================

def run_agent(question):
    messages = [{"role": "user", "content": question}]
    max_steps = 6

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0,
            tools=tools,
            system="You are a data analyst. Answer questions using SQL on the sales database. Be concise.",
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "No answer")

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name in tool_functions:
                    result = tool_functions[block.name](**block.input)
                else:
                    result = json.dumps({"error": f"Unknown tool: {block.name}"})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Max steps reached"

# ============================================================
# TEST CASES
# ============================================================

test_cases = [
    {
        "question": "How many customers are in the database?",
        "must_contain": ["10"],
        "category": "simple count"
    },
    {
        "question": "How many customers are in the north region?",
        "must_contain": ["3"],
        "category": "filtered count"
    },
    {
        "question": "What is the total revenue across all orders?",
        "must_contain": ["482"],
        "category": "aggregation"
    },
    {
        "question": "Which product has the highest total revenue?",
        "must_contain": ["Enterprise"],
        "category": "group by + sort"
    },
    {
        "question": "Who is the top customer by total spending?",
        "must_contain": ["Amit"],
        "category": "join + aggregation"
    },
    {
        "question": "How many orders are there for Dashboard Pro?",
        "must_contain": ["8"],
        "category": "filtered count"
    },
    {
        "question": "What is the average order value?",
        "must_contain": ["24125"],
        "category": "average"
    },
    {
        "question": "Which region has the highest total revenue?",
        "must_contain": ["north", "North"],
        "category": "join + group by"
    },
    {
        "question": "How many customers signed up in 2024?",
        "must_contain": ["10"],
        "category": "date filter"
    },
    {
        "question": "If total revenue grew by 20%, what would it be?",
        "must_contain": ["579"],
        "category": "calculation"
    },
]

# ============================================================
# RUN EVALUATION
# ============================================================

print("=" * 70)
print("  EVALUATION HARNESS: Testing Agent on 10 Questions")
print("=" * 70)

results = []

for i, test in enumerate(test_cases, 1):
    start_time = time.time()
    answer = run_agent(test["question"])
    elapsed = round(time.time() - start_time, 1)

    # Check if ANY of the must_contain strings appear in the answer
    passed = any(keyword.lower() in answer.lower() for keyword in test["must_contain"])

    status = "PASS" if passed else "FAIL"
    results.append({"status": status, "time": elapsed, "category": test["category"]})

    print(f"\n  [{status}] Q{i}: {test['question']}")
    print(f"    Expected: {test['must_contain']}")
    print(f"    Answer: {answer[:120]}...")
    print(f"    Time: {elapsed}s | Category: {test['category']}")

# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*70}")
print("  RESULTS SUMMARY")
print(f"{'='*70}")

passed = sum(1 for r in results if r["status"] == "PASS")
failed = sum(1 for r in results if r["status"] == "FAIL")
avg_time = round(sum(r["time"] for r in results) / len(results), 1)

print(f"  Passed: {passed}/{len(results)}")
print(f"  Failed: {failed}/{len(results)}")
print(f"  Pass Rate: {round(passed/len(results)*100)}%")
print(f"  Avg Response Time: {avg_time}s")

if failed > 0:
    print(f"\n  Failed questions:")
    for i, (r, t) in enumerate(zip(results, test_cases), 1):
        if r["status"] == "FAIL":
            print(f"    Q{i}: {t['question']}")
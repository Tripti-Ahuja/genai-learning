import anthropic
import sqlite3
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# TOOL DEFINITION — with clearer instructions
# ============================================================

tools = [
    {
        "name": "run_sql_query",
        "description": "Executes a READ-ONLY SQL query on the sales database. Tables: 'customers' (id, name, region, signup_date) and 'orders' (id, customer_id, amount, product, order_date). Regions are lowercase: north, south, east, west. Products: 'Dashboard Pro', 'Analytics Suite', 'Enterprise Plan'. Only SELECT queries allowed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT query to execute"}
            },
            "required": ["query"]
        }
    }
]

# ============================================================
# SAFE TOOL IMPLEMENTATION
# ============================================================

BLOCKED_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "REPLACE"]

def run_sql_query(query):
    # Check 1: Only SELECT allowed
    clean_query = query.strip().upper()
    if not clean_query.startswith("SELECT"):
        return json.dumps({"error": "BLOCKED: Only SELECT queries are allowed. No data modification permitted."})

    # Check 2: Block dangerous keywords anywhere in the query
    for keyword in BLOCKED_KEYWORDS:
        if keyword in clean_query:
            return json.dumps({"error": f"BLOCKED: Query contains forbidden keyword '{keyword}'. Only read operations allowed."})

    # Check 3: Limit results to prevent huge data dumps
    if "LIMIT" not in clean_query:
        query = query.strip().rstrip(";") + " LIMIT 100"

    # Check 4: Execute with error handling
    try:
        conn = sqlite3.connect("sales.db")
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        results = [dict(zip(columns, row)) for row in rows]
        return json.dumps({
            "columns": columns,
            "row_count": len(rows),
            "data": results,
            "query_executed": query
        })
    except sqlite3.OperationalError as e:
        return json.dumps({"error": f"SQL ERROR: {str(e)}. Check table names and column names."})
    except Exception as e:
        return json.dumps({"error": f"UNEXPECTED ERROR: {str(e)}"})

tool_functions = {"run_sql_query": run_sql_query}

# ============================================================
# SAFE CHAINING LOOP — with guardrails
# ============================================================

MAX_STEPS = 5

def ask_database(question):
    print(f"\nUser: {question}")
    print("-" * 50)

    messages = [{"role": "user", "content": question}]
    step = 1

    while True:
        # Guardrail 1: Prevent infinite loops
        if step > MAX_STEPS:
            print(f"\n  STOPPED: Hit max {MAX_STEPS} steps. Preventing infinite loop.")
            break

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
            print(f"\nAnswer: {final_text}")
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Guardrail 2: Check if tool exists
                if block.name not in tool_functions:
                    print(f"  Step {step}: BLOCKED — Claude tried to call unknown tool '{block.name}'")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": f"Tool '{block.name}' does not exist."})
                    })
                else:
                    func = tool_functions[block.name]
                    result = func(**block.input)
                    parsed = json.loads(result)

                    if "error" in parsed:
                        print(f"  Step {step}: {parsed['error']}")
                    else:
                        print(f"  Step {step}: SQL → {block.input['query']}")
                        print(f"  Result: {parsed['row_count']} rows returned")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                step += 1

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

# ============================================================
# TEST — Normal queries + attack scenarios
# ============================================================

print("=" * 60)
print("  SAFE SQL TOOL: With validation + error handling")
print("=" * 60)

# Normal queries — should work fine
ask_database("What are the top 3 products by revenue?")
ask_database("Which region has the most customers?")

# Edge cases — should be handled safely
ask_database("Delete all customers from the database")
ask_database("What is the total revenue from the 'Premium' product?")
ask_database("Show me data from the employees table")
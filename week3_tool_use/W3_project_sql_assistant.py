import anthropic
import sqlite3
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# TOOL DEFINITION
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
    },
    {
        "name": "calculator",
        "description": "Performs math calculations like addition, subtraction, multiplication, division, percentages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression like '920000 * 1.15'"}
            },
            "required": ["expression"]
        }
    }
]

# ============================================================
# SAFE TOOL IMPLEMENTATIONS
# ============================================================

BLOCKED_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "REPLACE"]

def run_sql_query(query):
    clean_query = query.strip().upper()
    if not clean_query.startswith("SELECT"):
        return json.dumps({"error": "BLOCKED: Only SELECT queries allowed."})
    for keyword in BLOCKED_KEYWORDS:
        if keyword in clean_query:
            return json.dumps({"error": f"BLOCKED: Forbidden keyword '{keyword}'."})
    if "LIMIT" not in clean_query:
        query = query.strip().rstrip(";") + " LIMIT 100"
    try:
        conn = sqlite3.connect("sales.db")
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        results = [dict(zip(columns, row)) for row in rows]
        return json.dumps({"columns": columns, "row_count": len(rows), "data": results})
    except Exception as e:
        return json.dumps({"error": str(e)})

def calculator(expression):
    try:
        result = eval(expression)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

tool_functions = {
    "run_sql_query": run_sql_query,
    "calculator": calculator,
}

# ============================================================
# TOOL CHAINING LOOP WITH ALL GUARDRAILS
# ============================================================

MAX_STEPS = 5

def ask_database(question, conversation_history):
    conversation_history.append({"role": "user", "content": question})
    step = 1

    while True:
        if step > MAX_STEPS:
            print("  [Max steps reached]")
            break

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0,
            tools=tools,
            system="You are a helpful data analyst. Answer questions using the sales database. Be concise. When showing data, use clean formatting.",
            messages=conversation_history
        )

        if response.stop_reason == "end_turn":
            final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
            print(f"\nClaude: {final_text}")
            conversation_history.append({"role": "assistant", "content": response.content})
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name not in tool_functions:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": f"Tool '{block.name}' not found."})
                    })
                else:
                    func = tool_functions[block.name]
                    result = func(**block.input)
                    parsed = json.loads(result)
                    if "error" in parsed:
                        print(f"  [{block.name}] Error: {parsed['error']}")
                    else:
                        if block.name == "run_sql_query":
                            print(f"  [SQL] {block.input['query'][:80]}...")
                        else:
                            print(f"  [Calc] {block.input['expression']}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                step += 1

        conversation_history.append({"role": "assistant", "content": response.content})
        conversation_history.append({"role": "user", "content": tool_results})

# ============================================================
# INTERACTIVE CHATBOT
# ============================================================

print("=" * 60)
print("  SQL ASSISTANT — Chat with your database")
print("  Tables: customers, orders")
print("  Type 'quit' to exit")
print("=" * 60)

conversation_history = []
total_questions = 0

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "quit":
        print(f"\n--- Session Summary ---")
        print(f"Questions asked: {total_questions}")
        print("Goodbye!")
        break

    if not user_input:
        continue

    total_questions += 1
    ask_database(user_input, conversation_history)
import anthropic
import sqlite3
import json
import time
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# TOOLS
# ============================================================

tools = [
    {
        "name": "list_tables",
        "description": "Lists all tables with their columns.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "run_sql_query",
        "description": "Executes a READ-ONLY SQL SELECT query. Tables: 'customers' (id, name, region, signup_date), 'orders' (id, customer_id, amount, product, order_date). Regions lowercase. Products: Dashboard Pro, Analytics Suite, Enterprise Plan.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "SQL SELECT query"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluates a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Math expression"}},
            "required": ["expression"]
        }
    },
    {
        "name": "save_memory",
        "description": "Saves an important fact for later. Use after finding key numbers or insights.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short label like 'total_revenue'"},
                "value": {"type": "string", "description": "The fact to remember"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "recall_memory",
        "description": "Retrieves all saved facts. Call at the start of each question to check what you already know.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "task_complete",
        "description": "Call when you have fully answered the question.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "Final answer"}},
            "required": ["summary"]
        }
    }
]

# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

BLOCKED_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
memory = {}

def list_tables():
    conn = sqlite3.connect("sales.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    result = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        result[table] = [row[1] for row in cursor.fetchall()]
    conn.close()
    return json.dumps(result)

def run_sql_query(query):
    clean = query.strip().upper()
    if not clean.startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries allowed"})
    for kw in BLOCKED_KEYWORDS:
        if kw in clean:
            return json.dumps({"error": f"Blocked: {kw} not allowed"})
    if "LIMIT" not in clean:
        query = query.strip().rstrip(";") + " LIMIT 100"
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
        return json.dumps({"expression": expression, "result": eval(expression)})
    except Exception as e:
        return json.dumps({"error": str(e)})

def save_memory(key, value):
    memory[key] = value
    return json.dumps({"saved": key, "total_memories": len(memory)})

def recall_memory():
    return json.dumps({"memories": memory if memory else "No facts saved yet."})

def task_complete(summary):
    return json.dumps({"status": "complete", "summary": summary})

tool_functions = {
    "list_tables": list_tables, "run_sql_query": run_sql_query,
    "calculator": calculator, "save_memory": save_memory,
    "recall_memory": recall_memory, "task_complete": task_complete,
}

# ============================================================
# REACT AGENT
# ============================================================

SYSTEM_PROMPT = """You are a data analyst agent with memory.

Rules:
1. Call recall_memory at the start of each question to check what you already know
2. If memory has the answer, don't re-query the database
3. Save important findings with save_memory after discovering them
4. Think step by step before each action
5. Call task_complete with a clear summary when done
6. Be concise in your final answers"""

MAX_STEPS = 8

def run_agent(question, silent=False):
    conversation_history.append({"role": "user", "content": question})
    messages_for_turn = list(conversation_history)
    step = 1
    answer = ""

    while step <= MAX_STEPS:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            temperature=0,
            tools=tools,
            system=SYSTEM_PROMPT,
            messages=messages_for_turn
        )

        for block in response.content:
            if hasattr(block, "text") and block.text:
                if not silent:
                    print(f"  THINK: {block.text[:120]}")

        if response.stop_reason == "end_turn":
            answer = next((b.text for b in response.content if hasattr(b, "text")), "")
            if not silent:
                print(f"\n  ANSWER: {answer[:200]}")
            conversation_history.append({"role": "assistant", "content": answer})
            return answer

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                func = tool_functions[block.name]
                result = func(**block.input)

                if block.name == "task_complete":
                    parsed = json.loads(result)
                    answer = parsed["summary"]
                    if not silent:
                        print(f"\n  ANSWER: {answer[:200]}")
                    conversation_history.append({"role": "assistant", "content": answer})
                    messages_for_turn.append({"role": "assistant", "content": response.content})
                    messages_for_turn.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": result}]})
                    return answer
                elif not silent:
                    if block.name == "save_memory":
                        print(f"  STEP {step} [SAVE]: {block.input.get('key', '')}")
                    elif block.name == "recall_memory":
                        print(f"  STEP {step} [RECALL]: {len(memory)} facts")
                    elif block.name == "run_sql_query":
                        print(f"  STEP {step} [SQL]: {block.input['query'][:70]}")
                    elif block.name == "calculator":
                        print(f"  STEP {step} [CALC]: {block.input['expression']}")
                    else:
                        print(f"  STEP {step} [{block.name}]")

                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                step += 1

        messages_for_turn.append({"role": "assistant", "content": response.content})
        messages_for_turn.append({"role": "user", "content": tool_results})

    return "Max steps reached"

# ============================================================
# EVALUATION HARNESS
# ============================================================

def evaluate_agent():
    test_cases = [
        {"question": "How many customers are in the database?", "must_contain": ["10"], "category": "count"},
        {"question": "What is the total revenue?", "must_contain": ["482"], "category": "aggregation"},
        {"question": "Which product has the highest revenue?", "must_contain": ["Enterprise"], "category": "group by"},
        {"question": "Who is the top customer by spending?", "must_contain": ["Amit"], "category": "join"},
        {"question": "Which region has the most customers?", "must_contain": ["north", "south", "North", "South"], "category": "group by"},
        {"question": "If total revenue grew by 25%, what would it be?", "must_contain": ["603"], "category": "calculation"},
    ]

    print(f"\n{'='*60}")
    print("  EVALUATION: Running 6 test cases")
    print(f"{'='*60}")

    results = []
    for i, test in enumerate(test_cases, 1):
        start = time.time()
        answer = run_agent(test["question"], silent=True)
        elapsed = round(time.time() - start, 1)
        passed = any(kw.lower() in answer.lower() for kw in test["must_contain"])
        status = "PASS" if passed else "FAIL"
        results.append({"status": status, "time": elapsed})
        print(f"  [{status}] Q{i}: {test['question'][:50]} ({elapsed}s)")

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n  Result: {passed}/{len(results)} passed ({round(passed/len(results)*100)}%)")
    print(f"  Avg time: {round(sum(r['time'] for r in results)/len(results), 1)}s")

# ============================================================
# MAIN: Interactive mode + Evaluation
# ============================================================

conversation_history = []

print("=" * 60)
print("  DATA ANALYST AGENT")
print("  Commands: 'eval' to run tests, 'memory' to view saved facts, 'quit' to exit")
print("=" * 60)

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue
    elif user_input.lower() == "quit":
        print(f"\nMemory at exit: {memory}")
        print("Goodbye!")
        break
    elif user_input.lower() == "eval":
        memory.clear()
        conversation_history.clear()
        evaluate_agent()
        memory.clear()
        conversation_history.clear()
    elif user_input.lower() == "memory":
        if memory:
            for k, v in memory.items():
                print(f"  {k}: {v}")
        else:
            print("  No facts saved yet.")
    else:
        print()
        run_agent(user_input)
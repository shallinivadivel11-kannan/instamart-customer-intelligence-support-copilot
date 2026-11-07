
from pathlib import Path
import sys
import sqlite3
import json
import csv
import time
from datetime import datetime, timezone

# --------------------------------------------------
# 1. PROJECT PATHS AND IMPORTS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm
from src.policy_retrieval import retrieve

DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. TOOL FUNCTIONS
# --------------------------------------------------

def get_order(order_id):
    """Retrieve order details from SQLite."""

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

        if row is None:
            return {"error": "order not found"}

        return dict(row)

    finally:
        connection.close()


def search_policy(query):
    """Retrieve relevant policy chunks."""

    results = retrieve(query, k=3)

    return {"results": results}


def calculate_refund(order_amount, delivery_minutes):
    """Calculate late-delivery refund."""

    if order_amount is None or delivery_minutes is None:
        return {"refund_inr": 0}

    if float(delivery_minutes) > 45:
        return {"refund_inr": float(order_amount)}

    return {"refund_inr": 0}



def write_ledger(tool_name, order_id, refund_inr):
    """Record a refund only if the same transaction does not already exist."""

    refund_inr = float(refund_inr)

    # Check existing ledger entries
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                if (
                    record["tool_name"] == tool_name
                    and record["order_id"] == order_id
                    and float(record["refund_inr"]) == refund_inr
                ):
                    return {
                        "refund_inr": refund_inr,
                        "already_recorded": True
                    }

    # Create a new ledger entry
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {
        "refund_inr": refund_inr,
        "already_recorded": False
    }



def issue_refund(order_id, refund_inr):
    """Record an automatically approved refund."""

    return write_ledger(
        "issue_refund",
        order_id,
        refund_inr
    )


def escalate_to_human(order_id, refund_inr):
    """Record a refund requiring human approval."""

    return write_ledger(
        "escalate_to_human",
        order_id,
        refund_inr
    )


# --------------------------------------------------
# 3. TOOL SPECIFICATIONS AND TOOL MAP
# --------------------------------------------------

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve order details using an order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search company policy documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the late-delivery refund amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": [
                    "order_amount",
                    "delivery_minutes"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an automatically approved refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Send a refund for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# --------------------------------------------------
# 4. SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an AGENT that handles customer refund requests.

Follow these rules:

1. Identify the order ID.
2. Use get_order to retrieve order details.
3. Use search_policy to retrieve the refund policy.
4. Only completed orders can receive refunds.
5. Completed orders delivered in more than 45 minutes
   qualify for a full refund.
6. Orders delivered in 45 minutes or less receive no refund.
7. Refunds up to INR 1000 can be issued automatically.
8. Refunds above INR 1000 must be escalated to a human.
9. Never issue a refund for a cancelled or missing order.
10. Use the appropriate tools and explain the result.

Return a clear final answer with the correct status.
"""


# --------------------------------------------------
# 5. CHECKPOINT SAVING
# --------------------------------------------------

def save_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    step_number,
    tools_executed,
    llm_calls,
    steps
):
    """Save the current agent state to a JSON checkpoint."""

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "tools_executed": tools_executed,
        "llm_calls": llm_calls,
        "steps": steps
    }

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    with checkpoint_path.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=4)

    return checkpoint_path


# --------------------------------------------------
# 6. FALLBACK FUNCTIONS
# --------------------------------------------------

def get_status(final_answer):

    if not final_answer:
        return "FALLBACK"

    return final_answer.split(":", 1)[0].strip().upper()


def make_fallback(status, steps):

    summary = "; ".join(
        f"{step['tool']}: {step['observation']}"
        for step in steps[-3:]
    )

    return (
        f"{status}: Processing stopped safely. "
        "Human support should review the request. "
        f"Work completed so far: {summary}"
    )


# --------------------------------------------------
# 7. REACT AGENT LOOP
# --------------------------------------------------

def continue_agent(
    run_id,
    scenario_id,
    messages,
    steps,
    llm_calls,
    tokens_used,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):

    start_time = time.monotonic()
    final_answer = ""
    status = "FALLBACK"

    starting_calls = llm_calls

    for iteration in range(max_iterations):

        # Time safeguard
        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = make_fallback(status, steps)
            break

        response = mock_llm.chat(
            messages,
            tools=TOOLS_SPEC
        )

        llm_calls += 1
        tokens_used += response.usage.total_tokens

        message = response.choices[0].message
        tool_calls = message.tool_calls

        # Model returned final answer
        if not tool_calls:
            final_answer = message.content or ""
            status = get_status(final_answer)
            break

        # Token safeguard
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = make_fallback(status, steps)
            break

        # Maximum iteration safeguard
        if llm_calls >= starting_calls + max_iterations:
            status = "FALLBACK_MAX_ITERATIONS"
            final_answer = make_fallback(status, steps)
            break

        assistant_tool_calls = []

        for call in tool_calls:

            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            assistant_tool_calls.append({
                "id": call.id,
                "name": call.name,
                "arguments": arguments
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        # Execute requested tools
        for call in tool_calls:

            tool_name = call.name
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            try:
                observation = TOOL_MAP[tool_name](**arguments)

            except Exception as error:
                observation = {
                    "error": str(error)
                }

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            tools_executed = [
                step["tool"] for step in steps
            ]

            # Save after every tool observation
            save_checkpoint(
                run_id,
                scenario_id,
                messages,
                tokens_used,
                len(steps),
                tools_executed,
                llm_calls,
                steps
            )

            # Simulate interruption after checkpoint
            if (
                crash_after_tools is not None
                and len(steps) == crash_after_tools
            ):
                raise RuntimeError(
                    "Simulated interruption after checkpoint."
                )

    else:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = make_fallback(status, steps)

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None
):
    """Start a new ReAct agent run."""

    mock_llm.reset()

    if run_id is None:
        run_id = scenario_id

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    return continue_agent(
        run_id=run_id,
        scenario_id=scenario_id,
        messages=messages,
        steps=[],
        llm_calls=0,
        tokens_used=0,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools
    )


# --------------------------------------------------
# 8. RESUME AGENT
# --------------------------------------------------

def resume_agent(run_id, max_iterations=6,
                 token_limit=6000, time_limit_seconds=10):

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    with checkpoint_path.open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    print("Resuming run:", run_id)
    print("Previously executed tools:",
          checkpoint["tools_executed"])

    # Do not reset mock_llm here.
    # Continue using the saved conversation.

    return continue_agent(
        run_id=run_id,
        scenario_id=checkpoint["scenario_id"],
        messages=checkpoint["messages"],
        steps=checkpoint["steps"],
        llm_calls=checkpoint["llm_calls"],
        tokens_used=checkpoint["tokens_used"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )


# --------------------------------------------------
# 9. REFUND AMOUNT FROM AGENT RESULT
# --------------------------------------------------

def get_refund_amount(result):

    for step in reversed(result["steps"]):

        observation = step["observation"]

        if isinstance(observation, dict):
            if "refund_inr" in observation:
                return observation["refund_inr"]

    return 0


# --------------------------------------------------
# 10. RUN SEVEN OFFICIAL SCENARIOS
# --------------------------------------------------

def run_scenarios():

    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(SCENARIOS_PATH)

    with SCENARIOS_PATH.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        scenarios = list(csv.DictReader(file))

    results = []

    # Fresh reproducible output files
    RUNS_PATH.write_text("", encoding="utf-8")
    LEDGER_PATH.write_text("", encoding="utf-8")

    for scenario in scenarios:

        scenario_id = scenario["scenario_id"]

        user_message = (
            scenario.get("message")
            or scenario.get("user_message")
            or ""
        )

        if not user_message:
            raise ValueError(
                f"Missing message for {scenario_id}"
            )

        print("\nRunning:", scenario_id)

        result = run_agent(
            user_message,
            scenario_id,
            run_id=f"scenario_{scenario_id}"
        )

        record = {
            "scenario_id": scenario_id,
            "status": result["status"],
            "final_answer": result["final_answer"],
            "refund_inr": get_refund_amount(result),
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"]
        }

        with RUNS_PATH.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")

        results.append(record)

        print("Status:", result["status"])
        print("Refund:", record["refund_inr"])
        print("Model calls:", result["llm_calls"])
        print("Tokens:", result["tokens_used"])

    return results


# --------------------------------------------------
# 11. SAFEGUARD DEMONSTRATIONS
# --------------------------------------------------

def run_safeguard_demos():

    print("\nTOKEN LIMIT DEMONSTRATION")

    token_result = run_agent(
        "My order ORD-111469 was delivered late. "
        "Please process my refund.",
        "TOKEN_DEMO",
        token_limit=200,
        run_id="TOKEN_DEMO"
    )

    print("Token demo status:", token_result["status"])

    print("\nTIME LIMIT DEMONSTRATION")

    time_result = run_agent(
        "My order ORD-111469 was delivered late. "
        "Please process my refund.",
        "TIME_DEMO",
        time_limit_seconds=0,
        run_id="TIME_DEMO"
    )

    print("Time demo status:", time_result["status"])

    print("\nMAX ITERATIONS DEMONSTRATION")

    iteration_result = run_agent(
        "[LOOP]",
        "ITERATION_DEMO",
        max_iterations=6,
        run_id="ITERATION_DEMO"
    )

    print(
        "Iteration demo status:",
        iteration_result["status"]
    )


# --------------------------------------------------
# 12. CHECKPOINT AND RESUME DEMONSTRATION
# --------------------------------------------------

def run_resume_demo():

    with SCENARIOS_PATH.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        scenarios = list(csv.DictReader(file))

    if not scenarios:
        raise ValueError("No scenarios found.")

    scenario = scenarios[0]

    scenario_id = scenario["scenario_id"]

    user_message = (
        scenario.get("message")
        or scenario.get("user_message")
        or ""
    )

    run_id = "resume_demo"

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("\nCHECKPOINT AND RESUME DEMONSTRATION")

    try:
        run_agent(
            user_message,
            scenario_id,
            run_id=run_id,
            crash_after_tools=2
        )

    except RuntimeError as error:
        print("Simulated crash:", error)

    print("\nResuming saved run...")

    result = resume_agent(run_id)

    print("Final status:", result["status"])
    print("Final answer:", result["final_answer"])

    print("Combined tool sequence:")

    for step in result["steps"]:
        print(step["tool"])

    return result


# --------------------------------------------------
# 13. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting Refund Agent project...")

    print("\nRunning official scenarios...")
    scenario_results = run_scenarios()

    print("\nRunning safeguard demonstrations...")
    run_safeguard_demos()

    print("\nRunning checkpoint and resume demonstration...")
    resume_result = run_resume_demo()

    print("\nALL EXECUTION TASKS COMPLETED.")

    print("Scenario count:", len(scenario_results))
    print("Resume status:", resume_result["status"])

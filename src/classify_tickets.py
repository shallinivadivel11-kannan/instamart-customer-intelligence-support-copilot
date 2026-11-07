from pathlib import Path
import json
import pandas as pd
import re
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
sys.path.append(str(Path(__file__).resolve().parent.parent))
import mock_llm

ROOT = Path(__file__).resolve().parent.parent

MAX_INPUT_TOKENS = 400

# Load tickets without converting empty strings to NaN
tickets = pd.read_csv(
    ROOT / "data" / "raw" / "tickets.csv",
    keep_default_na=False
)

# Allowed categories
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# Zero-shot system prompt
ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket.

Allowed categories: billing, delivery, technical, fraud.

Return JSON only in this format:
{"category": "...", "reason": "..."}
"""

# Few-shot system prompt
FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket.

Allowed categories: billing, delivery, technical, fraud.

Return JSON only in this format:
{"category": "...", "reason": "..."}

EXAMPLES:
Example 1:
Ticket: My payment was deducted twice.
Answer: {"category": "billing", "reason": "Duplicate payment"}

Example 2:
Ticket: Someone accessed my account without permission.
Answer: {"category": "fraud", "reason": "Unauthorized access"}
"""

# Redaction function
def redact_text(text):
    counts = {
        "emails": 0,
        "phones": 0,
        "order_ids": 0
    }

    text, counts["emails"] = re.subn(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        "[EMAIL]",
        text
    )

    text, counts["phones"] = re.subn(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        text
    )

    text, counts["order_ids"] = re.subn(
        r"ORD-\d{6}",
        "[ORDER_ID]",
        text
    )

    return text, counts


# Process tickets in original order
redacted_tickets = []
total_counts = {
    "emails": 0,
    "phones": 0,
    "order_ids": 0
}

for text in tickets["text"]:
    redacted, counts = redact_text(text)
    redacted_tickets.append(redacted)

    for key in total_counts:
        total_counts[key] += counts[key]

# Verify no personal information remains
for text in redacted_tickets:
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    assert not re.search(r"\b[6-9]\d{9}\b", text)
    assert not re.search(r"ORD-\d{6}", text)

print("========== INPUT GUARDRAILS ==========")

empty_count = sum(
    1 for text in tickets["text"]
    if not text.strip()
)

long_count = sum(
    1 for text in tickets["text"]
    if text.strip() and mock_llm.est_tokens(text) > MAX_INPUT_TOKENS
)

print("Total tickets:", len(tickets))
print("Empty tickets:", empty_count)
print("Long tickets:", long_count)

print("\n========== REDACTION RESULTS ==========")
print("Emails redacted:", total_counts["emails"])
print("Phones redacted:", total_counts["phones"])
print("Order IDs redacted:", total_counts["order_ids"])

total_redactions = sum(total_counts.values())
print("Total substitutions:", total_redactions)

print("\nPrivacy verification: PASSED")

import random
import time

BASE_DELAY = 0.05
MAX_RETRIES = 3

def call_with_retry(messages, ticket_id):
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Ticket {ticket_id}: retry {retry_number}, "
                f"waiting {wait:.4f} seconds"
            )

            time.sleep(wait)

            import json

def validate_response(reply):
    # Try JSON validation first
    try:
        data = json.loads(reply)
        category = data.get("category")

        if category in CATEGORIES:
            return category, "classified_json"

        return None, "flagged_invalid_output"

    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: extract category words
    words = re.findall(r"[a-z]+", reply.lower())

    matches = [
        word for word in words
        if word in CATEGORIES
    ]

    if len(matches) == 1:
        return matches[0], "classified_keyword"

    return None, "flagged_invalid_output"

def process_tickets(prompt, run_name):
    mock_llm.reset()

    results = []

    for index, row in tickets.iterrows():
        ticket_id = row["ticket_id"]
        original_text = row["text"]
        redacted_text = redacted_tickets[index]

        # Guardrail 1: Empty ticket
        if not original_text.strip():
            results.append({
                "ticket_id": ticket_id,
                "status": "skipped_empty",
                "category": None,
                "tokens_used": 0
            })
            continue

        # Guardrail 2: Long ticket
        if mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": ticket_id,
                "status": "flagged_long",
                "category": None,
                "tokens_used": 0
            })
            continue

        # Ensure only redacted text reaches the model
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", redacted_text)
        assert not re.search(r"\b[6-9]\d{9}\b", redacted_text)
        assert not re.search(r"ORD-\d{6}", redacted_text)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted_text}
        ]

        try:
            response = call_with_retry(messages, ticket_id)

            reply = response.choices[0].message.content
            category, status = validate_response(reply)
            tokens_used = response.usage.total_tokens

        except mock_llm.TransientLLMError:
            category = None
            status = "flagged_invalid_output"
            tokens_used = 0

        results.append({
            "ticket_id": ticket_id,
            "status": status,
            "category": category,
            "tokens_used": tokens_used
        })

    results_df = pd.DataFrame(results)

    print(f"\n========== {run_name} RESULTS ==========")
    print(results_df["status"].value_counts())

    return results_df


# Run few-shot classification
few_shot_results = process_tickets(
    FEW_SHOT_PROMPT,
    "FEW-SHOT"
)

print("\nClassification run completed!")
def calculate_metrics(results_df, run_name):
    # Match predictions with the original ticket labels
    comparison = results_df.merge(
        tickets[["ticket_id", "true_category"]],
        on="ticket_id",
        how="left"
    )

    classified = comparison[
        comparison["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    accuracy = accuracy_score(
        classified["true_category"],
        classified["category"]
    )

    total_tokens = classified["tokens_used"].sum()
    tokens_per_ticket = total_tokens / len(classified)

    print(f"\n========== {run_name} METRICS ==========")
    print("Classified tickets:", len(classified))
    print(f"Accuracy: {accuracy:.1%}")
    print("Total tokens:", total_tokens)
    print(f"Tokens per classified ticket: {tokens_per_ticket:.2f}")

    return comparison, classified


# Calculate few-shot metrics
few_comparison, few_classified = calculate_metrics(
    few_shot_results,
    "FEW-SHOT"
)
# Run zero-shot classification
zero_shot_results = process_tickets(
    ZERO_SHOT_PROMPT,
    "ZERO-SHOT"
)

# Calculate zero-shot metrics
zero_comparison, zero_classified = calculate_metrics(
    zero_shot_results,
    "ZERO-SHOT"
)

# Confusion matrix
print("\n========== ZERO-SHOT CONFUSION MATRIX ==========")

print(confusion_matrix(
    zero_classified["true_category"],
    zero_classified["category"],
    labels=CATEGORIES
))

print("Category order:", CATEGORIES)

def run_budget_demo():
    TOKEN_BUDGET = 6000
    mock_llm.reset()

    running_tokens = 0
    results = []

    for index, row in tickets.iterrows():
        ticket_id = row["ticket_id"]
        original_text = row["text"]
        redacted_text = redacted_tickets[index]

        if not original_text.strip():
            status = "skipped_empty"
            category = None
            tokens_used = 0

        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
            category = None
            tokens_used = 0

        elif running_tokens >= TOKEN_BUDGET:
            status = "budget_exhausted"
            category = None
            tokens_used = 0

        else:
            messages = [
                {"role": "system", "content": ZERO_SHOT_PROMPT},
                {"role": "user", "content": redacted_text}
            ]

            response = call_with_retry(messages, ticket_id)
            reply = response.choices[0].message.content

            category, status = validate_response(reply)
            tokens_used = response.usage.total_tokens
            running_tokens += tokens_used

        results.append({
            "ticket_id": ticket_id,
            "status": status,
            "category": category,
            "tokens_used": tokens_used
        })

    budget_df = pd.DataFrame(results)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    budget_df.to_csv(
        output_dir / "triage_budget_demo.csv",
        index=False
    )

    print("\n========== TOKEN BUDGET DEMO ==========")
    print(budget_df["status"].value_counts())
    print("Tokens used:", running_tokens)
    print("Budget:", TOKEN_BUDGET)
    print("Saved: outputs/triage_budget_demo.csv")


run_budget_demo()

def create_final_triage():
    segments = pd.read_csv(
        ROOT / "outputs" / "customer_segments.csv"
    )

    # Customer-to-segment lookup
    segment_lookup = segments.set_index("customer_id")[
        "segment_name"
    ].to_dict()

    # Customer-to-premium lookup
    premium_lookup = segments.set_index("customer_id")[
        "is_premium"
    ].to_dict()

    # Use few-shot classification results
    final_df = few_shot_results.merge(
        tickets[["ticket_id", "customer_id"]],
        on="ticket_id",
        how="left"
    )

    final_df["segment_name"] = final_df["customer_id"].map(
        segment_lookup
    )

    final_df["is_premium"] = (
        final_df["customer_id"].map(premium_lookup)
        .fillna(0)
        .astype(int)
    )

    final_df["priority"] = "NORMAL"
    final_df["route"] = "support_queue"

    for index, row in final_df.iterrows():

        if row["status"] in [
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted"
        ]:
            final_df.at[index, "route"] = "human_review"

        elif row["category"] == "fraud":
            final_df.at[index, "priority"] = "HIGH"
            final_df.at[index, "route"] = "security_team"

        elif row["is_premium"] == 1:
            final_df.at[index, "priority"] = "HIGH"
            final_df.at[index, "route"] = "premium_desk"

    final_df = final_df[
        [
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used"
        ]
    ]

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    final_df.to_csv(
        output_dir / "triage_results.csv",
        index=False
    )

    print("\n========== FINAL TRIAGE RESULTS ==========")
    print("Total tickets:", len(final_df))
    print("\nStatus counts:")
    print(final_df["status"].value_counts())
    print("\nRouting counts:")
    print(final_df["route"].value_counts())
    print("\nPriority counts:")
    print(final_df["priority"].value_counts())
    print("\nSaved: outputs/triage_results.csv")

create_final_triage()
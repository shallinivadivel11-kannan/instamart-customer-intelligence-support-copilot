# Instamart Customer Intelligence & Support Copilot

**Capstone Project | AI/ML | Data Analytics**

## 1. Overview

The Instamart Customer Intelligence & Support Copilot is an end-to-end data and AI project designed to improve customer experience through data-driven insights and automated support.

The project addresses three key business challenges:

- Unreliable raw customer and order data.
- Difficulty identifying customers at risk of leaving.
- High volumes of customer support tickets and refund requests.

**Solution:** A unified system combining data engineering, customer analytics, machine learning, ticket classification and a policy-based refund agent.

## 2. Reproduce

**Requirements:** Python, pandas, NumPy, scikit-learn, joblib and SQLite.

Install dependencies:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python generate_data.py
```

Run the project scripts in order:

```bash
python src/clean_data.py
python src/build_database.py
python src/check_data_quality.py
python src/customer_features.py
python src/rfm_analysis.py
python src/customer_segmentation.py
python src/lapse_dataset.py
python src/lapse_features.py
python src/lapse_model.py
python src/retention_list.py
python src/classify_tickets.py
python src/validate_triage.py
python src/policy_retrieval.py
python src/refund_agent.py
```

## 3. Part A – Data Foundation

**Objective:** Clean, validate and store customer, product and order data.

- Audited raw datasets and addressed duplicates, missing values and invalid records.
- Standardised customer information and cleaned order records.
- Created a SQLite database with three tables.
- Executed seven SQL queries and cross-checked results using Pandas.

**Key results:**

| Metric | Result |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Clean orders | 12,309 |
| Duplicate order IDs after cleaning | 0 |
| Completed-order revenue | ₹29,98,668.80 |

**Outcome:** A validated and structured data foundation for analytics and machine learning.

## 4. Part B – Customer Segmentation & Lapse Prediction

**Objective:** Identify customer groups and predict potential customer inactivity.

- Performed Recency, Frequency and Monetary (RFM) analysis.
- Applied K-Means clustering and identified four customer segments.
- Built Logistic Regression and Random Forest lapse prediction models.
- Evaluated class imbalance using oversampling.
- Generated a retention list of high-priority customers.

**Key results:**

| Metric | Result |
|---|---:|
| Customer segments | 4 |
| Customers analysed for segmentation | 1,175 |
| Lapse prediction accuracy | 73% |
| Best model cross-validation F1 | 0.6873 |
| Oversampled model lapse F1 | 0.69 |

**Outcome:** Customer segmentation and predictive insights to support targeted retention strategies.

## 5. Part C – Support Ticket Triage

**Objective:** Automate support ticket classification while maintaining privacy and controlling model usage.

- Processed 164 support tickets.
- Implemented email, phone number and order ID redaction.
- Added input validation, retry handling and JSON output validation.
- Compared zero-shot and few-shot classification.
- Implemented token-budget controls and automated ticket routing.

**Key results:**

| Metric | Result |
|---|---:|
| Tickets processed | 164 |
| PII substitutions | 278 |
| Few-shot classification accuracy | 100% |
| Zero-shot classification accuracy | 85.7% |
| Support routing categories | 4 |

**Outcome:** Structured ticket classification with privacy safeguards and controlled processing.

## 6. Part D – Policy RAG Assistant & Refund Agent

**Objective:** Provide policy-grounded answers and automate eligible refund workflows.

- Created 15 policy chunks from five policy documents.
- Implemented TF-IDF retrieval and RAG-based answers.
- Evaluated 14 policy questions.
- Developed a ReAct agent with five tools connected to SQLite and policy retrieval.
- Implemented refund approval, human escalation and safe fallback handling.
- Added token, time and iteration safeguards.
- Implemented checkpoint recovery and duplicate transaction protection.

**Policy evaluation:**

| Metric | With 10-word overlap | Without overlap |
|---|---:|---:|
| Correct expected facts | 12/12 | 10/12 |
| Correct fallback responses | 2/2 | 2/2 |

**Refund agent evaluation:**

| Scenario | Result |
|---|---|
| S1–S2 | Successful refunds |
| S3 | Human escalation |
| S4–S5 | Refund denied |
| S6 | Order not found |
| S7 | Maximum-iteration fallback |

**Results:** All 7 scenarios matched expected outcomes. The checkpoint/resume demonstration also completed successfully without creating a duplicate ledger transaction.

**Outcome:** A policy-grounded refund workflow with controlled tool execution and recovery support.

## 7. Limitations and Next Steps

**Current limitations:**
- The project uses synthetic data.
- The supplied offline mock LLM is deterministic and is not a production language model.
- TF-IDF retrieval has limited semantic understanding.
- The refund ledger is a local JSONL implementation.

**Future improvements:**
- Test with real-world datasets.
- Explore semantic search and embedding-based retrieval.
- Improve model generalisation and customer retention predictions.
- Integrate secure production-grade payment and support systems.

---

**Project Deliverables:** Python source code, cleaned datasets, SQLite database, trained ML model, evaluation outputs, refund ledger and checkpoint files.

**Repository:** Instamart Customer Intelligence & Support Copilot
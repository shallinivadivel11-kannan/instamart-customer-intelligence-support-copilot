
from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(ROOT))
import mock_llm


# --------------------------------------------------
# STEP 1: LOAD POLICY DOCUMENTS AND CREATE CHUNKS
# --------------------------------------------------

def load_policy_chunks(chunk_size=40, overlap=10):
    documents = []
    print("Loading policy documents...")

    step = chunk_size - overlap

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        print("Reading:", file_path.name)

        text = file_path.read_text(encoding="utf-8")
        words = text.split()

        file_chunks = []

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                break

            documents.append({
                "source": file_path.name,
                "chunk": len(file_chunks),
                "text": " ".join(chunk_words)
            })

            file_chunks.append(" ".join(chunk_words))

            if start + chunk_size >= len(words):
                break

        print("Chunks loaded:", len(file_chunks))

    print("\nTotal policy chunks:", len(documents))
    return documents


documents = load_policy_chunks(chunk_size=40, overlap=10)

texts = [item["text"] for item in documents]

vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(texts)

print("Vocabulary size:", len(vectorizer.get_feature_names_out()))


# --------------------------------------------------
# STEP 2: TF-IDF POLICY RETRIEVAL
# --------------------------------------------------

def retrieve(query, k=3):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "source": documents[index]["source"],
            "score": round(float(scores[index]), 4),
            "text": documents[index]["text"]
        })

    return results


# --------------------------------------------------
# STEP 3: RAG POLICY ANSWER FUNCTION
# --------------------------------------------------

def answer(question, retrieval_function=retrieve):
    results = retrieval_function(question, k=3)

    context_parts = []

    for item in results:
        context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )

    context = "\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a POLICY ASSISTANT. "
                "Answer only using the policy context provided. "
                "Do not use outside knowledge or invent policy rules. "
                "If the answer is not available in the context, "
                "reply exactly: I could not find that information "
                "in the policy documents."
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            )
        }
    ]

    response = mock_llm.chat(messages)

    answer_text = response.choices[0].message.content

    sources = list(dict.fromkeys(
        item["source"] for item in results
    ))

    return answer_text, sources


# --------------------------------------------------
# STEP 4: EVALUATE POLICY QUESTIONS
# --------------------------------------------------


def evaluate_policy(retrieval_function, label):
    eval_path = RAW_DIR / "policy_eval.csv"
    evaluation_data = pd.read_csv(eval_path)

    results = []

    for _, row in evaluation_data.iterrows():
        question = str(row["question"])
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = (
            "" if pd.isna(row["expected_fact"])
            else str(row["expected_fact"]).strip()
        )

        answer_text, sources = answer(
            question,
            retrieval_function
        )

        # NONE means the question is unanswerable.
        is_unanswerable = expected_doc.upper() == "NONE"

        if not is_unanswerable:
            document_correct = expected_doc in sources
            fact_correct = expected_fact in answer_text
            fallback_correct = None
        else:
            document_correct = None
            fact_correct = None
            fallback_correct = answer_text.startswith(
                "I could not find"
            )

        results.append({
            "question": question,
            "answer": answer_text,
            "sources": ", ".join(sources),
            "document_correct": document_correct,
            "fact_correct": fact_correct,
            "fallback_correct": fallback_correct
        })

    result_df = pd.DataFrame(results)

    answerable = result_df[
        result_df["document_correct"].notna()
    ]

    unanswerable = result_df[
        result_df["fallback_correct"].notna()
    ]

    document_score = int(
        answerable["document_correct"].sum()
    )

    fact_score = int(
        answerable["fact_correct"].sum()
    )

    fallback_score = int(
        unanswerable["fallback_correct"].sum()
    )

    print("\n" + "=" * 55)
    print(label)
    print("=" * 55)

    print("Total questions:", len(result_df))
    print(
        "Correct document:",
        document_score,
        "/",
        len(answerable)
    )
    print(
        "Correct expected fact:",
        fact_score,
        "/",
        len(answerable)
    )
    print(
        "Correct unanswerable fallback:",
        fallback_score,
        "/",
        len(unanswerable)
    )

    return result_df


def retrieve_without_overlap(query, k=3):
    """Retrieve policy chunks created without overlap."""

    zero_documents = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        words = file_path.read_text(
            encoding="utf-8"
        ).split()

        for start in range(0, len(words), 40):
            chunk_words = words[start:start + 40]

            if not chunk_words:
                break

            zero_documents.append({
                "source": file_path.name,
                "text": " ".join(chunk_words)
            })

    zero_texts = [
        item["text"] for item in zero_documents
    ]

    zero_vectorizer = TfidfVectorizer()
    zero_vectors = zero_vectorizer.fit_transform(zero_texts)

    query_vector = zero_vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        zero_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "source": zero_documents[index]["source"],
            "score": round(float(scores[index]), 4),
            "text": zero_documents[index]["text"]
        })

    return results

# --------------------------------------------------
# STEP 6: RUN EVALUATIONS AND SAVE OUTPUTS
# --------------------------------------------------

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\nStarting policy evaluation...")

    overlap_results = evaluate_policy(
        retrieve,
        "40-WORD CHUNKS WITH 10-WORD OVERLAP"
    )

    no_overlap_results = evaluate_policy(
        retrieve_without_overlap,
        "40-WORD CHUNKS WITH ZERO OVERLAP"
    )

    overlap_path = OUTPUT_DIR / "policy_eval_overlap.csv"
    no_overlap_path = OUTPUT_DIR / "policy_eval_no_overlap.csv"

    overlap_results.to_csv(
        overlap_path,
        index=False
    )

    no_overlap_results.to_csv(
        no_overlap_path,
        index=False
    )

    print("\nEvaluation files saved:")
    print(overlap_path)
    print(no_overlap_path)

    print("\nPolicy evaluation completed.")

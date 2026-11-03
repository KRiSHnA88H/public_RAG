
import csv
import time
import math
import statistics
from pathlib import Path

import retrieval
import final


# Questions and relevant passages taken from your cat.txt dataset.
# Each passage is a substring of a fact in the knowledge base.
TEST_CASES = [
    {
        "question": "How much of their day do cats spend sleeping?",
        "relevant": ["cats spend 2/3 of every day sleeping"],
    },
    {
        "question": "What is the technical term for a cat hairball?",
        "relevant": ['technical term for a cat’s hairball is a “bezoar”'],
    },
    {
        "question": "What is a group of cats called?",
        "relevant": ['group of cats is called a “clowder”'],
    },
    {
        "question": "How many different sounds can cats make?",
        "relevant": ["Cats make about 100 different sounds"],
    },
    {
        "question": "What was the first cat in space called?",
        "relevant": ["first cat in space was a French cat named Felicette"],
    },
    {
        "question": "How fast can a cat run over a short distance?",
        "relevant": ["top speed of approximately 31 mph"],
    },
    {
        "question": "How high can cats jump?",
        "relevant": ["jump up to five times its own height"],
    },
    {
        "question": "How many whiskers does a cat usually have on each side?",
        "relevant": ["about 12 whiskers on each side"],
    },
    {
        "question": "What is the normal body temperature of a cat?",
        "relevant": ["between 100.5 ° and 102.5 °F"],
    },
    {
        "question": "How many bones does a cat have?",
        "relevant": ["cat has 230 bones in its body"],
    },
]


TOP_K = 3
OUTPUT_CSV = Path(__file__).resolve().parent / "evaluation_results.csv"


def percentile(values, percent):
    if not values:
        return 0.0

    values = sorted(values)
    index = max(0, math.ceil(percent / 100 * len(values)) - 1)
    return values[index]


def main():
    print("=" * 55)
    print("CAT RAG: RETRIEVAL AND PERFORMANCE EVALUATION")
    print("=" * 55)

    print("\nBuilding the embedding index...")
    start = time.perf_counter()
    retrieval.build_index()
    index_time = time.perf_counter() - start

    print(f"Index preparation time: {index_time:.3f} seconds")
   

    print("Embedding index preparation completed.")

    # Warm up embedding/retrieval once; exclude it from measurements.
    retrieval.retrieve(TEST_CASES[0]["question"], top_n=TOP_K)

    retrieval_times = []
    answer_times = []
    recalls = []
    reciprocal_ranks = []
    rows = []

    for i, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        relevant_phrases = [
            phrase.casefold() for phrase in test["relevant"]
        ]

        try:
            # Measure retrieval latency.
            start = time.perf_counter()
            results = retrieval.retrieve(question, top_n=TOP_K)
            retrieval_seconds = time.perf_counter() - start

            retrieval_times.append(retrieval_seconds)

            # Find ranks of relevant chunks using the known passages.
            relevant_ranks = []

            for rank, (chunk, score) in enumerate(results, start=1):
                normalized_chunk = chunk.casefold()

                if any(
                    phrase in normalized_chunk
                    for phrase in relevant_phrases
                ):
                    relevant_ranks.append(rank)

            # Binary Recall@3 for each question: relevant passage found?
            recall_at_3 = float(bool(relevant_ranks))

            # Reciprocal rank of the first relevant passage.
            mrr = (
                1.0 / min(relevant_ranks)
                if relevant_ranks else 0.0
            )

            recalls.append(recall_at_3)
            reciprocal_ranks.append(mrr)

            # Measure end-to-end answer generation separately.
            start = time.perf_counter()
            answer = final.generate_answer(question)
            answer_seconds = time.perf_counter() - start

            answer_times.append(answer_seconds)

            scores = [float(score) for _, score in results]

            rows.append({
                "question": question,
                "retrieval_latency_s": round(retrieval_seconds, 4),
                "answer_latency_s": round(answer_seconds, 4),
                "recall_at_3": recall_at_3,
                "reciprocal_rank": round(mrr, 4),
                "top_similarity": round(max(scores), 4) if scores else "",
                "retrieved_chunks": len(results),
                "answer_characters": len(answer),
                "status": "success",
            })

            print(f"\n[{i}/{len(TEST_CASES)}] {question}")
            print(f"  Recall@3: {recall_at_3:.0f}")
            print(f"  Reciprocal rank: {mrr:.2f}")
            print(f"  Retrieval: {retrieval_seconds:.3f}s")
            print(f"  Answer generation: {answer_seconds:.3f}s")

        except Exception as error:
            print(f"\n[{i}/{len(TEST_CASES)}] FAILED: {error}")

            rows.append({
                "question": question,
                "retrieval_latency_s": "",
                "answer_latency_s": "",
                "recall_at_3": "",
                "reciprocal_rank": "",
                "top_similarity": "",
                "retrieved_chunks": "",
                "answer_characters": "",
                "status": f"failed: {error}",
            })

    # Aggregate only successful test cases.
    successful = [row for row in rows if row["status"] == "success"]

    if not successful:
        print("\nNo successful queries. Check Ollama and your models.")
        return

    successful_recalls = [
        row["recall_at_3"] for row in successful
    ]
    successful_mrrs = [
        row["reciprocal_rank"] for row in successful
    ]
    retrieval_times = [
        row["retrieval_latency_s"] for row in successful
    ]
    answer_times = [
        row["answer_latency_s"] for row in successful
    ]

    print("\n" + "=" * 55)
    print("FINAL METRICS")
    print("=" * 55)

    print(f"Successful queries: {len(successful)}/{len(rows)}")
    print(f"Recall@3: {statistics.mean(successful_recalls):.2%}")
    print(f"MRR: {statistics.mean(successful_mrrs):.4f}")

    print("\nRetrieval latency:")
    print(f"  Mean: {statistics.mean(retrieval_times):.4f}s")
    print(f"  P50:  {percentile(retrieval_times, 50):.4f}s")
    print(f"  P95:  {percentile(retrieval_times, 95):.4f}s")

    print("\nEnd-to-end answer latency:")
    print(f"  Mean: {statistics.mean(answer_times):.4f}s")
    print(f"  P50:  {percentile(answer_times, 50):.4f}s")
    print(f"  P95:  {percentile(answer_times, 95):.4f}s")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-question results saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()

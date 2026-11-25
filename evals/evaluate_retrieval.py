import json
import sys
from pathlib import Path
from uuid import UUID

API_DIR = Path(__file__).resolve().parents[1] / "apps" / "api"
sys.path.insert(0, str(API_DIR))

from app.db.session import SessionLocal
from app.services.retrieval import retrieve_chunks


DATASET = Path(__file__).parent / "datasets" / "retrieval_v1.jsonl"


def main():
    cases = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    answerable_total = 0
    answerable_hits = 0
    abstention_total = 0
    correct_abstentions = 0

    reciprocal_rank_sum = 0.0

    with SessionLocal() as db:
        for case in cases:
            results = retrieve_chunks(
                db=db,
                query=case["query"],
                version_id=UUID(case["version_id"]),
                limit=5,
            )

            retrieved_sections = {
                item["chunk"].section_path[-1]
                for item in results
                if item["chunk"].section_path
            }

            if case["should_abstain"]:
                abstention_total += 1
                passed = len(results) == 0

                if passed:
                    correct_abstentions += 1
            else:
                answerable_total += 1
            
                relevant_rank = next(
                    (
                        rank
                        for rank, item in enumerate(results, start=1)
                        if item["chunk"].section_path
                        and set(case["expected_sections"])
                        & set(item["chunk"].section_path)
                    ),
                    None,
                )
            
                passed = relevant_rank is not None

            if passed:
                answerable_hits += 1
                reciprocal_rank_sum += 1 / relevant_rank

            print(
                f"{case['id']}: {'PASS' if passed else 'FAIL'} "
                f"| retrieved={len(results)} "
                f"| sections={sorted(retrieved_sections)}"
            )

    total_passed = answerable_hits + correct_abstentions

    mrr = (
        reciprocal_rank_sum / answerable_total
        if answerable_total
        else 0.0
    )

    print("\n--- BENCHMARK SUMMARY ---")
    print(f"Total cases: {len(cases)}")
    print(f"Answerable retrieval hits: {answerable_hits}/{answerable_total}")
    print(f"Correct abstentions: {correct_abstentions}/{abstention_total}")
    print(f"Overall passed: {total_passed}/{len(cases)}")
    print(f"MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()

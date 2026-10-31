"""Standalone Evaluator test, modeled after tests/test_generator_pipeline.py.

Put this file in your project's tests/ folder, open it in VS Code, and click
Run Python File. It does not import the Generator, ChromaDB, or the workflow.

Tests:
  1. Redis / Memurai connectivity
  2. Evaluator rejects an answer contradicted by evidence
  3. Evaluator accepts a correct grounded answer
  4. Evaluator accepts an honest refusal when evidence is missing
  5. Independent Evaluator memory in Redis
  6. Structured output and score ranges

The three evaluation cases make real OpenRouter calls and consume API tokens.
"""

import logging
import os
import sys
import time
import uuid

# Windows console: display Unicode safely
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Make the project root importable when running this script directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

_passed = 0
_failed = 0


def divider(title: str):
    print("\n" + "=" * 74)
    print("  " + title)
    print("=" * 74)


def check(label: str, condition: bool, detail: str = ""):
    global _passed, _failed
    print(f"   {'PASS' if condition else 'FAIL'} | {label}")
    if detail:
        print("       " + detail)
    if condition:
        _passed += 1
    else:
        _failed += 1


def test_case(evaluator, label, question, answer, context, should_approve):
    divider(label)
    print("Question:", question)
    print("Candidate answer:", answer)
    print("Retrieved context:", context)
    start = time.perf_counter()

    evaluation = evaluator.evaluate(
        query=question,
        answer=answer,
        retrieved_context=context,
        loop=1,
    )

    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
    print("Approved:", evaluation.is_acceptable)
    print("Scores:", {
        "accuracy": evaluation.accuracy,
        "relevance": evaluation.relevance,
        "completeness": evaluation.completeness,
        "consistency": evaluation.consistency,
        "grounding": evaluation.grounding,
    })
    print("Unsupported claims:", evaluation.unsupported_claims)
    print("Feedback:", evaluation.feedback)

    check(
        "Correct approval / rejection decision",
        evaluation.is_acceptable is should_approve,
    )
    check("Returned structured EvaluationResult", evaluation.__class__.__name__ == "EvaluationResult")
    check(
        "All scores are in the 0-10 range",
        all(
            isinstance(score, int) and 0 <= score <= 10
            for score in (
                evaluation.accuracy,
                evaluation.relevance,
                evaluation.completeness,
                evaluation.consistency,
                evaluation.grounding,
            )
        ),
    )
    if not should_approve:
        check("Rejected answer includes feedback", bool(evaluation.feedback.strip()))
    return evaluation


def main():
    from core.cache.redis_client import get_redis_client, is_redis_available
    from core.evaluator.evaluator_agent import EvaluatorAgent
    from core.memory.evaluator_memory import EvaluatorMemory

    session_id = "test-evaluator-" + uuid.uuid4().hex[:10]
    print("Evaluator session:", session_id)

    divider("TEST 1: Redis / Memurai Connectivity")
    redis_ok = is_redis_available()
    check("Redis is reachable", redis_ok)
    if not redis_ok:
        print("Redis is unavailable. LLM evaluation may work, but memory saves may fail.")

    evaluator = EvaluatorAgent(session_id=session_id)
    completed_evaluations = 0

    cases = [
        (
            "TEST 2: Incorrect answer must be rejected",
            "When was Python first released?",
            "Python was first released in 1995.",
            "Python was first released in 1991.",
            False,
        ),
        (
            "TEST 3: Correct grounded answer should be accepted",
            "When was Python first released?",
            "Python was first released in 1991.",
            "Python was first released in 1991.",
            True,
        ),
        (
            "TEST 4: Unknown information should be handled honestly",
            "What is the population of the fictional city Zorbland?",
            "The provided sources do not contain that information.",
            "The provided source discusses Python programming basics only.",
            True,
        ),
    ]

    for title, question, answer, context, expected in cases:
        try:
            test_case(evaluator, title, question, answer, context, expected)
            completed_evaluations += 1
        except Exception as exc:
            divider(title)
            check("Evaluator invocation completed", False, str(exc))
            print("Check your OpenRouter credentials, LLM model, and JSON output parsing.")
            break

    divider("TEST 5: Evaluator Memory")
    history = evaluator.memory.get_history()
    print("Redis key:", evaluator.memory._key)
    print("Evaluation records:", len(history))
    if redis_ok:
        check("Memory saved each completed evaluation", len(history) == completed_evaluations)
        if history:
            print("Last recorded decision:", history[-1]["evaluation"]["is_acceptable"])
            print("Last recorded feedback:", history[-1]["evaluation"]["feedback"])
        client = get_redis_client()
        check("Evaluator memory uses Redis list", client.type(evaluator.memory._key) == "list")
    else:
        print("Memory checks skipped because Redis is unavailable.")

    divider("TEST 6: Independent Memory Namespace")
    check("Evaluator key prefix is memory:eval:", EvaluatorMemory.KEY_PREFIX == "memory:eval:")
    check("Evaluator does not use Generator memory prefix", not evaluator.memory._key.startswith("memory:gen:"))

    divider("TEST SUMMARY")
    total = _passed + _failed
    print(f"Passed: {_passed}/{total}")
    print(f"Failed: {_failed}/{total}")
    print("Session ID:", session_id)
    if _failed == 0:
        print("Evaluator tests passed.")
    else:
        print("Review failed checks above; send the terminal output for debugging.")


if __name__ == "__main__":
    main()

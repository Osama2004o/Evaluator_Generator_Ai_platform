"""
Offline test for the Evaluator–Generator loop using fake agents
(no LLM, Redis, or ChromaDB — no API tokens used).

Run:
    python tests/test_workflow_loop.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.evaluator.schemas import EvaluationResult  # noqa: E402
from workflow import EvaluatorGeneratorWorkflow  # noqa: E402

_failed = 0


def check(label: str, condition: bool):
    global _failed
    print(f"   {'PASS' if condition else 'FAIL'} | {label}")
    _failed += not condition


class FakeGenerator:
    session_id = "test-session"

    def __init__(self, fail_on_loop=None):
        self.fail_on_loop = fail_on_loop
        self.calls = []

    def generate(self, query, loop, evaluator_feedback, previous_answer):
        self.calls.append((loop, evaluator_feedback, previous_answer))
        if loop == self.fail_on_loop:
            return {"answer": "[Generator Error] boom"}
        return {"answer": f"draft {loop}", "sources": ["test_sample.txt"], "retrieved_context": "ctx"}


class FakeEvaluator:
    def __init__(self, approve_on_loop=None, fail_on_loop=None):
        self.approve_on_loop = approve_on_loop
        self.fail_on_loop = fail_on_loop
        self.calls = []

    def evaluate(self, query, answer, retrieved_context, loop):
        self.calls.append(loop)
        if loop == self.fail_on_loop:
            raise RuntimeError("Evaluator failed")
        ok = loop == self.approve_on_loop
        return EvaluationResult(
            is_acceptable=ok, accuracy=9, relevance=9, completeness=9, consistency=9, grounding=9,
            unsupported_claims=[] if ok else ["made-up claim"],
            feedback="" if ok else f"fix loop {loop}",
        )


def run(gen, ev, query="What is a Singleton?"):
    return EvaluatorGeneratorWorkflow(gen, ev).run(query)


print("1. Approved on first loop")
gen, ev = FakeGenerator(), FakeEvaluator(approve_on_loop=1)
r = run(gen, ev)
check("approved after 1 loop", r["approved"] and r["loops_used"] == 1 and r["note"] is None)

print("2. Approved on loop 3 after feedback")
gen, ev = FakeGenerator(), FakeEvaluator(approve_on_loop=3)
r = run(gen, ev)
check("approved after 3 loops", r["approved"] and r["loops_used"] == 3 and r["answer"] == "draft 3")
check("feedback + unsupported claims sent to Generator", "fix loop 1" in gen.calls[1][1] and "made-up claim" in gen.calls[1][1])
check("previous answer sent to Generator", gen.calls[1][2] == "draft 1")

print("3. Never approved -> stops at 4 loops")
gen, ev = FakeGenerator(), FakeEvaluator()
r = run(gen, ev)
check("status max_loops_reached", r["status"] == "max_loops_reached")
check("exactly 4 generate + 4 evaluate calls", len(gen.calls) == 4 and ev.calls == [1, 2, 3, 4])
check("latest answer with note", r["answer"] == "draft 4" and r["note"])

print("4. Evaluator fails on loop 2")
r = run(FakeGenerator(), FakeEvaluator(fail_on_loop=2))
check("returns loop-2 draft with note", r["status"] == "error" and r["answer"] == "draft 2" and r["note"])

print("5. Generator fails on loop 2")
gen, ev = FakeGenerator(fail_on_loop=2), FakeEvaluator()
r = run(gen, ev)
check("keeps draft 1, no error text", r["status"] == "error" and r["answer"] == "draft 1" and ev.calls == [1])

print("6. Generator fails on loop 1")
r = run(FakeGenerator(fail_on_loop=1), FakeEvaluator())
check("safe failure, no error text shown", r["status"] == "error" and "[Generator Error]" not in r["answer"])

print("7. Invalid input")
gen, ev = FakeGenerator(), FakeEvaluator()
check("empty question rejected", run(gen, ev, "  ")["status"] == "invalid_input" and not gen.calls)

print("8. max_loops must be 1..4")
for bad in (0, 5):
    try:
        EvaluatorGeneratorWorkflow(FakeGenerator(), FakeEvaluator(), max_loops=bad)
        check(f"max_loops={bad} refused", False)
    except ValueError:
        check(f"max_loops={bad} refused", True)

print("\nALL PASSED" if not _failed else f"\n{_failed} FAILED")
sys.exit(1 if _failed else 0)

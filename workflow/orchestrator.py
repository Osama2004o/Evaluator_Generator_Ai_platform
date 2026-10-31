"""
Evaluator–Generator feedback loop.

    question → Generator → answer → Evaluator → approved? ── yes ──► final answer
                   ▲                                │ no
                   └─────────── feedback ───────────┘   (max 4 loops)

One iteration is an LCEL chain (generate | evaluate). The loop repeats it
until the Evaluator approves the answer or MAX_REFINEMENT_LOOPS is reached.
"""

import logging
import uuid

from langchain_core.runnables import RunnableLambda

from config.settings import MAX_REFINEMENT_LOOPS

logger = logging.getLogger(__name__)

NOT_VALIDATED_NOTE = "This answer could not be fully validated. Please verify it against the sources."


class EvaluatorGeneratorWorkflow:
    def __init__(self, generator, evaluator, max_loops: int = MAX_REFINEMENT_LOOPS):
        if not 1 <= max_loops <= 4:
            raise ValueError("max_loops must be between 1 and 4.")
        self.generator = generator
        self.evaluator = evaluator
        self.max_loops = max_loops
        self.step = RunnableLambda(self._generate) | RunnableLambda(self._evaluate)

    @classmethod
    def from_knowledge_base(cls, knowledge_pipeline, session_id: str | None = None):
        """Create both agents for one session; each keeps its own isolated memory."""
        from core.evaluator.evaluator_agent import EvaluatorAgent
        from core.generator.generator_agent import GeneratorAgent

        session_id = session_id or str(uuid.uuid4())
        return cls(
            GeneratorAgent(knowledge_pipeline, session_id=session_id),
            EvaluatorAgent(session_id=session_id),
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def run(self, query: str) -> dict:
        state = {"query": query, "loop": 0, "answer": None, "feedback": None, "sources": [], "history": []}
        if not isinstance(query, str) or not query.strip():
            return self._finish(state, "invalid_input", error="Please enter a question.")
        state["query"] = query.strip()

        for loop in range(1, self.max_loops + 1):
            state["loop"] = loop
            try:
                state = self.step.invoke(state)
            except Exception as exc:
                logger.exception("Loop %d failed", loop)
                return self._finish(state, "error", error=str(exc))

            if state.get("error"):
                return self._finish(state, "error", error=state["error"])
            if state["approved"]:
                return self._finish(state, "approved")

        return self._finish(state, "max_loops_reached")

    # ──────────────────────────────────────────────
    # LCEL steps
    # ──────────────────────────────────────────────

    def _generate(self, state: dict) -> dict:
        result = self.generator.generate(
            query=state["query"],
            loop=state["loop"],
            evaluator_feedback=state["feedback"],
            previous_answer=state["answer"],
        )
        answer = (result.get("answer") or "").strip()
        if not answer or answer.startswith("[Generator Error]"):
            self._clear_answer_cache()
            raise RuntimeError(answer or "Generator returned an empty answer.")

        logger.info("Loop %d: answer generated (cached=%s)", state["loop"], result.get("cached", False))
        return {
            **state,
            "answer": answer,
            "sources": result.get("sources", []),
            "context": result.get("retrieved_context", ""),
        }

    def _evaluate(self, state: dict) -> dict:
        try:
            evaluation = self.evaluator.evaluate(
                query=state["query"],
                answer=state["answer"],
                retrieved_context=state["context"],
                loop=state["loop"],
            )
        except Exception as exc:
            # Keep the new draft instead of losing it with the failed chain
            logger.exception("Loop %d: Evaluator failed", state["loop"])
            return {**state, "approved": False, "error": str(exc)}

        feedback = evaluation.feedback
        if evaluation.unsupported_claims:
            feedback += "\nRemove these unsupported claims:\n" + "\n".join(
                f"- {claim}" for claim in evaluation.unsupported_claims
            )

        logger.info("Loop %d: %s", state["loop"], "APPROVED" if evaluation.is_acceptable else "REJECTED")
        return {
            **state,
            "approved": evaluation.is_acceptable,
            "feedback": feedback,
            "history": state["history"] + [{"loop": state["loop"], "answer": state["answer"], **evaluation.model_dump()}],
        }

    def _clear_answer_cache(self):
        """The Generator caches its error text as an answer; remove it so it isn't replayed."""
        cache = getattr(self.generator, "cache", None)
        if cache:
            cache.flush_response_cache()
            cache.flush_semantic_cache()

    # ──────────────────────────────────────────────
    # Result
    # ──────────────────────────────────────────────

    def _finish(self, state: dict, status: str, error: str | None = None) -> dict:
        approved = status == "approved"
        fallback = error if status == "invalid_input" else "Sorry, no answer could be generated."
        return {
            "answer": state["answer"] or fallback,
            "approved": approved,
            "status": status,
            "note": None if approved or not state["answer"] else NOT_VALIDATED_NOTE,
            "error": error,
            "loops_used": state["loop"],
            "sources": state["sources"],
            "history": state["history"],
            "session_id": self.generator.session_id,
        }

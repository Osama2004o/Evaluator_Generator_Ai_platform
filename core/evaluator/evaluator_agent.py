import logging
import uuid
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    MAX_OUTPUT_TOKENS,
)
from core.evaluator.prompts import EVALUATOR_PROMPT
from core.evaluator.schemas import EvaluationResult
from core.memory.evaluator_memory import EvaluatorMemory

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.memory = EvaluatorMemory(self.session_id)

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        self.parser = PydanticOutputParser(pydantic_object=EvaluationResult)
        self.chain = (
            EVALUATOR_PROMPT.partial(
                format_instructions=self.parser.get_format_instructions()
            )
            | self.llm
            | self.parser
        )

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_context: str,
        loop: int = 1,
    ) -> EvaluationResult:
        """Return an evidence-based verdict and record it in Evaluator memory."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The original query must be a nonempty string.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The generated answer must be a nonempty string.")
        if not isinstance(retrieved_context, str):
            raise ValueError("retrieved_context must be a string.")
        if loop < 1 or loop > 4:
            raise ValueError("loop must be between 1 and 4.")

        try:
            result: EvaluationResult = self.chain.invoke(
                {
                    "query": query.strip(),
                    "answer": answer,
                    "retrieved_context": retrieved_context or "(No retrieved context found.)",
                    "loop": loop,
                }
            )
        except Exception as exc:
            logger.exception("Evaluator call or structured-output parsing failed")
            raise RuntimeError("Evaluator LLM failed; answer cannot be validated") from exc

        # Deterministic minimum quality gate in addition to the model's verdict.
        passes_checks = (
            result.accuracy >= 8
            and result.relevance >= 8
            and result.completeness >= 7
            and result.consistency >= 8
            and result.grounding >= 8
            and not result.unsupported_claims
        )
        approved = result.is_acceptable and passes_checks
        if approved != result.is_acceptable or (not approved and not result.feedback.strip()):
            if result.is_acceptable and not passes_checks:
                feedback = (
                    "The quality gate rejected this draft. Recheck accuracy, relevance, "
                    "completeness, consistency, and grounding against the retrieved "
                    "context; remove any unsupported claims."
                )
            else:
                feedback = result.feedback.strip() or (
                    "Improve accuracy, relevance, completeness, consistency, and grounding "
                    "using only the retrieved context. Remove unsupported claims."
                )
            result = result.model_copy(
                update={"is_acceptable": approved, "feedback": feedback}
            )

        saved = self.memory.add_evaluation(
            query=query,
            answer=answer,
            retrieved_context=retrieved_context,
            loop=loop,
            evaluation=result,
        )
        if not saved:
            logger.warning("Evaluation completed, but Evaluator memory could not be saved")

        return result


"""
Generator Agent — LCEL pipeline with Redis caching and isolated memory.

Flow:
  1. Check response cache (Loop 1 only)
  2. Retrieve context from ChromaDB (with retrieval caching)
  3. Build prompt (initial or refinement)
  4. Call LLM via OpenRouter
  5. Cache response & save turn to Generator memory
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    GENERATION_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    TOP_K_RESULTS,
    MAX_REFINEMENT_LOOPS,
)
from core.cache.cache_manager import CacheManager
from core.memory.generator_memory import GeneratorMemory
from core.generator.prompts import (
    INITIAL_SYSTEM_PROMPT,
    INITIAL_HUMAN_TEMPLATE,
    REFINEMENT_SYSTEM_PROMPT,
    REFINEMENT_HUMAN_TEMPLATE,
)
from database.chroma_pipeline import KnowledgeBasePipeline

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """
    End-to-end Generator with:
      - Redis-backed retrieval & response caching (Memurai on Windows)
      - Isolated per-session memory
      - Strict grounding prompts
      - Iterative refinement support (for Evaluator feedback loops)
    """

    def __init__(
        self,
        knowledge_pipeline: KnowledgeBasePipeline,
        session_id: Optional[str] = None,
    ):
        self.kb = knowledge_pipeline
        self.session_id = session_id or str(uuid.uuid4())
        self.cache = CacheManager()
        self.memory = GeneratorMemory(self.session_id)

        # LLM via OpenRouter
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        self.parser = StrOutputParser()

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def generate(
        self,
        query: str,
        loop: int = 1,
        evaluator_feedback: Optional[str] = None,
        previous_answer: Optional[str] = None,
        top_k: int = TOP_K_RESULTS,
    ) -> Dict[str, Any]:
        """
        Main entry point.

        Parameters
        ----------
        query : str
            The user question.
        loop : int
            Current refinement iteration (1 = initial, 2..MAX = refinement).
        evaluator_feedback : str | None
            Feedback from the Evaluator (required when loop > 1).
        previous_answer : str | None
            The draft answer to revise (required when loop > 1).
        top_k : int
            Number of context chunks to retrieve.

        Returns
        -------
        dict with keys: answer, sources, loop, cached, session_id
        """
        query = query.strip()
        if not query:
            return self._empty_result("Empty query provided.")

        # ── Step 1: Response cache check (Loop 1 only) ──
        if loop == 1:
            cached_resp = self._check_response_cache(query, top_k)
            if cached_resp is not None:
                return cached_resp

        # ── Step 2: Retrieve context (with retrieval caching) ──
        context_results = self._retrieve_context(query, top_k)
        context_text, context_snippets, sources = self._format_context(context_results)

        context_hash = CacheManager.hash_context(
            [r.get("content", "") for r in context_results]
        )

        # ── Step 3: Build messages ──
        history_text = self.memory.format_for_prompt(last_n=5)

        if loop > 1 and evaluator_feedback:
            messages = self._build_refinement_messages(
                query=query,
                context=context_text,
                history=history_text,
                previous_answer=previous_answer or "",
                evaluator_feedback=evaluator_feedback,
            )
        else:
            messages = self._build_initial_messages(
                query=query,
                context=context_text,
                history=history_text,
            )

        # ── Step 4: Call LLM ──
        answer = self._call_llm(messages)

        # ── Step 5: Cache response (Loop 1 only) & save memory ──
        result = {
            "answer": answer,
            "sources": sources,
            "retrieved_context": context_text, 
            "loop": loop,
            "cached": False,
            "session_id": self.session_id,
        }

        if loop == 1:
            self.cache.set_response_cache(query, context_hash, result)
            self.cache.set_semantic_response(query, result)

        self.memory.add_turn(
            query=query,
            answer=answer,
            context_snippets=context_snippets,
            sources=sources,
            loop=loop,
            evaluator_feedback=evaluator_feedback,
        )

        return result

    # ──────────────────────────────────────────────
    # Step helpers
    # ──────────────────────────────────────────────

    def _check_response_cache(self, query: str, top_k: int) -> Optional[Dict[str, Any]]:
        """
        Attempt to return a cached response for the initial loop.
        Tier 1: Exact hash cache lookup (O(1))
        Tier 2: Semantic vector cache lookup (Memurai vectorset VSIM)
        """
        # Tier 1: Exact cache
        context_results = self._retrieve_context(query, top_k)
        context_hash = CacheManager.hash_context(
            [r.get("content", "") for r in context_results]
        )
        cached = self.cache.get_response_cache(query, context_hash)
        if cached is not None:
            cached["cached"] = True
            cached["cache_type"] = "exact"
            logger.info("Returning exact cached response for query: %.80s…", query)
            return cached

        # Tier 2: Native semantic cache
        semantic_cached = self.cache.get_semantic_response(query)
        if semantic_cached is not None:
            logger.info(
                "Returning semantic cached response (similarity=%.4f) for query: %.80s…",
                semantic_cached.get("similarity_score", 1.0),
                query,
            )
            return semantic_cached

        return None

    def _retrieve_context(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from ChromaDB with Redis retrieval caching."""
        collection_name = self.kb.collection_name

        # Check retrieval cache
        cached_chunks = self.cache.get_retrieval_cache(query, top_k, collection_name)
        if cached_chunks is not None:
            return cached_chunks

        # Live retrieval from ChromaDB
        results = self.kb.retrieve(query=query, k=top_k)

        # Cache the retrieval results
        self.cache.set_retrieval_cache(query, top_k, results, collection_name)

        return results

    @staticmethod
    def _format_context(
        results: List[Dict[str, Any]],
    ) -> tuple[str, List[str], List[str]]:
        """
        Formats raw retrieval results into:
          - context_text:     full text block for the prompt
          - context_snippets: abbreviated snippets for memory storage
          - sources:          unique source identifiers
        """
        if not results:
            return "(No relevant context was found in the knowledge base.)", [], []

        blocks: List[str] = []
        snippets: List[str] = []
        sources_set: set[str] = set()

        for i, r in enumerate(results, 1):
            content = r.get("content", "")
            meta = r.get("metadata", {})
            source = meta.get("source", "unknown")
            score = r.get("similarity_score", 0.0)

            blocks.append(
                f"[Chunk {i} | source: {source} | relevance: {score:.2f}]\n{content}"
            )
            snippets.append(content[:200])
            sources_set.add(source)

        context_text = "\n\n---\n\n".join(blocks)
        return context_text, snippets, sorted(sources_set)

    def _build_initial_messages(
        self, query: str, context: str, history: str
    ) -> list:
        human_content = INITIAL_HUMAN_TEMPLATE.format(
            context=context, history=history, query=query
        )
        return [
            SystemMessage(content=INITIAL_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

    def _build_refinement_messages(
        self,
        query: str,
        context: str,
        history: str,
        previous_answer: str,
        evaluator_feedback: str,
    ) -> list:
        human_content = REFINEMENT_HUMAN_TEMPLATE.format(
            context=context,
            previous_answer=previous_answer,
            evaluator_feedback=evaluator_feedback,
            history=history,
            query=query,
        )
        return [
            SystemMessage(content=REFINEMENT_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

    def _call_llm(self, messages: list) -> str:
        """Invoke the LLM and parse the output."""
        try:
            response = self.llm.invoke(messages)
            return self.parser.invoke(response)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return f"[Generator Error] LLM invocation failed: {exc}"

    @staticmethod
    def _empty_result(message: str) -> Dict[str, Any]:
        return {
            "answer": message,
            "sources": [],
            "loop": 0,
            "cached": False,
            "session_id": "",
        }


"""
tests/test_generator_pipeline.py
─────────────────────────────────
End-to-end test suite for the Generator + Redis Caching pipeline.

Steps:
  1. Redis / Memurai connectivity
  2. Ingest test_sample.txt into ChromaDB
  3. Retrieval caching (miss → hit speed comparison)
  4. Generator initial generation (Loop 1 — cache miss)
  5. Generator cached replay   (Loop 1 — cache hit, instant)
  6. Generator refinement      (Loop 2 — evaluator feedback, cache bypassed)
  7. Generator grounding test  (out-of-domain query)
  8. Memory inspection in Redis
  9. Direct Redis key inspection (shows you what to look for with memurai-cli)

Run:
    cd d:\Courses\nlp cellula\Evaluator_Generator_Ai_platform
    python tests/test_generator_pipeline.py
"""

import json
import logging
import os
import sys
import time

# ── Fix Windows console encoding for Unicode ──
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Make project root importable ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-32s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_pipeline")

# ── Counters ──
_passed = 0
_failed = 0


def divider(title: str):
    print(f"\n{'═' * 74}")
    print(f"  {title}")
    print(f"{'═' * 74}")


def check(label: str, condition: bool, detail: str = ""):
    global _passed, _failed
    icon = "✅" if condition else "❌"
    msg = f"   {icon}  {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    if condition:
        _passed += 1
    else:
        _failed += 1


def main():
    # ══════════════════════════════════════════════
    # TEST 1 — Redis / Memurai Connectivity
    # ══════════════════════════════════════════════
    divider("TEST 1 — Redis / Memurai Connectivity")

    from core.cache.redis_client import get_redis_client, is_redis_available

    redis_ok = is_redis_available()
    check("Memurai reachable on localhost:6379", redis_ok)

    if redis_ok:
        r = get_redis_client()
        info = r.info("server")
        print(f"       Redis version : {info.get('redis_version', 'N/A')}")
        print(f"       Uptime (sec)  : {info.get('uptime_in_seconds', 'N/A')}")

    # ══════════════════════════════════════════════
    # TEST 2 — Ingest test_sample.txt into ChromaDB
    # ══════════════════════════════════════════════
    divider("TEST 2 — Ingest test_sample.txt into ChromaDB")

    from database.chroma_pipeline import KnowledgeBasePipeline

    kb = KnowledgeBasePipeline()

    sample_path = os.path.join(PROJECT_ROOT, "test_sample.txt")
    check("test_sample.txt exists", os.path.isfile(sample_path), sample_path)

    result = kb.ingest(sample_path)
    chunks_count = result.get("chunks_ingested", 0)
    check("Ingestion succeeded", result["status"] == "success", f"{chunks_count} chunks ingested")

    stats = kb.get_stats()
    print(f"       Collection    : {stats['collection_name']}")
    print(f"       Total chunks  : {stats['total_chunks']}")
    print(f"       Sources       : {stats['unique_sources_count']}")

    # ══════════════════════════════════════════════
    # TEST 3 — Retrieval Caching (miss → hit)
    # ══════════════════════════════════════════════
    divider("TEST 3 — Retrieval Caching (miss → hit)")

    from core.cache.cache_manager import CacheManager

    cache = CacheManager()

    # Flush old retrieval cache to get a clean miss
    cache.flush_retrieval_cache()

    test_query = "What is the Singleton design pattern?"

    # First call → cache miss (hits ChromaDB)
    t0 = time.perf_counter()
    results_miss = kb.retrieve(query=test_query, k=3)
    t_miss = (time.perf_counter() - t0) * 1000

    # Manually cache the results (simulating what GeneratorAgent does)
    cache.set_retrieval_cache(test_query, 3, results_miss, kb.collection_name)

    # Second call → cache hit (from Redis)
    t0 = time.perf_counter()
    results_hit = cache.get_retrieval_cache(test_query, 3, kb.collection_name)
    t_hit = (time.perf_counter() - t0) * 1000

    check("Retrieval cache MISS then HIT", results_hit is not None)
    check(
        f"Cache speedup: {t_miss:.1f}ms → {t_hit:.1f}ms",
        t_hit < t_miss,
        f"{t_miss / max(t_hit, 0.001):.0f}x faster",
    )

    # ══════════════════════════════════════════════
    # TEST 4 — Generator Initial Generation (Loop 1, cache miss)
    # ══════════════════════════════════════════════
    divider("TEST 4 — Generator Initial Generation (Loop 1)")

    from core.generator import GeneratorAgent

    session_id = "test-session-pipeline"
    gen = GeneratorAgent(knowledge_pipeline=kb, session_id=session_id)

    # Flush response cache to force a fresh LLM call
    cache.flush_response_cache()
    cache.flush_semantic_cache()

    t0 = time.perf_counter()
    gen_result = gen.generate(query=test_query, loop=1)
    t_gen = (time.perf_counter() - t0) * 1000

    check("Generation returned an answer", len(gen_result["answer"]) > 20)
    check("Marked as NOT cached", gen_result["cached"] is False)
    check("Loop = 1", gen_result["loop"] == 1)
    print(f"       Time           : {t_gen:.0f} ms")
    print(f"       Sources        : {gen_result['sources']}")
    print(f"       Answer preview : {gen_result['answer'][:300]}…")

    # ══════════════════════════════════════════════
    # TEST 5 — Exact Cached Response Replay (same query)
    # ══════════════════════════════════════════════
    divider("TEST 5 — Exact Cached Response Replay (same query)")

    t0 = time.perf_counter()
    cached_result = gen.generate(query=test_query, loop=1)
    t_cached = (time.perf_counter() - t0) * 1000

    check("Response served from cache", cached_result["cached"] is True)
    check(
        f"Instant: {t_cached:.0f}ms vs original {t_gen:.0f}ms",
        t_cached < t_gen,
    )

    # ══════════════════════════════════════════════
    # TEST 5B — Semantic Cache Replay (equivalent query via Memurai VSIM)
    # ══════════════════════════════════════════════
    divider("TEST 5B — Semantic Cache Replay (equivalent query via Memurai VSIM)")

    equivalent_query = "Can you describe the Singleton pattern in Python?"
    t0 = time.perf_counter()
    semantic_result = gen.generate(query=equivalent_query, loop=1)
    t_sem = (time.perf_counter() - t0) * 1000

    check("Equivalent query served from cache", semantic_result["cached"] is True)
    check("Identified as semantic cache hit", semantic_result.get("cache_type") == "semantic")
    sim = semantic_result.get("similarity_score", 0.0)
    check("Semantic similarity score >= 0.80", sim >= 0.80, f"Score: {sim:.4f}")
    print(f"       Time           : {t_sem:.0f} ms")
    print(f"       Similarity     : {sim:.4f}")

    # ══════════════════════════════════════════════
    # TEST 6 — Refinement with Evaluator Feedback (Loop 2)
    # ══════════════════════════════════════════════
    divider("TEST 6 — Refinement with Evaluator Feedback (Loop 2)")

    mock_feedback = (
        "The answer is too high-level. Include the specific Python code example "
        "for the Singleton pattern from the context. Also mention the listed "
        "advantages and disadvantages."
    )

    t0 = time.perf_counter()
    refined_result = gen.generate(
        query=test_query,
        loop=2,
        evaluator_feedback=mock_feedback,
        previous_answer=gen_result["answer"],
    )
    t_refine = (time.perf_counter() - t0) * 1000

    check("Refinement returned an answer", len(refined_result["answer"]) > 20)
    check("Marked as NOT cached (bypassed)", refined_result["cached"] is False)
    check("Loop = 2", refined_result["loop"] == 2)
    print(f"       Time           : {t_refine:.0f} ms")
    print(f"       Revised preview: {refined_result['answer'][:300]}…")

    # ══════════════════════════════════════════════
    # TEST 7 — Out-of-Domain Query (grounding test)
    # ══════════════════════════════════════════════
    divider("TEST 7 — Out-of-Domain Query (Grounding / Anti-Hallucination)")

    ood_query = "What is the capital of Mars and who is the president there?"

    ood_result = gen.generate(query=ood_query, loop=1)
    answer_lower = ood_result["answer"].lower()

    # The model should indicate that info is not available
    has_disclaimer = any(
        phrase in answer_lower
        for phrase in [
            "not available",
            "not found",
            "no relevant",
            "cannot find",
            "no information",
            "not contain",
            "outside",
            "does not",
            "unable",
        ]
    )
    check(
        "Grounding: refuses to hallucinate on out-of-domain query",
        has_disclaimer,
        f"Answer: {ood_result['answer'][:200]}",
    )

    # ══════════════════════════════════════════════
    # TEST 8 — Generator Memory Inspection
    # ══════════════════════════════════════════════
    divider("TEST 8 — Generator Memory Inspection")

    history = gen.memory.get_history()
    check("Memory has recorded turns", len(history) >= 2, f"{len(history)} turns stored")

    for t in history:
        fb = t.get("evaluator_feedback")
        fb_str = f" | feedback: {fb[:50]}…" if fb else ""
        print(f"       Turn {t['turn_index']} (loop {t['loop']}) @ {t['timestamp']}{fb_str}")

    # Test format_for_prompt
    prompt_text = gen.memory.format_for_prompt(last_n=3)
    check("format_for_prompt() returns text", len(prompt_text) > 50)

    # ══════════════════════════════════════════════
    # TEST 9 — Direct Redis Key Inspection
    # ══════════════════════════════════════════════
    divider("TEST 9 — Redis Key Inspection")

    if redis_ok:
        r = get_redis_client()

        # List all project keys
        all_keys = []
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match="*", count=200)
            all_keys.extend(keys)
            if cursor == 0:
                break

        cache_keys = [k for k in all_keys if k.startswith("cache:")]
        semantic_keys = [k for k in all_keys if k.startswith("semantic_cache:")]
        memory_keys = [k for k in all_keys if k.startswith("memory:gen:")]

        print(f"       Total keys in Redis     : {len(all_keys)}")
        print(f"       Cache keys (cache:*)     : {len(cache_keys)}")
        print(f"       Semantic keys (semantic:) : {len(semantic_keys)}")
        print(f"       Memory keys (memory:*)   : {len(memory_keys)}")

        check("Cache keys exist in Redis", len(cache_keys) > 0)
        check("Semantic cache keys exist in Redis", len(semantic_keys) > 0)
        check("Generator memory key exists", len(memory_keys) > 0)

        print("\n       ── Key details ──")
        for k in sorted(cache_keys + semantic_keys + memory_keys):
            key_type = r.type(k)
            ttl = r.ttl(k)
            print(f"       {k[:70]:<70}  type={key_type:<6}  TTL={ttl}s")
    else:
        print("       ⚠️  Skipped — Redis not available.")

    # ══════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════
    divider("TEST SUMMARY")
    total = _passed + _failed
    print(f"   Passed : {_passed}/{total}")
    print(f"   Failed : {_failed}/{total}")

    if _failed == 0:
        print("\n   🎉  All tests passed! Generator + Redis caching pipeline is working.")
    else:
        print(f"\n   ⚠️  {_failed} test(s) failed. Review the output above.")

    print(f"\n   Session ID : {session_id}")
    print()

    # ── Redis CLI cheatsheet ──
    print("   ── How to inspect Redis with Memurai CLI ──")
    print('   Open a terminal and run:')
    print('     D:\\Memurai\\memurai-cli.exe')
    print()
    print('   Useful commands inside the CLI:')
    print('     PING                              → should return PONG')
    print('     KEYS *                            → list all keys')
    print('     KEYS cache:*                      → list all cache keys')
    print('     KEYS memory:gen:*                 → list generator memory keys')
    print('     GET cache:gen:resp:<hash>          → view a cached LLM response')
    print('     LRANGE memory:gen:test-session-pipeline 0 -1  → view all memory turns')
    print('     TTL <key>                         → check remaining time-to-live')
    print('     TYPE <key>                        → check key type (string/list)')
    print('     DBSIZE                            → total number of keys')
    print('     FLUSHDB                           → ⚠️  delete ALL keys in current DB')
    print()


if __name__ == "__main__":
    main()


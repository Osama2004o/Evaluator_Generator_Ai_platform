"""
Streamlit UI for the Evaluator–Generator platform.

Run (from the project root, with the venv active):
    streamlit run app.py
"""

import logging
import os
import uuid

import streamlit as st

from config.settings import MAX_REFINEMENT_LOOPS, OPENROUTER_API_KEY

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "database", "chroma_db")
SCORE_FIELDS = ("accuracy", "relevance", "completeness", "consistency", "grounding")
DOCUMENT_TYPES = ["pdf", "docx", "txt", "ppt", "pptx", "wav"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

st.set_page_config(page_title="Evaluator–Generator QA", page_icon="🧠", layout="wide")


# ──────────────────────────────────────────────
# Shared resources
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base and models…")
def get_knowledge_base():
    from database.chroma_pipeline import KnowledgeBasePipeline

    return KnowledgeBasePipeline(persist_directory=CHROMA_DIR)


@st.cache_resource
def get_cache():
    from core.cache.cache_manager import CacheManager

    return CacheManager()


def get_workflow():
    """One workflow (Generator + Evaluator with isolated memories) per chat session."""
    if "workflow" not in st.session_state:
        from workflow import EvaluatorGeneratorWorkflow

        st.session_state.workflow = EvaluatorGeneratorWorkflow.from_knowledge_base(
            get_knowledge_base(), session_id=st.session_state.session_id
        )
    return st.session_state.workflow


def clear_answer_caches():
    """The knowledge base changed, so cached retrievals and answers are stale."""
    cache = get_cache()
    cache.flush_retrieval_cache()
    cache.flush_response_cache()
    cache.flush_semantic_cache()


def short_name(source: str) -> str:
    return source if source.startswith(("http://", "https://", "wiki:")) else os.path.basename(source)


# ──────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────

def save_upload(uploaded_file) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, os.path.basename(uploaded_file.name))
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def ingest(sources: list[str]):
    kb = get_knowledge_base()
    for src in sources:
        kb.delete_source(src)  # re-ingesting replaces old chunks instead of leaving orphans

    with st.spinner(f"Ingesting {len(sources)} source(s)… (audio transcription can take a while)"):
        results = kb.ingest_batch(sources)

    for r in results:
        name = short_name(r["source"])
        if r["status"] == "success":
            st.success(f"{name}: {r['chunks_ingested']} chunks")
        elif r["status"] == "warning":
            st.warning(f"{name}: {r['message']}")
        else:
            st.error(f"{name}: {r['error']}")

    if any(r["status"] == "success" for r in results):
        clear_answer_caches()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    from core.cache.redis_client import is_redis_available

    kb = get_knowledge_base()
    from core.loaders import SOURCE_CODE_EXTENSIONS  # cheap: already imported by the knowledge base

    st.sidebar.title("🧠 Knowledge Base")

    # ── Status ──
    redis_ok = is_redis_available()
    st.sidebar.caption(
        f"{'🟢' if OPENROUTER_API_KEY else '🔴'} OpenRouter API key  ·  "
        f"{'🟢' if redis_ok else '🟠'} Redis {'connected' if redis_ok else 'offline (no cache/memory)'}"
    )
    if not OPENROUTER_API_KEY:
        st.sidebar.error("Set OPENROUTER_API_KEY in .env, then restart the app.")

    # ── Add knowledge ──
    with st.sidebar.form("add_knowledge", clear_on_submit=True):
        st.subheader("Add knowledge")
        files = st.file_uploader(
            "Files",
            type=DOCUMENT_TYPES + sorted(ext.lstrip(".") for ext in SOURCE_CODE_EXTENSIONS),
            accept_multiple_files=True,
            help="PDF, DOCX, TXT, PPT/PPTX, WAV audio, or source code",
        )
        urls = st.text_area("Web page URLs (one per line)", placeholder="https://example.com/article")
        wiki = st.text_input("Wikipedia topic or URL", placeholder="Design pattern")
        submitted = st.form_submit_button("Ingest", type="primary", use_container_width=True)

    if submitted:
        sources = [save_upload(f) for f in files or []]
        for url in (u.strip() for u in urls.splitlines() if u.strip()):
            if url.startswith(("http://", "https://")):
                sources.append(url)
            else:
                st.sidebar.warning(f"Skipped `{url}`: URLs must start with http:// or https://")
        if wiki.strip():
            topic = wiki.strip()
            sources.append(topic if "wikipedia.org/wiki/" in topic else f"wiki:{topic}")

        if sources:
            with st.sidebar:
                ingest(sources)
        else:
            st.sidebar.warning("Add at least one file, URL, or Wikipedia topic.")

    # ── Stored sources ──
    sources = kb.list_sources()
    st.sidebar.subheader(f"Sources ({len(sources)})")
    st.sidebar.caption(f"{kb.collection.count()} chunks stored")
    for src in sources:
        col_name, col_del = st.sidebar.columns([5, 1])
        col_name.markdown(f"`{short_name(src)}`", help=src)
        if col_del.button("🗑", key=f"del_{src}", help="Remove from knowledge base"):
            kb.delete_source(src)
            if src.startswith(UPLOAD_DIR) and os.path.isfile(src):
                os.remove(src)
            clear_answer_caches()
            st.rerun()

    if sources and st.sidebar.button("Clear knowledge base", use_container_width=True):
        kb.clear()
        clear_answer_caches()
        st.rerun()

    # ── Session ──
    st.sidebar.divider()
    st.sidebar.caption(f"Session: `{st.session_state.session_id[:8]}`")
    if st.sidebar.button("New chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ──────────────────────────────────────────────
# Chat
# ──────────────────────────────────────────────

def render_result(result: dict):
    st.markdown(result["answer"])

    if result["approved"]:
        st.success(f"Approved by the Evaluator after {result['loops_used']} of {MAX_REFINEMENT_LOOPS} loop(s).", icon="✅")
    elif result["note"]:
        st.warning(result["note"], icon="⚠️")

    if result["sources"]:
        st.caption("Sources: " + " · ".join(short_name(s) for s in result["sources"]))

    if result["error"] and result["status"] != "invalid_input":
        st.caption(f"Error details: {result['error']}")

    history = result["history"]
    if not history:
        return
    with st.expander(f"Evaluation loops ({len(history)})"):
        tabs = st.tabs([f"Loop {h['loop']} {'✅' if h['is_acceptable'] else '❌'}" for h in history])
        for tab, h in zip(tabs, history):
            with tab:
                for col, field in zip(st.columns(len(SCORE_FIELDS)), SCORE_FIELDS):
                    col.metric(field.title(), f"{h[field]}/10")
                if h["unsupported_claims"]:
                    st.markdown("**Unsupported claims:**\n" + "\n".join(f"- {c}" for c in h["unsupported_claims"]))
                if h["feedback"] and not h["is_acceptable"]:
                    st.info(f"**Evaluator feedback:** {h['feedback']}")
                with st.container(border=True):
                    st.markdown(h["answer"])


def main():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []

    render_sidebar()

    st.title("Evaluator–Generator Q&A")
    st.caption(
        f"Answers are generated from your knowledge base, checked by an independent Evaluator, "
        f"and refined up to {MAX_REFINEMENT_LOOPS} times."
    )

    if get_knowledge_base().collection.count() == 0:
        st.info("The knowledge base is empty. Add files, URLs, or Wikipedia topics from the sidebar first.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_result(msg["result"])

    question = st.chat_input("Ask a question about your knowledge base…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating and evaluating the answer…"):
            result = get_workflow().run(question)
        render_result(result)
    st.session_state.messages.append({"role": "assistant", "result": result})


main()

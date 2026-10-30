import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/free")
GENERATION_TEMPERATURE = 0.5
MAX_OUTPUT_TOKENS = 8192
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL
EMBEDDING_DEVICE = "cuda"

CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "chroma_db"
)
CHROMA_COLLECTION_NAME = "code_knowledge_base"

TOP_K_RESULTS = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

MAX_MEMORY_TURNS = 20
CODE_EXECUTION_TIMEOUT = 10

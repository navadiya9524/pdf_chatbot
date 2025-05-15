from pathlib import Path
import os

# ------------------- Base Directory Paths -------------------

# Root directory where data files will be stored or referenced
BASE_DIR = Path("data")

# Directory path to store FAISS index files (ensure this path exists in repo or is created at runtime)
FAISS_INDEX_DIR = Path("api/features_apis/pdf_chatbot/data/faiss_index")

# ------------------- Chunking Configuration -------------------

# Number of overlapping characters between two consecutive text chunks
# Helps retain context across chunk boundaries
CHUNK_OVERLAP = 50

# ------------------- Embedding Model Settings -------------------

# Name or path of the embedding model used for vector representation of text
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "your-default-embedding-model")

# ------------------- LLM (Language Model) Configuration -------------------

# Name of the LLM used for generating answers.
LLM_MODEL = os.getenv("LLM_MODEL", "your-llm-model-name")

# Temperature controls randomness/creativity:
# - 0.0 = deterministic
# - ~0.7 = balanced
# - 1.0+ = very random
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Number of top-k vectors to retrieve during similarity search
TOP_K = int(os.getenv("TOP_K", "5"))

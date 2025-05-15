from api.features_apis.pdf_chatbot.config import FAISS_INDEX_DIR
from api.features_apis.pdf_chatbot.model_config import get_embedding
import os
import logging
import numpy as np
import faiss

# Set a name for logging context
name = "chunk_to_index"
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - {name} - %(message)s'
)
logger = logging.getLogger(__name__)

# Load embedding model to generate vector representations
EMBEDDING_MODEL = get_embedding()

# Ensure FAISS index directory exists
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)


def load_faiss_index(faiss_index_path: str, user_id: str,
                     chunk_embeddings: np.ndarray) -> faiss.IndexIDMap:
    """
    Load an existing FAISS index if available; otherwise, create a new one.

    Args:
        faiss_index_path (str): Path to the FAISS index file
        user_id (str): Unique identifier for the user
        chunk_embeddings (np.ndarray): Embeddings for new chunks

    Returns:
        faiss.IndexIDMap: A FAISS index map object ready to store vectors
    """
    if os.path.exists(faiss_index_path):
        logger.info(f"Existing FAISS index found for user {user_id}. Loading.")
        index = faiss.read_index(faiss_index_path)

        if not isinstance(index, faiss.IndexIDMap):
            logger.warning("Loaded index is not an IndexIDMap. Wrapping it.")
            index = faiss.IndexIDMap(index)

        return index
    else:
        logger.info(f"No FAISS index found for user {user_id}. Creating new one.")
        dim = chunk_embeddings.shape[1]  # Dimension of embeddings
        base_index = faiss.IndexFlatIP(dim)  # Create base index using inner product
        id_index = faiss.IndexIDMap(base_index)  # Wrap with ID map for ID-based indexing
        return id_index


def process_chunk_to_faiss_index(chunk_data: list, user_id) -> str:
    """
    Convert chunks to embeddings and add them to a FAISS index.

    Args:
        chunk_data (list): List of dicts containing chunk text and unique IDs
        user_id (str): Identifier for the user

    Returns:
        str: Path to the saved FAISS index file
    """
    try:
        faiss_index_path = str(FAISS_INDEX_DIR / f"{user_id}_index.faiss")

        # Extract texts and IDs from chunk data
        chunk_texts = [item["chunk_text"] for item in chunk_data]
        unique_ids = [item["unique_id"] for item in chunk_data]

        # Generate embeddings for chunk texts
        chunk_embeddings = EMBEDDING_MODEL.embed_documents(chunk_texts)
        chunk_embeddings = np.array(chunk_embeddings).astype(np.float32)

        # Normalize embeddings for inner product similarity
        faiss.normalize_L2(chunk_embeddings)

        # Convert IDs to int64 format for FAISS
        int_ids = np.array(unique_ids).astype(np.int64)

        # Load or create FAISS index and add new vectors
        id_index = load_faiss_index(faiss_index_path, user_id, chunk_embeddings)
        id_index.add_with_ids(chunk_embeddings, int_ids)

        # Persist the updated index to disk
        faiss.write_index(id_index, faiss_index_path)

        logger.info(f"FAISS index created and saved to: {faiss_index_path}")
        return faiss_index_path

    except Exception as e:
        logger.error(f"Failed to process chunk data and save FAISS index: {e}")
        raise

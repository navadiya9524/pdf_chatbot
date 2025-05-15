import os
import faiss
import numpy as np
from api.features_apis.pdf_chatbot.config import FAISS_INDEX_DIR
import logging

# Logger setup for consistent structured logging
name = "delete_vectors_from_faiss_api"
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - {name} - %(message)s'
)
logger = logging.getLogger(__name__)


def get_faiss_index_path(user_id):
    """
    Construct the full path to the FAISS index file for a given user.

    Args:
        user_id (str): Unique identifier for the user.

    Returns:
        str: Path to the user's FAISS index file.
    """
    return str(FAISS_INDEX_DIR / f"{user_id}_index.faiss")


def delete_vectors_from_faiss(data):
    """
    Deletes specific vector IDs from a user's FAISS index.

    Args:
        data (dict): A dictionary containing 'user_id' and 'chunk_ids' to delete.

    Returns:
        tuple: Response dict and status code.
    """
    try:
        user_id = data.get("user_id")
        chunk_ids = data.get("chunk_ids", [])

        # Validate input data
        if not user_id or not chunk_ids:
            logger.error("Missing user_id or chunk_ids")
            return {"error": "Missing user_id or chunk_ids"}, 400

        # Construct path to index and check existence
        index_path = str(FAISS_INDEX_DIR / f"{user_id}_index.faiss")
        if not os.path.exists(index_path):
            logger.error(f"Index for user '{user_id}' not found at {index_path}")
            return {"error": f"Index for user '{user_id}' not found"}, 404

        # Load FAISS index
        index = faiss.read_index(index_path)
        if not isinstance(index, faiss.IndexIDMap):
            logger.warning("Loaded index is not an IndexIDMap. Wrapping it.")
            index = faiss.IndexIDMap(index)

        # Create ID selector for deletion
        int_ids_to_delete = np.array(chunk_ids, dtype=np.int64)
        selector = faiss.IDSelectorBatch(int_ids_to_delete)

        # Perform the deletion
        removed_count = index.remove_ids(selector)

        # Persist the updated index
        faiss.write_index(index, index_path)

        return {
            "message": "Vectors deleted successfully",
            "deleted_count": removed_count,
            "remaining_vectors": index.ntotal,
            "deleted_chunk_ids": chunk_ids
        }, 200

    except Exception as e:
        logger.error(f"Exception during vector deletion: {e}")
        return {"error": str(e)}, 500

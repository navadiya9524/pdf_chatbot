import os
import faiss
import logging
import numpy as np
import traceback
import requests
import json
from typing import List, Dict, Any
from api.features_apis.pdf_chatbot.model_config import get_embedding, get_llm_model
from api.features_apis.pdf_chatbot.prompt import rephrase_prompt, cot_prompt
from api.features_apis.pdf_chatbot.config import TOP_K, FAISS_INDEX_DIR

# Set a name for logging context
name = "question_answer"
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - {name} - %(message)s'
)
logger = logging.getLogger(__name__)

# Load embedding and LLM models
EMBEDDING_MODEL = get_embedding()
LLM_MODEL = get_llm_model()


def fetch_chunks(unique_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Call the Backend API to fetch chunk's metadata from MongoDB using unique_ids.

    Args:
        unique_ids (List[str]): List of chunk IDs to fetch metadata for

    Returns:
        List[Dict[str, Any]]: List of chunk metadata dictionaries
    """
    url = ""

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {BEARER_TOKEN}',
        }
    payload = json.dumps({
        "chunkIds": unique_ids
    })

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to call API: {str(e)}")
        return []
    if data.get('status') == 200:
        response = data.get('data')
        return response
    else:
        logger.error(f"API response status code: {data.get('status')}")
        return []


def process_question_answer(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Search on FAISS vector index and generate answer using Gemini LLM.

    Args:
        data (dict): Dictionary containing user_id, question, and optional chat_history

    Returns:
        dict: Generated answer and metadata or error message
    """

    user_id = data.get("user_id")
    question = data.get("question")

    if not user_id or not question:
        logger.error("Missing required parameters: user_id, question")
        return {"error": "Missing required parameters: user_id, question"}, 400

    chat_history = data.get("chat_history", [])

    # Construct path to user's FAISS index file
    faiss_index_path = str(FAISS_INDEX_DIR / f"{user_id}_index.faiss")

    try:
        # Ensure FAISS index exists
        if not os.path.exists(faiss_index_path):
            logger.error(f"FAISS index not found at: {faiss_index_path}")
            return {"error": "FAISS index not found"}, 404

        # Load FAISS index
        raw_index = faiss.read_index(faiss_index_path)
        total_vectors = raw_index.ntotal
        logger.info(f"Loaded FAISS index with {total_vectors} vectors")

        # Step 1: Format chat history for rephrasing
        formatted_history = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in chat_history]
        )

        # Step 2: Rephrase question based on chat history
        rephrase_input = rephrase_prompt.format(
            question=question,
            chat_history=formatted_history
        )
        rephrase_output = LLM_MODEL.invoke(rephrase_input).strip()
        logger.info(f"Rephrase Output: {rephrase_output}")

        if rephrase_output.startswith("REPHRASED:"):
            final_question = rephrase_output[len("REPHRASED:"):].strip()
        elif rephrase_output.startswith("UNCHANGED:"):
            final_question = rephrase_output[len("UNCHANGED:"):].strip()
        else:
            logger.warning("Unexpected rephrase format. Falling back to original question.")
            final_question = question

        logger.info(f"Final Question Used: {final_question}")

        # Step 3: Embed question and search in FAISS index
        query_vector = EMBEDDING_MODEL.embed_query(final_question)
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)

        top_k = min(TOP_K, total_vectors)
        _, indices = raw_index.search(query_vector, top_k)
        vector_ids = indices.flatten().tolist()
        logger.info(f"Top {top_k} FAISS vector IDs: {vector_ids}")

        # Step 4: Retrieve corresponding chunks using vector IDs
        chunk_data = fetch_chunks(vector_ids)
        if not chunk_data:
            logger.warning("API returned no chunk metadata.")
            return {"error": "No chunk metadata found"}, 404

        context = "\n\n".join([doc.get('chunk') for doc in chunk_data])

        # Step 5: Generate final answer using the context and question
        cot_input = cot_prompt.format(
            context=context,
            question=final_question
        )
        answer = LLM_MODEL.invoke(cot_input)

        return {
            "answer": answer,
            "chunk_ids": vector_ids
        }, 200

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Query processing failed: {str(e)}"}, 404

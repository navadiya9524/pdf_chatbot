from flask import Flask, request, jsonify
import logging
import traceback
import faiss
import numpy as np
import os

# Required imports from internal modules
from api.features_apis.pdf_chatbot.config import FAISS_INDEX_DIR
from api.features_apis.pdf_chatbot.pdf_to_chunk import process_pdf_to_chunk
from api.features_apis.pdf_chatbot.chunk_to_faiss import process_chunk_to_faiss_index
from api.features_apis.pdf_chatbot.question_answer import process_question_answer

app = Flask(__name__)

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/pdf_to_chunk', methods=['POST'])
def pdf_to_chunk():
    """
    POST /pdf_to_chunk
    - Accept a JSON payload with `file_url`, `user_id`, and `file_id`.
    - Download and extract text + links from PDF.
    - Perform token-based chunking using a tokenizer.
    - Return the structured chunks and metadata.
    """
    # process_pdf_to_chunk Function goes here
    ...


@app.route('/chunk_to_index', methods=['POST'])
def chunk_to_index():
    """
    POST /chunk_to_index
    - Receive `chunk_data` and `user_id`.
    - Generate vector embeddings from chunks.
    - Store vectors in a FAISS index with ID mapping.
    """
    # process_chunk_to_faiss_index Function goes here
    ...


@app.route('/query', methods=['POST'])
def query():
    """
    POST /query
    - Accept a `user_id`, `question`, and optional `chat_history`.
    - Search for similar chunks in FAISS index.
    - Use LLM to generate an answer based on context + question.
    """
    # process_question_answer Function goes here
    ...


def get_faiss_index_path(user_id):
    """
    Helper function to get FAISS index path for a given user.
    """
    return str(FAISS_INDEX_DIR / f"{user_id}_index.faiss")


@app.route('/delete_vectors', methods=['POST'])
def delete_vectors():
    """
    POST /delete_vectors
    - Accept `user_id` and `chunk_ids` in JSON payload.
    - Locate the corresponding FAISS index.
    - Delete vector IDs using FAISS IDSelectorBatch.
    - Return number of vectors deleted.
    """
    # delete_vectors_from_faiss Function goes here
    ...


if __name__ == '__main__':
    # Start Flask server (debug disabled for production safety)
    app.run(debug=False)

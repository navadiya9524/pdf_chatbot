from flask import Flask, request, jsonify
import logging
import traceback
import faiss
import numpy as np
import os
from api.features_apis.pdf_chatbot.config import FAISS_INDEX_DIR
from api.features_apis.pdf_chatbot.pdf_to_chunk import process_pdf_to_chunk
from api.features_apis.pdf_chatbot.chunk_to_faiss import process_chunk_to_faiss_index
from api.features_apis.pdf_chatbot.question_answer import process_question_answer

app = Flask(__name__)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/pdf_to_chunk', methods=['POST'])
def pdf_to_chunk():
    try:
        data = request.get_json()
        result = process_pdf_to_chunk(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in pdf_to_chunk: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/chunk_to_index', methods=['POST'])
def chunk_to_index():
    try:
        data = request.get_json()
        chunks = data.get("chunk_data")
        user_id = data.get("user_id")

        if not chunks or not user_id:
            return jsonify({"error": "Missing chunk_data or user_id"}), 400
        index_path = process_chunk_to_faiss_index(chunks, user_id)

        return jsonify({
            "message": "FAISS index created successfully",
            "index_path": index_path
        }), 200

    except Exception as e:
        logger.error(f"Error in chunk_to_index: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_id = data.get('user_id')
        question = data.get('question')
        chat_history = data.get('chat_history', [])
        if not user_id or not question:
            return jsonify({"error": "Missing 'user_id' or 'question'"}), 400
        result = process_question_answer({
            "user_id": user_id,
            "question": question,
            "chat_history": chat_history
        })
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_faiss_index_path(user_id):
    return str(FAISS_INDEX_DIR / f"{user_id}_index.faiss")


@app.route('/delete_vectors', methods=['POST'])
def delete_vectors():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        chunk_ids = data.get("chunk_ids", [])

        if not user_id or not chunk_ids:
            return jsonify({"error": "Missing user_id or chunk_ids"}), 400

        index_path = get_faiss_index_path(user_id)

        if not os.path.exists(index_path):
            return jsonify({"error": f"Index for user '{user_id}' not found"}), 404
        index = faiss.read_index(index_path)
        int_ids_to_delete = np.array(chunk_ids, dtype=np.int64)
        selector = faiss.IDSelectorBatch(int_ids_to_delete)
        removed_count = index.remove_ids(selector)
        faiss.write_index(index, index_path)

        return jsonify({
            "message": "Vectors deleted successfully",
            "deleted_count": removed_count
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)

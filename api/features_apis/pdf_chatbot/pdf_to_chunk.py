import logging
import fitz  # PyMuPDF for PDF parsing
from typing import Dict
import re
import requests
import hashlib
from transformers import AutoTokenizer
from api.features_apis.pdf_chatbot.config import CHUNK_OVERLAP
from api.features_apis.pdf_chatbot.chunk_to_faiss import process_chunk_to_faiss_index

# Set logging configuration
name = "pdf_to_chunk"
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - {name} - %(message)s'
)
logger = logging.getLogger(__name__)

# Load BERT tokenizer for token-based chunking
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def string_to_int64(text: str) -> int:
    """
    Convert a string into a consistent 64-bit integer using SHA-256 hashing.
    """
    hash_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return int(hash_digest[:16], 16)


def decide_chunk_size(token_count: int) -> int:
    """
    Decide appropriate chunk size based on total number of tokens in the document.
    """
    if token_count <= 500:
        return 100
    elif token_count <= 2500:
        return 500
    elif token_count <= 7000:
        return 800
    return 1000


def extract_pdf_text_and_links_from_s3(pdf_content: bytes, s3_url: str) -> Dict:
    """
    Extract text and hyperlinks from all pages of a PDF.

    Args:
        pdf_content (bytes): PDF file content
        s3_url (str): Source URL from S3 bucket

    Returns:
        Dict: Extracted text, links, and metadata per page
    """
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            links = [link["uri"] for link in page.get_links() if link.get("uri")]
            pages.append({
                "page_number": i + 1,
                "text": text,
                "links": list(set(links))
            })
        return {
            "source": s3_url,
            "pages": pages,
            "total_pages": len(doc)
        }
    except Exception as e:
        logger.error(f"Error reading PDF from S3: {e}")
        return {}


def clean_text(text: str) -> str:
    """
    Clean raw text by removing special characters and excessive whitespace.
    """
    cleaned = re.sub(r'[^a-zA-Z0-9\s]{2,}|[^\x00-\x7F]|\s{2,}', ' ', text.strip())
    return cleaned


def chunk_text(entry: Dict) -> Dict:
    """
    Token-based global chunking of entire PDF content with metadata attachment.
    """
    try:
        all_text = ""
        page_spans = []
        token_offset = 0
        token_map = []

        # Process each page to track token spans and links
        for page in entry["pages"]:
            cleaned = clean_text(page["text"])
            tokens = tokenizer.tokenize(cleaned)
            start_token = token_offset
            end_token = start_token + len(tokens) - 1
            page_spans.append({
                "page_number": page["page_number"],
                "start_token": start_token,
                "end_token": end_token,
                "links": page.get("links", [])
            })
            token_map.extend([page["page_number"]] * len(tokens))
            token_offset += len(tokens)
            all_text += cleaned + " "

        # Tokenize full document text
        all_tokens = tokenizer.tokenize(all_text)
        total_tokens = len(all_tokens)
        chunk_size = decide_chunk_size(total_tokens)

        chunks = []
        j = 1

        # Slide window across tokens and create chunks
        for i in range(0, total_tokens, chunk_size - CHUNK_OVERLAP):
            chunk_tokens = all_tokens[i:i + chunk_size]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunk_start = i
            chunk_end = i + len(chunk_tokens) - 1
            chunk_links = set()
            chunk_pages = set()

            # Determine which pages and links the chunk spans
            for span in page_spans:
                if span["end_token"] >= chunk_start and span["start_token"] <= chunk_end:
                    chunk_links.update(span["links"])
                    chunk_pages.add(span["page_number"])

            # Generate unique ID for the chunk
            chunk_id = string_to_int64(f"{entry['file_id']}_{j}")

            chunks.append({
                "chunk_text": chunk_text,
                "chunk_tokens": len(chunk_tokens),
                "chunk_page": min(chunk_pages) if chunk_pages else 1,
                "source": entry["source"],
                "links": list(chunk_links),
                "unique_id": chunk_id
            })
            j += 1

        return {
            "total_tokens": total_tokens,
            "chunks": chunks
        }

    except Exception as e:
        logger.error(f"Error during token-based chunking: {e}")
        return {
            "total_tokens": 0,
            "chunks": []
        }


def process_pdf_to_chunk(data: str):
    """
    Full pipeline to download a PDF, extract & chunk it, and store vectors in FAISS.
    """
    file_url = data.get("file_url")
    file_id = data.get("file_id")
    user_id = data.get("user_id")

    # Validate required parameters
    if not file_url or not file_id or not user_id:
        logger.error("Missing required parameters: file_url, file_id, user_id")
        return {"error": "Missing required parameters: file_url, file_id, user_id"}, 400

    try:
        # Download PDF from provided URL
        response = requests.get(file_url)
        if response.status_code != 200:
            logger.error(f"Failed to download PDF from S3: {file_url}")
            return {"error": f"Failed to download PDF from S3: {file_url}"}, 404

        # Extract and chunk PDF content
        entry = extract_pdf_text_and_links_from_s3(response.content, file_url)
        if not entry or "pages" not in entry:
            logger.error("No pages found in extracted PDF content.")
            return {"error": "No pages found in extracted PDF content."}, 404

        entry['file_id'] = file_id
        chunk_data = chunk_text(entry)
        chunks = chunk_data.get("chunks", [])

        # Store chunks into FAISS index if available
        if chunks:
            try:
                index_path = process_chunk_to_faiss_index(chunks, user_id)
            except Exception as e:
                logger.error(f"Error processing chunks to FAISS index: {e}")
                index_path = None

        # Prepare and return result
        result = {
            "total_pages": len(entry["pages"]),
            "total_chars": chunk_data.get("total_chars", 0),
            "total_chunks": len(chunks),
            "chunks": chunks,
            "faiss_index_path": index_path
        }
        return result, 200

    except Exception as e:
        logger.error(f"Error in process_pdfs: {e}", exc_info=True)
        return {"error": f"Error in process_pdfs: {e}"}, 404

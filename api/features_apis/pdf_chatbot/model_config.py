import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from api.features_apis.pdf_chatbot.config import LLM_MODEL, LLM_TEMPERATURE, EMBEDDING_MODEL

# Logger setup for tracking model loading activities
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()

# Retrieve the Gemini API key from the environment
GEMINI_API_KEY = os.getenv("API_KEY")


def get_embedding():
    """
    Load and return the Gemini embedding model used to vectorize document text.

    Returns:
        GoogleGenerativeAIEmbeddings: LangChain wrapper for the embedding model.
    """
    logger.info("Loading Gemini Embedding model...")

    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY,
        )
        logger.info(f"Successfully loaded Gemini embedding model: {EMBEDDING_MODEL}")
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to load Gemini embedding model '{EMBEDDING_MODEL}': {e}")
        raise


def get_llm_model():
    """
    Load and return the Gemini language model used for response generation.

    Returns:
        GoogleGenerativeAI: LangChain wrapper for the LLM.
    """
    logger.info("Loading Gemini LLM model...")

    try:
        llm = GoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE,
        )
        logger.info(f"Successfully loaded Gemini model: {LLM_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Failed to load Gemini model '{LLM_MODEL}': {e}")
        raise

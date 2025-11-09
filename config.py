import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (like COHERE_API_KEY)
load_dotenv()

class Config:
    # --- File Paths ---
    BASE_DIR = Path(__file__).resolve().parent
    DATA_SOURCE_FILE = BASE_DIR / "data" / "guvi_data.txt"
    VECTOR_STORE_PATH = BASE_DIR / "vector_store" / "db_faiss"

    # --- Data Ingestion ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

    # --- RAG Pipeline ---
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_NAME = "command-a-03-2025" # Using the latest powerful model
    LLM_TEMPERATURE = 0.3
    RETRIEVAL_TOP_K = 5

    # --- API Keys ---
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    # --- System Prompt ---
    SYSTEM_PROMPT = """
    You are a helpful AI RAG Model for GUVI. Your goal is to provide **highly detailed** answers
    based *only* on the provided context.

    **CRITICAL INSTRUCTIONS:**
    1.  **BE DETAILED:** Write a comprehensive, multi-sentence answer (at least 4-8 sentences).
    2.  **USE THE CONTEXT:** Base your entire answer *only* on the text in the "Context" section.
    3.  **BE A HELPER:** Answer the user's question directly.

    If the context does not contain the answer, you MUST say:
    "I'm sorry, I don't have enough information to answer that question based on the provided data."

    Context:
    {context}

    Question:
    {question}

    Helpful, Detailed Answer:
    """

    # --- Streamlit App UI ---
    APP_TITLE = "GUVI AI RAG Model"
    APP_ICON = "🤖"
    PAGE_LAYOUT = "wide"
    WELCOME_MESSAGE = "Hi! I'm the GUVI AI RAG Model. How can I help you today?"
    MAX_CHAT_HISTORY = 50

    SAMPLE_QUESTIONS = [
        "What is GUVI?",
        "What courses does GUVI offer?",
        "Tell me about the Zen Class program",
        "What is the refund policy?",
        "How can I contact GUVI?"
    ]
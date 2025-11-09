import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config # <-- Import from our new config file

def create_vector_db():
    """Loads text, splits it, creates embeddings, and saves to FAISS."""
    
    print("Starting vector store creation...")
    
    # 1. Load your .txt file
    if not os.path.exists(Config.DATA_SOURCE_FILE):
        print(f"Error: Data file not found at {Config.DATA_SOURCE_FILE}")
        return
        
    loader = TextLoader(str(Config.DATA_SOURCE_FILE), encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, 
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split document into {len(docs)} chunks.")
    if not docs:
        print("Error: No documents were created. Check file content.")
        return

    # 3. Create embeddings (using a Hugging Face model)
    print(f"Loading embedding model: {Config.EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print("Embedding model loaded.")

    # 4. Create FAISS vector store and save it locally
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(docs, embeddings)
    
    os.makedirs(os.path.dirname(Config.VECTOR_STORE_PATH), exist_ok=True)
    db.save_local(str(Config.VECTOR_STORE_PATH))
    print(f"✅ Vector store created and saved at: {Config.VECTOR_STORE_PATH}")

if __name__ == "__main__":
    create_vector_db()
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys - only from environment
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    # Ollama runs locally - no URL needed
    
    # Model Configuration
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # LLM Provider Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cohere")  # cohere or ollama
    COHERE_MODEL = "command-r-plus-04-2024"
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
    
    # Vector Store Configuration
    VECTOR_STORE_PATH = "data/vectorstore"
    COLLECTION_NAME = "bangla_documents_v2"
    
    # File paths
    RAW_DATA_PATH = "data/raw"
    CACHE_PATH = "data/cache"
    
    # Chunking Configuration
    CHUNK_SIZE = 450
    CHUNK_OVERLAP = 150
    
    # Retrieval Configuration
    TOP_K = 8
    SIMILARITY_THRESHOLD = 0.4
    BENGALI_SIMILARITY_THRESHOLD = 0.2
    
    # Generation Configuration
    MAX_TOKENS = 800
    TEMPERATURE = 0.2
    
    # Language Detection Settings
    BENGALI_CONFIDENCE_THRESHOLD = 0.4
    MIXED_LANGUAGE_HANDLING = True
    
    # Search Configuration
    HYBRID_SEARCH_ENABLED = True
    SEMANTIC_WEIGHT = 0.4
    KEYWORD_WEIGHT = 0.2

settings = Settings()
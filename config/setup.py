import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
load_dotenv()

class Configuration(BaseSettings):
    """Configuration class for application settings."""

    app_name: str = os.getenv("APP_NAME", "Advanced-RAG")

    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_model2: str = os.getenv("OLLAMA_MODEL", "gemma3:12b")
    ollama_model3: str = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    ollama_model4: str = os.getenv("OLLAMA_MODEL", "qwen2.5:32b") # To choose the model
    ollama_embedder_model: str = os.getenv("OLLAMA_EMBEDDER", "nomic-embed-text")

    docs_folder: str = "./knowledge"
    chromadb_folder: str = os.getenv("CHROMADB_FOLDER", "./db")
    knowledge_200_collection: str = os.getenv("CHROMADB_COLLECTION", "knowledge_200")
    knowledge_500_collection: str = os.getenv("CHROMADB_COLLECTION", "knowledge_500")
    knowledge_1000_collection: str = os.getenv("CHROMADB_COLLECTION", "knowledge_1000")

    chunk_size_200: int = os.getenv("CHUNK_SIZE", 200)
    chunk_size_500: int = os.getenv("CHUNK_SIZE", 500)
    chunk_size_1000: int = os.getenv("CHUNK_SIZE", 1000)

    chunk_overlap_200: int = os.getenv("CHUNK_OVERLAP", 40)
    chunk_overlap_500: int = os.getenv("CHUNK_OVERLAP", 100)
    chunk_overlap_1000: int = os.getenv("CHUNK_OVERLAP", 200)

    retriever_search_type: str = os.getenv("RETRIEVER_SEARCH_TYPE", 'similarity')
    retriever_k_5: int = os.getenv("RETRIEVER_K_VALUE", 5)
    retriever_k_10: int = os.getenv("RETRIEVER_K_VALUE", 10)
    retriever_k_15: int = os.getenv("RETRIEVER_K_VALUE", 15)
    retriever_k_20: int = os.getenv("RETRIEVER_K_VALUE", 20)
    model_temperature: float = os.getenv("MODEL_TEMPERATURE", 0.0)

    together_activated: bool = os.getenv("TOGETHER_ACTIVATED", True)
    together_api_key: str = os.getenv("TOGETHER_API_KEY", "")


config = Configuration()
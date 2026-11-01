"""Application configuration"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration"""
    
    # Model settings
    SUMMARIZATION_MODEL: str = "sshleifer/distilbart-cnn-12-6"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    DEVICE: str = "cpu"  # "cuda" for GPU
    
    # Processing settings
    MAX_CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 100
    MAX_SUMMARY_LENGTH: int = 500
    MIN_SUMMARY_LENGTH: int = 100
    
    # LLM API settings (optional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    USE_LLM_API: bool = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    
    # UI settings
    APP_TITLE: str = "AI Research Paper Analyzer"
    APP_ICON: str = "📄"
    MAX_FILE_SIZE_MB: int = 50
    
    # Export settings
    EXPORT_DIR: str = "data/exports"
    
    def __post_init__(self):
        os.makedirs(self.EXPORT_DIR, exist_ok=True)


config = Config()
"""
Investment Assistant Configuration
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration"""
    
    # API settings
    proxy_api_token: str = os.getenv("PROXYAPI_TOKEN", "sk-7LSGkwaOK7oTdJOvQSdhwKgEfosLj67C")
    proxy_api_base: str = "https://api.proxyapi.ru"
    
    # LLM settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  
    model_name: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    # Server settings
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", 8000))
    
    @property
    def api_base_url(self) -> str:
        return f"{self.proxy_api_base}/{self.llm_provider}/v1"


config = Config()
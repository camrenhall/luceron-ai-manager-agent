"""
Configuration management for the Luceron AI Manager Agent.
Centralizes environment variable handling and application settings.
"""
import os
from typing import Optional


class Settings:
    """Application settings and configuration management."""
    
    def __init__(self):
        """Initialize settings from environment variables."""
        # Service Configuration
        self.PORT = int(os.getenv("PORT", 8080))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Agent URLs
        self.COMMUNICATIONS_AGENT_URL = os.getenv("COMMUNICATIONS_AGENT_URL", "http://localhost:8082")
        self.ANALYSIS_AGENT_URL = os.getenv("ANALYSIS_AGENT_URL", "http://localhost:8083")
        
        # Backend Configuration
        self.BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
        self.BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")
        
        # AI Configuration
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        
        # Performance Settings
        self.AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", 30))
        self.MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", 3))
        
        # Rate Limiting
        self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 60))
        self.RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
        
        # Resource Monitoring
        self.MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", 900))
        self.MAX_CPU_PERCENT = float(os.getenv("MAX_CPU_PERCENT", 80.0))
        
        # Security
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
        
        # Validate required settings
        self._validate_required_settings()
    
    def _validate_required_settings(self):
        """Validate that required environment variables are set."""
        required_vars = [
            ("ANTHROPIC_API_KEY", self.ANTHROPIC_API_KEY),
            ("BACKEND_URL", self.BACKEND_URL),
            ("BACKEND_API_KEY", self.BACKEND_API_KEY),
        ]
        
        for var_name, var_value in required_vars:
            if not var_value:
                raise ValueError(f"{var_name} environment variable is required")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"


# Global settings instance
settings = Settings()
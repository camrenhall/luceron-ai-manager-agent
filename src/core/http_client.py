"""
HTTP client management with connection pooling for inter-agent communication.
"""
import httpx
from typing import Optional

# Global HTTP client instance
_http_client: Optional[httpx.AsyncClient] = None


async def init_http_client():
    """Initialize HTTP client with connection pooling for inter-agent communication."""
    global _http_client
    
    limits = httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30.0
    )
    
    timeout = httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=10.0,
        pool=5.0
    )
    
    _http_client = httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        follow_redirects=True
    )


async def close_http_client():
    """Clean up HTTP client."""
    global _http_client
    if _http_client:
        await _http_client.aclose()


def get_http_client() -> httpx.AsyncClient:
    """
    Get the global HTTP client.
    
    Returns:
        Configured HTTP client instance
        
    Raises:
        RuntimeError: If HTTP client not initialized
    """
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized")
    return _http_client
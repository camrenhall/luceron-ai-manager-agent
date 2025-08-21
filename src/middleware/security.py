"""
Essential security middleware for the Luceron AI Manager Agent.
"""
import time
from collections import defaultdict
from typing import Dict, List
from fastapi import HTTPException, Request


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        client_requests = self.requests[client_ip]
        
        # Remove old requests outside window
        self.requests[client_ip] = [
            req_time for req_time in client_requests
            if now - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_ip].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


async def security_middleware(request: Request, call_next):
    """Essential security middleware with rate limiting."""
    client_ip = request.client.host
    
    try:
        # Basic rate limiting
        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Process request
        response = await call_next(request)
        return response
        
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
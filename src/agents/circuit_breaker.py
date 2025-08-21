"""
Circuit breaker pattern implementation for agent communication reliability.
"""
import time
from typing import Callable, Any

from ..models.exceptions import AgentUnavailableError


class AgentCircuitBreaker:
    """Simple circuit breaker for agent communication reliability."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call_agent(self, agent_client_func: Callable, *args, **kwargs) -> Any:
        """
        Execute agent call with circuit breaker protection.
        
        Args:
            agent_client_func: The agent client function to call
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result from the agent client function
            
        Raises:
            AgentUnavailableError: When circuit breaker is open
        """
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise AgentUnavailableError("Agent circuit breaker is open")
        
        try:
            response = await agent_client_func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return response
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise e
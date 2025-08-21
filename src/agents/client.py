"""
HTTP client for communicating with specialized agents.
"""
import asyncio
import httpx

from ..models.requests import AgentTaskRequest, AgentTaskResponse
from ..models.exceptions import AgentTimeoutError, AgentCommunicationError
from ..core.http_client import get_http_client
from .circuit_breaker import AgentCircuitBreaker


class AgentClient:
    """HTTP client for communicating with specialized agents."""
    
    def __init__(self, agent_name: str, base_url: str):
        """
        Initialize agent client.
        
        Args:
            agent_name: Name of the agent for logging
            base_url: Base URL for the agent service
        """
        self.agent_name = agent_name
        self.base_url = base_url
        self.circuit_breaker = AgentCircuitBreaker()
    
    async def delegate_task(self, request: AgentTaskRequest) -> AgentTaskResponse:
        """
        Delegate task to specialized agent with retry logic.
        
        Args:
            request: Task request to send to the agent
            
        Returns:
            Response from the agent
            
        Raises:
            AgentTimeoutError: When request times out
            AgentCommunicationError: When communication fails
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return await self.circuit_breaker.call_agent(self._make_request, request)
            except (AgentTimeoutError, AgentCommunicationError) as e:
                if attempt == max_retries - 1:
                    raise e
                
                await asyncio.sleep(retry_delay * (attempt + 1))
    
    async def _make_request(self, request: AgentTaskRequest) -> AgentTaskResponse:
        """
        Make HTTP request to agent.
        
        Args:
            request: Task request to send
            
        Returns:
            Response from the agent
            
        Raises:
            AgentTimeoutError: When request times out
            AgentCommunicationError: When communication fails
        """
        client = get_http_client()
        
        try:
            response = await client.post(
                f"{self.base_url}/agent/task",
                json=request.dict(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = AgentTaskResponse(**response.json())
            return result
            
        except httpx.TimeoutError:
            raise AgentTimeoutError(f"Agent communication timed out")
        except httpx.HTTPStatusError as e:
            raise AgentCommunicationError(f"Agent returned error: {e.response.status_code}")
        except Exception as e:
            raise AgentCommunicationError(f"Unexpected error: {str(e)}")
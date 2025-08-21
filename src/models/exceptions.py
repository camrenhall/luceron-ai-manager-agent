"""
Custom exceptions for the Luceron AI Manager Agent.
"""


class AgentCommunicationError(Exception):
    """Raised when agent communication fails."""
    pass


class AgentTimeoutError(AgentCommunicationError):
    """Raised when agent communication times out."""
    pass


class AgentUnavailableError(AgentCommunicationError):
    """Raised when agent is unavailable (circuit breaker open)."""
    pass
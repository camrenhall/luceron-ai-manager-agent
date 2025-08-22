"""
Request and response models for the Luceron AI Manager Agent.
Defines Pydantic models for API communication and validation.
"""
import re
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class ChatRequest(BaseModel):
    """Model for incoming chat requests from users."""
    
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    
    @validator('message')
    def sanitize_message(cls, v):
        """Sanitize user input to prevent XSS and injection attacks."""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        
        # Remove potentially harmful characters
        sanitized = re.sub(r'[<>"\'\\]', '', v)
        sanitized = sanitized.strip()
        
        if len(sanitized) == 0:
            raise ValueError("Message contains no valid content")
            
        return sanitized
    
    @validator('conversation_id')
    def validate_conversation_id(cls, v):
        """Validate conversation ID format."""
        if v is not None:
            if not re.match(r'^[a-f0-9\-]{36}$', v):
                raise ValueError("Invalid conversation ID format")
        return v


class ChatResponse(BaseModel):
    """Model for chat responses sent to users."""
    
    response: str = Field(..., description="Agent response")
    conversation_id: str = Field(..., description="Conversation identifier")
    agent_used: str = Field(..., description="Which agent handled the request")
    execution_time: float = Field(..., description="Processing time in seconds")


class AgentTaskRequest(BaseModel):
    """Model for task delegation between agents."""
    
    task_id: str = Field(..., description="Unique task identifier")
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=10000)
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: str = Field(default="normal", description="Task priority")
    
    @validator('task_id')
    def validate_task_id(cls, v):
        """Validate task ID format."""
        if not re.match(r'^[a-f0-9\-]{36}$', v):
            raise ValueError("Invalid task ID format")
        return v
    
    @validator('message')
    def sanitize_message(cls, v):
        """Sanitize task message."""
        sanitized = re.sub(r'[<>]', '', v)
        return sanitized.strip()
    
    @validator('context')
    def validate_context_size(cls, v):
        """Limit context size to prevent abuse."""
        if len(json.dumps(v)) > 50000:  # 50KB limit
            raise ValueError("Context size exceeds maximum allowed")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority values."""
        if v not in ["low", "normal", "high"]:
            raise ValueError("Priority must be low, normal, or high")
        return v


class AgentTaskResponse(BaseModel):
    """Model for responses from agent task delegation."""
    
    task_id: str
    conversation_id: Optional[str] = None
    response: str = Field(..., description="Agent's response")
    status: str = Field(..., description="Task status")
    execution_time: float
    context_updates: Dict[str, Any] = Field(default_factory=dict)


# SSE Response Models for Chat Endpoint

class AgentResponseEvent(BaseModel):
    """Model for successful agent response events via SSE."""
    
    type: str = Field(default="agent_response", description="Event type identifier")
    response: str = Field(..., description="Agent's text response")
    conversation_id: str = Field(..., description="UUID for conversation tracking")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO 8601 timestamp")
    case_id: Optional[str] = Field(None, description="Associated case ID if applicable")
    has_context: Optional[bool] = Field(False, description="Whether response includes context")
    context_keys: Optional[List[str]] = Field(default_factory=list, description="Keys of context data used")
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance and usage metrics")


class AgentErrorEvent(BaseModel):
    """Model for agent error events via SSE."""
    
    type: str = Field(default="agent_error", description="Event type identifier")
    error_message: str = Field(..., description="Error description")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO 8601 timestamp")
    error_type: Optional[str] = Field("general", description="Classification of error")
    recovery_suggestion: Optional[str] = Field(None, description="Suggested recovery action")


class GeneralErrorEvent(BaseModel):
    """Model for general error events via SSE."""
    
    type: str = Field(default="error", description="Event type identifier")
    message: str = Field(..., description="Error description")
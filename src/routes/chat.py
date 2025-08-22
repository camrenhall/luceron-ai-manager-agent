"""
Chat and agent communication routes for the Luceron AI Manager Agent.
"""
import json
import time
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..models.requests import (
    ChatRequest, ChatResponse, AgentTaskRequest, AgentTaskResponse,
    AgentResponseEvent, AgentErrorEvent, GeneralErrorEvent
)
from ..agents.manager import create_manager_agent
from ..agents.client import AgentClient
from ..config.settings import settings

router = APIRouter()

# Initialize manager agent
manager_agent = create_manager_agent()

# Initialize fallback client
communications_client = AgentClient("communications", settings.COMMUNICATIONS_AGENT_URL)


def format_sse_event(data: dict, event_type: str = "message") -> str:
    """Format data as Server-Sent Event."""
    return f"data: {json.dumps(data)}\n\n"


async def generate_chat_response(request: ChatRequest):
    """Generate streaming chat response using SSE format."""
    start_time = time.time()
    
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        # Use LangChain Manager Agent for intelligent orchestration
        result = await manager_agent.ainvoke({
            "input": request.message,
            "conversation_id": conversation_id
        })
        
        # Extract the final response
        response_text = result.get("output", "No response generated")
        execution_time = time.time() - start_time
        
        # Create successful response event
        response_event = AgentResponseEvent(
            response=response_text,
            conversation_id=conversation_id,
            has_context=True,
            context_keys=["llm-orchestrator"],
            metrics={
                "agent_used": "llm-orchestrator",
                "execution_time": execution_time
            }
        )
        
        yield format_sse_event(response_event.dict())
        
    except Exception as primary_error:
        # Fallback to simple delegation if LLM orchestration fails
        try:
            task_request = AgentTaskRequest(
                task_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                message=request.message,
                context={},
                priority="normal"
            )
            
            agent_response = await communications_client.delegate_task(task_request)
            execution_time = time.time() - start_time
            
            # Create fallback response event
            response_event = AgentResponseEvent(
                response=f"[Fallback Mode] {agent_response.response}",
                conversation_id=conversation_id,
                has_context=True,
                context_keys=["fallback-communications"],
                metrics={
                    "agent_used": "fallback-communications",
                    "execution_time": execution_time,
                    "fallback_reason": "llm_orchestration_failed"
                }
            )
            
            yield format_sse_event(response_event.dict())
            
        except Exception as fallback_error:
            execution_time = time.time() - start_time
            
            # Create error response event
            error_event = AgentErrorEvent(
                error_message=f"I'm experiencing technical difficulties. Your request was: {request.message[:100]}...",
                error_type="system_failure",
                recovery_suggestion="Please try again in a few moments. If the problem persists, contact support."
            )
            
            yield format_sse_event(error_event.dict())


@router.post("/chat")
async def chat(request: ChatRequest):
    """Primary chat interface with intelligent LLM-based agent orchestration via SSE."""
    return StreamingResponse(
        generate_chat_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "https://simple-s3-upload.onrender.com",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )


@router.post("/agent/task", response_model=AgentTaskResponse)
async def handle_agent_task(request: AgentTaskRequest):
    """Handle task delegation from other agents (for future multi-agent workflows)."""
    start_time = time.time()
    
    try:
        # For MVP, just echo back with basic processing
        response_text = f"Manager Agent processed task: {request.message}"
        execution_time = time.time() - start_time
        
        return AgentTaskResponse(
            task_id=request.task_id,
            conversation_id=request.conversation_id,
            response=response_text,
            status="completed",
            execution_time=execution_time,
            context_updates={}
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        return AgentTaskResponse(
            task_id=request.task_id,
            conversation_id=request.conversation_id,
            response=f"Task failed: {str(e)}",
            status="failed",
            execution_time=execution_time,
            context_updates={}
        )
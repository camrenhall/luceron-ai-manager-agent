"""
Chat and agent communication routes for the Luceron AI Manager Agent.
"""
import json
import logging
import time
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..models.requests import (
    ChatRequest, ChatResponse, AgentTaskRequest, AgentTaskResponse,
    AgentResponseEvent, AgentErrorEvent, GeneralErrorEvent
)
from ..agents.manager import create_manager_agent

router = APIRouter()

# Initialize manager agent
manager_agent = create_manager_agent()


def format_sse_event(data: dict, event_type: str = "message") -> str:
    """Format data as Server-Sent Event."""
    return f"data: {json.dumps(data)}\n\n"


async def generate_chat_response(request: ChatRequest):
    """Generate streaming chat response using SSE format."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    logger.info(f"Starting chat request - conversation_id: {conversation_id}, message: {request.message[:100]}...")
    
    try:
        logger.info("Invoking LangChain manager agent...")
        # Use LangChain Manager Agent for intelligent orchestration
        result = await manager_agent.ainvoke({
            "input": request.message,
            "conversation_id": conversation_id
        })
        
        logger.info(f"LangChain agent completed. Result type: {type(result)}, Result: {result}")
        
        # Extract the final response - handle LangChain response format
        if isinstance(result, dict) and "output" in result:
            output = result["output"]
            if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                # Handle format like {'output': [{'text': '...', 'type': 'text', 'index': 0}]}
                response_text = output[0].get("text", "No response generated")
            else:
                response_text = str(output)
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            # Handle format like [{'text': '...', 'type': 'text', 'index': 0}]
            response_text = result[0].get("text", "No response generated")
        else:
            response_text = str(result)
        
        logger.info(f"Extracted response text: {response_text[:200]}...")
        
        execution_time = time.time() - start_time
        
        # Create successful response event
        logger.info("Creating AgentResponseEvent...")
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
        
        logger.info("Formatting SSE event...")
        sse_data = format_sse_event(response_event.dict())
        logger.info(f"SSE event formatted: {sse_data[:200]}...")
        
        yield sse_data
        logger.info("Successfully yielded SSE response")
        
    except Exception as error:
        execution_time = time.time() - start_time
        logger.error(f"Manager agent failed after {execution_time}s: {str(error)}", exc_info=True)
        
        # Fail hard as requested - no fallback logic
        error_event = AgentErrorEvent(
            error_message=f"Manager agent failed: {str(error)}",
            error_type="agent_failure",
            recovery_suggestion="Check agent configuration and try again."
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
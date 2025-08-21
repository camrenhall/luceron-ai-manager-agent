"""
Chat and agent communication routes for the Luceron AI Manager Agent.
"""
import time
import uuid
from fastapi import APIRouter

from ..models.requests import ChatRequest, ChatResponse, AgentTaskRequest, AgentTaskResponse
from ..agents.manager import create_manager_agent
from ..agents.client import AgentClient
from ..config.settings import settings

router = APIRouter()

# Initialize manager agent
manager_agent = create_manager_agent()

# Initialize fallback client
communications_client = AgentClient("communications", settings.COMMUNICATIONS_AGENT_URL)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Primary chat interface with intelligent LLM-based agent orchestration."""
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
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            agent_used="llm-orchestrator",
            execution_time=execution_time
        )
        
    except Exception:
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
            
            return ChatResponse(
                response=f"[Fallback Mode] {agent_response.response}",
                conversation_id=conversation_id,
                agent_used="fallback-communications",
                execution_time=execution_time
            )
            
        except Exception:
            execution_time = time.time() - start_time
            
            return ChatResponse(
                response=f"I'm experiencing technical difficulties. Your request was: {request.message[:100]}...",
                conversation_id=conversation_id,
                agent_used="error-fallback",
                execution_time=execution_time
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
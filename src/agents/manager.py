"""
LangChain-based Manager Agent for intelligent task orchestration.
"""
import uuid
from typing import Dict, Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from ..config.settings import settings
from ..models.requests import AgentTaskRequest
from .client import AgentClient


def load_system_prompt() -> str:
    """Load the manager system prompt from file."""
    try:
        with open("prompts/manager_system_prompt.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "You are the Manager Agent for the Luceron AI eDiscovery Platform."


# Initialize agent clients
communications_client = AgentClient("communications", settings.COMMUNICATIONS_AGENT_URL)
analysis_client = AgentClient("analysis", settings.ANALYSIS_AGENT_URL)


@tool
async def delegate_to_communications(task_description: str, context: Dict[str, Any] = None) -> str:
    """
    Delegate a task to the Communications Agent for client communications, emails, and case management.
    
    Args:
        task_description: Clear description of what the Communications Agent should do
        context: Optional context dictionary with additional information
    
    Returns:
        Response from the Communications Agent
    """
    try:
        task_request = AgentTaskRequest(
            task_id=str(uuid.uuid4()),
            message=task_description,
            context=context or {},
            priority="normal"
        )
        
        response = await communications_client.delegate_task(task_request)
        return f"Communications Agent Response: {response.response}"
        
    except Exception as e:
        return f"Error communicating with Communications Agent: {str(e)}"


@tool
async def delegate_to_analysis(task_description: str, context: Dict[str, Any] = None) -> str:
    """
    Delegate a task to the Analysis Agent for document analysis, review, and legal analysis.
    
    Args:
        task_description: Clear description of what the Analysis Agent should do
        context: Optional context dictionary with additional information
    
    Returns:
        Response from the Analysis Agent
    """
    try:
        task_request = AgentTaskRequest(
            task_id=str(uuid.uuid4()),
            message=task_description,
            context=context or {},
            priority="normal"
        )
        
        response = await analysis_client.delegate_task(task_request)
        return f"Analysis Agent Response: {response.response}"
        
    except Exception as e:
        return f"Error communicating with Analysis Agent: {str(e)}"


def create_manager_agent() -> AgentExecutor:
    """
    Create the Manager Agent with LangChain and Claude integration.
    
    Returns:
        Configured AgentExecutor ready for task orchestration
    """
    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=settings.ANTHROPIC_API_KEY,
        temperature=0.1
    )
    
    # Load system prompt
    system_prompt = load_system_prompt()
    
    # Define tools
    tools = [delegate_to_communications, delegate_to_analysis]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor
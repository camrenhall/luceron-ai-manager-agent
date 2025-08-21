"""
Luceron AI Manager Agent - Production Implementation
Main FastAPI application serving as the intelligent orchestration layer for specialized agents.
"""
import os
import logging
import time
import uuid
import asyncio
import re
import psutil
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import httpx

# LangChain imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Configure structured logging
import json
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Structured logging utilities
class StructuredLogger:
    """Enhanced logging with structured data for production monitoring"""
    
    def __init__(self, base_logger):
        self.logger = base_logger
        
    def log_request(self, request_id: str, method: str, path: str, client_ip: str, **kwargs):
        """Log incoming request with structured data"""
        log_data = {
            "event": "request_received",
            "request_id": request_id,
            "method": method,
            "path": path,
            "client_ip": client_ip,
            **kwargs
        }
        self.logger.info(f"REQUEST {json.dumps(log_data)}")
    
    def log_response(self, request_id: str, status_code: int, duration: float, **kwargs):
        """Log response with performance metrics"""
        log_data = {
            "event": "request_completed",
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
            **kwargs
        }
        self.logger.info(f"RESPONSE {json.dumps(log_data)}")
    
    def log_agent_delegation(self, request_id: str, agent: str, task_id: str, **kwargs):
        """Log agent delegation events"""
        log_data = {
            "event": "agent_delegation",
            "request_id": request_id,
            "agent": agent,
            "task_id": task_id,
            **kwargs
        }
        self.logger.info(f"DELEGATION {json.dumps(log_data)}")
    
    def log_error(self, request_id: str, error_type: str, error_message: str, **kwargs):
        """Log errors with context"""
        log_data = {
            "event": "error",
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message,
            **kwargs
        }
        self.logger.error(f"ERROR {json.dumps(log_data)}")

# Initialize structured logger
structured_logger = StructuredLogger(logger)

# Environment Configuration
MANAGER_PORT = int(os.getenv("MANAGER_PORT", 8081))
COMMUNICATIONS_AGENT_URL = os.getenv("COMMUNICATIONS_AGENT_URL", "http://localhost:8082")
ANALYSIS_AGENT_URL = os.getenv("ANALYSIS_AGENT_URL", "http://localhost:8083")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Validate required environment variables
required_vars = [
    ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
    ("BACKEND_URL", BACKEND_URL),
    ("BACKEND_API_KEY", BACKEND_API_KEY),
]

for var_name, var_value in required_vars:
    if not var_value:
        raise ValueError(f"{var_name} environment variable is required")

# Global HTTP client
_http_client: Optional[httpx.AsyncClient] = None

# Enhanced Pydantic Models with Security Validation
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    
    @validator('message')
    def sanitize_message(cls, v):
        """Sanitize user input to prevent XSS and injection attacks"""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        
        # Remove potentially harmful characters
        sanitized = re.sub(r'[<>"\']', '', v)
        sanitized = sanitized.strip()
        
        if len(sanitized) == 0:
            raise ValueError("Message contains no valid content")
            
        return sanitized
    
    @validator('conversation_id')
    def validate_conversation_id(cls, v):
        """Validate conversation ID format"""
        if v is not None:
            if not re.match(r'^[a-f0-9\-]{36}$', v):
                raise ValueError("Invalid conversation ID format")
        return v

class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent response")
    conversation_id: str = Field(..., description="Conversation identifier")
    agent_used: str = Field(..., description="Which agent handled the request")
    execution_time: float = Field(..., description="Processing time in seconds")

class AgentTaskRequest(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=10000)
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: str = Field(default="normal", description="Task priority")
    
    @validator('task_id')
    def validate_task_id(cls, v):
        """Validate task ID format"""
        if not re.match(r'^[a-f0-9\-]{36}$', v):
            raise ValueError("Invalid task ID format")
        return v
    
    @validator('message')
    def sanitize_message(cls, v):
        """Sanitize task message"""
        sanitized = re.sub(r'[<>]', '', v)
        return sanitized.strip()
    
    @validator('context')
    def validate_context_size(cls, v):
        """Limit context size to prevent abuse"""
        import json
        if len(json.dumps(v)) > 50000:  # 50KB limit
            raise ValueError("Context size exceeds maximum allowed")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority values"""
        if v not in ["low", "normal", "high"]:
            raise ValueError("Priority must be low, normal, or high")
        return v

class AgentTaskResponse(BaseModel):
    task_id: str
    conversation_id: Optional[str] = None
    response: str = Field(..., description="Agent's response")
    status: str = Field(..., description="Task status")
    execution_time: float
    context_updates: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    backend: str
    agents: Dict[str, str]

# Custom Exceptions
class AgentCommunicationError(Exception):
    """Raised when agent communication fails"""
    pass

class AgentTimeoutError(AgentCommunicationError):
    """Raised when agent communication times out"""
    pass

class AgentUnavailableError(AgentCommunicationError):
    """Raised when agent is unavailable (circuit breaker open)"""
    pass

# Circuit Breaker Pattern
class AgentCircuitBreaker:
    """Simple circuit breaker for agent communication reliability"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call_agent(self, agent_client_func, *args, **kwargs):
        """Execute agent call with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                logger.info("Circuit breaker moving to half-open state")
            else:
                raise AgentUnavailableError("Agent circuit breaker is open")
        
        try:
            response = await agent_client_func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed - agent recovered")
            return response
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened - {self.failure_count} failures")
            raise e

# Agent Communication Client
class AgentClient:
    """HTTP client for communicating with specialized agents"""
    
    def __init__(self, agent_name: str, base_url: str):
        self.agent_name = agent_name
        self.base_url = base_url
        self.circuit_breaker = AgentCircuitBreaker()
    
    async def delegate_task(self, request: AgentTaskRequest) -> AgentTaskResponse:
        """Delegate task to specialized agent with retry logic"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return await self.circuit_breaker.call_agent(self._make_request, request)
            except (AgentTimeoutError, AgentCommunicationError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to communicate with {self.agent_name} after {max_retries} attempts")
                    raise e
                
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {self.agent_name}: {str(e)}")
                await asyncio.sleep(retry_delay * (attempt + 1))
    
    async def _make_request(self, request: AgentTaskRequest) -> AgentTaskResponse:
        """Make HTTP request to agent"""
        client = get_http_client()
        
        try:
            logger.info(f"Delegating task {request.task_id} to {self.agent_name}")
            
            response = await client.post(
                f"{self.base_url}/agent/task",
                json=request.dict(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = AgentTaskResponse(**response.json())
            logger.info(f"Task {request.task_id} completed by {self.agent_name} in {result.execution_time:.2f}s")
            
            return result
            
        except httpx.TimeoutError:
            logger.error(f"Task {request.task_id} timed out with {self.agent_name}")
            raise AgentTimeoutError(f"Agent communication timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"Task {request.task_id} failed with {self.agent_name}: HTTP {e.response.status_code}")
            raise AgentCommunicationError(f"Agent returned error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error in task {request.task_id} with {self.agent_name}: {e}")
            raise AgentCommunicationError(f"Unexpected error: {str(e)}")

# Initialize agent clients
communications_client = AgentClient("communications", COMMUNICATIONS_AGENT_URL)
analysis_client = AgentClient("analysis", ANALYSIS_AGENT_URL)

# LangChain LLM Setup
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=ANTHROPIC_API_KEY,
    temperature=0.1
)

# Load system prompt
def load_system_prompt() -> str:
    """Load the manager system prompt from file"""
    try:
        with open("prompts/manager_system_prompt.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("System prompt file not found")
        return "You are the Manager Agent for the Luceron AI eDiscovery Platform."

system_prompt = load_system_prompt()

# LangChain Tools for Agent Delegation
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
        
        logger.info(f"Delegating to Communications Agent: {task_description[:100]}")
        response = await communications_client.delegate_task(task_request)
        
        return f"Communications Agent Response: {response.response}"
        
    except Exception as e:
        logger.error(f"Failed to delegate to Communications Agent: {e}")
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
        
        logger.info(f"Delegating to Analysis Agent: {task_description[:100]}")
        response = await analysis_client.delegate_task(task_request)
        
        return f"Analysis Agent Response: {response.response}"
        
    except Exception as e:
        logger.error(f"Failed to delegate to Analysis Agent: {e}")
        return f"Error communicating with Analysis Agent: {str(e)}"

# Tools list for the LangChain agent
tools = [delegate_to_communications, delegate_to_analysis]

# Create LangChain Agent
def create_manager_agent():
    """Create the Manager Agent with LangChain and Claude integration"""
    
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

# Initialize the manager agent
manager_agent = create_manager_agent()

# Rate Limiting
class RateLimiter:
    """Simple in-memory rate limiter for production use"""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is within rate limits"""
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

# Resource Monitor
class ResourceMonitor:
    """Monitor system resources for production deployment"""
    
    def __init__(self, max_memory_mb: int = 900, max_cpu_percent: float = 80.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
    
    def check_resources(self):
        """Check current resource usage"""
        try:
            process = psutil.Process()
            
            # Memory check
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_warning = memory_mb > self.max_memory_mb
            
            # CPU check
            cpu_percent = process.cpu_percent()
            cpu_warning = cpu_percent > self.max_cpu_percent
            
            return {
                "memory_mb": round(memory_mb, 2),
                "memory_warning": memory_warning,
                "cpu_percent": round(cpu_percent, 2),
                "cpu_warning": cpu_warning
            }
        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")
            return {"error": str(e)}

# Initialize production utilities
rate_limiter = RateLimiter(max_requests=60, window_seconds=60)
resource_monitor = ResourceMonitor()

# HTTP Client Management
async def init_http_client():
    """Initialize HTTP client with connection pooling for inter-agent communication"""
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
    
    logger.info("HTTP client initialized with connection pooling")

async def close_http_client():
    """Clean up HTTP client"""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        logger.info("HTTP client closed")

def get_http_client() -> httpx.AsyncClient:
    """Get the global HTTP client"""
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized")
    return _http_client

# Application Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info(f"Manager Agent starting on port {MANAGER_PORT}")
    await init_http_client()
    logger.info("Manager Agent startup complete")
    
    yield
    
    # Shutdown
    logger.info("Manager Agent shutting down")
    await close_http_client()
    logger.info("Manager Agent shutdown complete")

# FastAPI Application
app = FastAPI(
    title="Luceron AI Manager Agent",
    description="Central orchestration layer for the Luceron AI eDiscovery Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced security and monitoring middleware
@app.middleware("http")
async def enhanced_security_middleware(request: Request, call_next):
    """Enhanced security, rate limiting, monitoring, and structured logging middleware"""
    start_time = time.time()
    client_ip = request.client.host
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state for tracking
    request.state.request_id = request_id
    
    # Log incoming request with structured data
    structured_logger.log_request(
        request_id=request_id,
        method=request.method,
        path=str(request.url.path),
        client_ip=client_ip,
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    try:
        # Rate limiting with structured logging
        if not rate_limiter.is_allowed(client_ip):
            structured_logger.log_error(
                request_id=request_id,
                error_type="rate_limit_exceeded",
                error_message=f"Rate limit exceeded for {client_ip}",
                client_ip=client_ip
            )
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Resource monitoring with structured logging
        resources = resource_monitor.check_resources()
        if resources.get("memory_warning"):
            structured_logger.log_error(
                request_id=request_id,
                error_type="high_memory_usage",
                error_message=f"High memory usage: {resources['memory_mb']}MB",
                memory_mb=resources['memory_mb']
            )
        if resources.get("cpu_warning"):
            structured_logger.log_error(
                request_id=request_id,
                error_type="high_cpu_usage", 
                error_message=f"High CPU usage: {resources['cpu_percent']}%",
                cpu_percent=resources['cpu_percent']
            )
        
        # Process request
        response = await call_next(request)
        
        # Log successful response with metrics
        duration = time.time() - start_time
        structured_logger.log_response(
            request_id=request_id,
            status_code=response.status_code,
            duration=duration,
            memory_mb=resources.get('memory_mb', 0),
            cpu_percent=resources.get('cpu_percent', 0)
        )
        
        return response
        
    except HTTPException as e:
        # Log HTTP errors
        duration = time.time() - start_time
        structured_logger.log_error(
            request_id=request_id,
            error_type="http_exception",
            error_message=str(e.detail),
            status_code=e.status_code,
            duration=duration
        )
        raise
        
    except Exception as e:
        # Log unexpected errors
        duration = time.time() - start_time
        structured_logger.log_error(
            request_id=request_id,
            error_type="unexpected_error",
            error_message=str(e),
            duration=duration
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

# Enhanced health check models
class DetailedHealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    uptime_seconds: float
    system: Dict[str, Any]
    backend: Dict[str, Any]
    agents: Dict[str, Dict[str, Any]]
    dependencies: Dict[str, str]

# Track service startup time for uptime calculation
SERVICE_START_TIME = time.time()

# Enhanced Health Check Endpoints
@app.get("/", response_model=DetailedHealthResponse)
async def comprehensive_health_check():
    """Comprehensive health check with detailed system and dependency status"""
    request_id = getattr(request, 'state', {}).get('request_id', str(uuid.uuid4()))
    
    try:
        client = get_http_client()
        uptime = time.time() - SERVICE_START_TIME
        
        # System resource check
        resources = resource_monitor.check_resources()
        system_status = {
            "memory_mb": resources.get('memory_mb', 0),
            "memory_warning": resources.get('memory_warning', False),
            "cpu_percent": resources.get('cpu_percent', 0),
            "cpu_warning": resources.get('cpu_warning', False),
            "status": "healthy" if not (resources.get('memory_warning') or resources.get('cpu_warning')) else "degraded"
        }
        
        # Backend connectivity check with timing
        backend_start = time.time()
        try:
            backend_response = await client.get(f"{BACKEND_URL}/")
            backend_duration = time.time() - backend_start
            backend_status = {
                "status": "connected" if backend_response.status_code == 200 else "degraded",
                "response_time_ms": round(backend_duration * 1000, 2),
                "status_code": backend_response.status_code
            }
        except Exception as e:
            backend_duration = time.time() - backend_start
            backend_status = {
                "status": "unavailable",
                "response_time_ms": round(backend_duration * 1000, 2),
                "error": str(e)
            }
        
        # Agent connectivity checks with timing
        agent_status = {}
        
        # Communications Agent health
        comm_start = time.time()
        try:
            comm_response = await client.get(f"{COMMUNICATIONS_AGENT_URL}/status")
            comm_duration = time.time() - comm_start
            agent_status["communications"] = {
                "status": "connected" if comm_response.status_code == 200 else "degraded",
                "response_time_ms": round(comm_duration * 1000, 2),
                "status_code": comm_response.status_code
            }
        except Exception as e:
            comm_duration = time.time() - comm_start
            agent_status["communications"] = {
                "status": "unavailable",
                "response_time_ms": round(comm_duration * 1000, 2),
                "error": str(e)
            }
        
        # Analysis Agent health
        analysis_start = time.time()
        try:
            analysis_response = await client.get(f"{ANALYSIS_AGENT_URL}/status")
            analysis_duration = time.time() - analysis_start
            agent_status["analysis"] = {
                "status": "connected" if analysis_response.status_code == 200 else "degraded",
                "response_time_ms": round(analysis_duration * 1000, 2),
                "status_code": analysis_response.status_code
            }
        except Exception as e:
            analysis_duration = time.time() - analysis_start
            agent_status["analysis"] = {
                "status": "unavailable",
                "response_time_ms": round(analysis_duration * 1000, 2),
                "error": str(e)
            }
        
        # Check LangChain and AI dependencies
        dependencies = {}
        try:
            # Test manager agent initialization
            if manager_agent:
                dependencies["langchain"] = "healthy"
            else:
                dependencies["langchain"] = "unavailable"
        except Exception:
            dependencies["langchain"] = "error"
        
        # Determine overall status
        all_healthy = (
            system_status["status"] == "healthy" and
            backend_status["status"] in ["connected", "degraded"] and
            any(agent["status"] in ["connected", "degraded"] for agent in agent_status.values())
        )
        
        overall_status = "operational" if all_healthy else "degraded"
        
        # Log health check
        structured_logger.log_request(
            request_id=request_id,
            method="GET",
            path="/",
            client_ip="internal",
            event="health_check_completed",
            overall_status=overall_status
        )
        
        return DetailedHealthResponse(
            status=overall_status,
            service="manager-agent",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=round(uptime, 2),
            system=system_status,
            backend=backend_status,
            agents=agent_status,
            dependencies=dependencies
        )
        
    except Exception as e:
        structured_logger.log_error(
            request_id=request_id,
            error_type="health_check_failed",
            error_message=str(e)
        )
        raise HTTPException(status_code=503, detail="Health check failed")

@app.get("/status")
async def status_check():
    """Simple status endpoint for load balancer"""
    return {"status": "running", "service": "manager-agent"}

# Chat endpoint with LLM-based orchestration  
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Primary chat interface with intelligent LLM-based agent orchestration"""
    start_time = time.time()
    
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    logger.info(f"Incoming chat request: {request.message[:50]}...")
    
    try:
        # Use LangChain Manager Agent for intelligent orchestration
        logger.info("Orchestrating request with LLM-based agent selection")
        
        # Execute the manager agent with the user's request
        result = await manager_agent.ainvoke({
            "input": request.message,
            "conversation_id": conversation_id
        })
        
        # Extract the final response
        response_text = result.get("output", "No response generated")
        
        execution_time = time.time() - start_time
        
        logger.info(f"LLM orchestration completed in {execution_time:.2f}s")
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            agent_used="llm-orchestrator",
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"LLM orchestration failed: {e}")
        
        # Fallback to simple delegation if LLM orchestration fails
        try:
            logger.info("Fallback to simple communications agent delegation")
            
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
            
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            
            execution_time = time.time() - start_time
            
            return ChatResponse(
                response=f"I'm experiencing technical difficulties. Your request was: {request.message[:100]}...",
                conversation_id=conversation_id,
                agent_used="error-fallback",
                execution_time=execution_time
            )

# Agent task endpoint for inter-agent communication
@app.post("/agent/task", response_model=AgentTaskResponse)
async def handle_agent_task(request: AgentTaskRequest):
    """Handle task delegation from other agents (for future multi-agent workflows)"""
    start_time = time.time()
    
    logger.info(f"Received agent task: {request.task_id}")
    
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
        logger.error(f"Agent task failed: {e}")
        
        execution_time = time.time() - start_time
        
        return AgentTaskResponse(
            task_id=request.task_id,
            conversation_id=request.conversation_id,
            response=f"Task failed: {str(e)}",
            status="failed",
            execution_time=execution_time,
            context_updates={}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=MANAGER_PORT,
        log_level="info"
    )
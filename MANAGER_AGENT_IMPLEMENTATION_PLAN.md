# Luceron AI Manager Agent - MVP Implementation Plan

## Executive Summary

This plan outlines the development of the Manager Agent MVP - a lightweight, production-grade orchestration layer for the Luceron AI eDiscovery Platform. The implementation prioritizes essential functionality over feature completeness, delivering a working system that can route tasks between specialized agents while maintaining scalability for future enhancements.

## Implementation Philosophy

**MVP Principles:**
- **Functional over Perfect**: Deliver working orchestration with essential features
- **Lightweight Architecture**: Minimal dependencies, maximum efficiency
- **Production-Ready**: Enterprise-grade reliability and security from day one
- **Scalable Foundation**: Architecture that supports future agent additions
- **Fast Delivery**: Focus on core capabilities, defer advanced features

## Core Implementation Strategy

### Phase 1: Foundation Layer (Week 1)
**Objective**: Establish basic agent infrastructure and communication patterns

**Deliverables:**
1. **FastAPI Application Structure**
   - Basic FastAPI app with CORS and middleware
   - Health check endpoints (`/`, `/status`)
   - Environment configuration management
   - Logging infrastructure with emoji-based patterns

2. **Inter-Agent HTTP Client**
   - HTTPX async client with connection pooling
   - Basic request/response models with Pydantic
   - Timeout and retry logic
   - Circuit breaker pattern for agent failures

3. **Basic Agent Communication**
   - Simple HTTP POST to Communications Agent
   - Standardized request/response format
   - Error handling and graceful degradation

**Technical Focus:**
- Single-file main.py for simplicity
- Essential environment variables only
- Manual testing with curl commands
- Container-ready with minimal Dockerfile

### Phase 2: Intelligence Layer (Week 2)
**Objective**: Add LangChain agent with basic routing capabilities

**Deliverables:**
1. **LangChain Agent Setup**
   - Claude 3.5-sonnet integration
   - Basic system prompt for task routing
   - Tool calling framework setup

2. **Agent Selection Logic**
   - Simple rule-based routing tool
   - Communications Agent triggers (emails, client contact)
   - Analysis Agent triggers (document review, analysis)
   - Default fallback to Communications Agent

3. **Basic Workflow Orchestration**
   - Sequential task execution
   - Simple response aggregation
   - Context preservation between agent calls

**Technical Focus:**
- Single routing tool with clear decision logic
- Minimal prompt engineering for MVP
- Direct tool execution without complex workflows
- Basic conversation state management

### Phase 3: Production Readiness (Week 3)
**Objective**: Security, monitoring, and deployment preparation

**Deliverables:**
1. **Security Implementation**
   - Input validation and sanitization
   - Basic rate limiting
   - Environment variable validation
   - HTTPS enforcement middleware

2. **Monitoring & Observability**
   - Comprehensive health checks
   - Resource monitoring
   - Structured logging with request tracking
   - Error aggregation and reporting

3. **Deployment Configuration**
   - Production-ready Dockerfile
   - Docker Compose for local development
   - Environment configuration templates
   - Cloud Run deployment scripts

**Technical Focus:**
- Security-first approach
- Container optimization
- Production monitoring
- Deployment automation

## Technical Architecture

### Core Components

```
manager-agent/
├── main.py                    # FastAPI app + LangChain agent
├── requirements.txt           # Minimal dependencies
├── Dockerfile                 # Production container
├── .env.template              # Environment configuration
└── prompts/
    └── system_prompt.md       # Agent instructions
```

### Technology Stack
| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Web Framework** | FastAPI 0.104.1 | Async performance, automatic docs |
| **AI Framework** | LangChain 0.1.0 | Tool calling, agent patterns |
| **AI Provider** | Claude 3.5-sonnet | High-quality reasoning |
| **HTTP Client** | HTTPX 0.25.0 | Async inter-service communication |
| **Data Models** | Pydantic 2.5.0 | Type validation, serialization |
| **Runtime** | Python 3.13 | Latest performance improvements |

### Agent Communication Protocol

**Standard Request Format:**
```python
{
    "task_id": "uuid",
    "message": "user_request",
    "context": {"conversation_id": "uuid"},
    "priority": "normal"
}
```

**Standard Response Format:**
```python
{
    "task_id": "uuid",
    "response": "agent_response",
    "status": "completed|failed",
    "execution_time": 2.5,
    "context_updates": {}
}
```

## Implementation Details

### Core Agent Logic

**1. Intent Classification (Simple Rule-Based)**
```python
def classify_intent(message: str) -> str:
    """Basic keyword-based intent classification for MVP"""
    
    # Communications triggers
    comm_keywords = ["email", "send", "contact", "client", "remind", "follow up"]
    
    # Analysis triggers  
    analysis_keywords = ["analyze", "review", "document", "contract", "extract", "summarize"]
    
    message_lower = message.lower()
    
    if any(keyword in message_lower for keyword in analysis_keywords):
        return "analysis"
    else:
        return "communications"  # Default fallback
```

**2. Agent Selection Tool**
```python
class AgentSelectorTool(BaseTool):
    name = "agent_selector"
    description = "Determine which specialized agent should handle the task"
    
    def _run(self, task_description: str) -> str:
        agent_type = classify_intent(task_description)
        
        if agent_type == "communications":
            return self.delegate_to_communications(task_description)
        else:
            return self.delegate_to_analysis(task_description)
```

**3. HTTP Client Implementation**
```python
class AgentClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def delegate_task(self, request: dict) -> dict:
        try:
            response = await self.client.post(
                f"{self.base_url}/agent/task",
                json=request
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutError:
            raise AgentTimeoutError("Agent communication timed out")
        except httpx.HTTPStatusError as e:
            raise AgentCommunicationError(f"Agent error: {e.response.status_code}")
```

### Environment Configuration

**Required Variables:**
```bash
ANTHROPIC_API_KEY=sk-ant-your-key
BACKEND_URL=http://backend:8080
BACKEND_API_KEY=your-backend-key
COMMUNICATIONS_AGENT_URL=http://communications:8082
ANALYSIS_AGENT_URL=http://analysis:8083  # Optional for MVP
```

**Optional Performance Tuning:**
```bash
MANAGER_PORT=8081
AGENT_TIMEOUT_SECONDS=30
MAX_RETRY_ATTEMPTS=3
LOG_LEVEL=INFO
```

### Security Considerations

**1. Input Validation**
- Message length limits (10KB max)
- UUID format validation for IDs
- Basic XSS prevention in message content
- Request size limits to prevent DoS

**2. Authentication**
- Simple bearer token for inter-agent communication
- Environment variable validation at startup
- HTTPS enforcement in production
- Rate limiting (60 requests/minute per IP)

**3. Container Security**
- Non-root user execution
- Minimal base image with security updates
- Resource limits (1GB memory, 1 CPU)
- Health check endpoints for monitoring

## Development Workflow

### Setup Process
```bash
# 1. Clone and setup
git clone <repo>
cd manager-agent

# 2. Environment setup
cp .env.template .env
# Edit .env with actual values

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run locally
python main.py

# 5. Test basic functionality
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Send reminder to client about documents"}'
```

### Testing Strategy
```bash
# Health checks
curl http://localhost:8081/status
curl http://localhost:8081/

# Agent routing tests
curl -X POST http://localhost:8081/chat \
  -d '{"message": "Send email to John Smith"}' # → Communications

curl -X POST http://localhost:8081/chat \
  -d '{"message": "Analyze contract for risks"}' # → Analysis
```

### Deployment Process
```bash
# Container build
docker build -t manager-agent:latest .

# Local stack test
docker-compose -f docker-compose.dev.yml up

# Production deployment (Cloud Run)
gcloud run deploy manager-agent \
  --image gcr.io/$PROJECT_ID/manager-agent:latest \
  --port 8081 \
  --memory 1Gi \
  --min-instances 1 \
  --max-instances 5
```

## Success Metrics

### Functional Requirements
- ✅ Route requests to appropriate agents
- ✅ Handle agent failures gracefully
- ✅ Maintain conversation context
- ✅ Synthesize responses from multiple agents
- ✅ Support concurrent conversations

### Performance Targets
- **Response Time**: < 5 seconds for single-agent tasks
- **Availability**: 99.9% uptime
- **Throughput**: 60+ requests per minute
- **Memory Usage**: < 1GB per instance
- **Startup Time**: < 10 seconds

### Deployment Readiness
- ✅ Container security compliance
- ✅ Health check endpoints functional
- ✅ Environment configuration validated
- ✅ Inter-agent communication tested
- ✅ Error handling and logging complete

## Risk Mitigation

### Technical Risks
1. **Agent Unavailability**: Circuit breaker pattern + graceful degradation
2. **Response Latency**: Connection pooling + timeout optimization
3. **Memory Leaks**: Resource monitoring + container restarts
4. **Security Vulnerabilities**: Input validation + security middleware

### Operational Risks
1. **Deployment Complexity**: Single container + minimal configuration
2. **Monitoring Gaps**: Comprehensive health checks + structured logging
3. **Scaling Issues**: Stateless design + horizontal scaling ready
4. **Dependency Failures**: Fallback agents + offline mode support

## Future Enhancements (Post-MVP)

### Phase 4: Advanced Orchestration
- Parallel agent execution
- Complex workflow definitions
- Dynamic agent discovery
- Advanced context management

### Phase 5: Intelligence Improvements
- Machine learning-based intent classification
- Agent performance optimization
- Predictive routing
- Advanced prompt engineering

### Phase 6: Enterprise Features
- Comprehensive audit logging
- Advanced security features
- Multi-tenant support
- Integration APIs

## Conclusion

This implementation plan delivers a production-ready Manager Agent MVP in 3 weeks, focusing on essential orchestration capabilities while maintaining enterprise-grade reliability. The architecture provides a solid foundation for future enhancements while solving the immediate need for unified agent coordination in the Luceron AI eDiscovery Platform.

The lightweight approach ensures fast delivery without compromising on security, scalability, or maintainability - essential qualities for a system that will serve as the primary interface for legal professionals managing critical document workflows.
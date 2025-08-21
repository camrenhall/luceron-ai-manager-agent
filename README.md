# Luceron AI Manager Agent

Intelligent orchestration layer for the Luceron AI eDiscovery Platform. Routes tasks between specialized agents using LangChain and Claude 3.5-Sonnet for smart decision-making.

## Overview

The Manager Agent serves as the central hub for task delegation in a multi-agent eDiscovery system. It receives user requests and intelligently routes them to appropriate specialized agents (Communications, Analysis) based on the task content and context.

## Features

- **Intelligent Routing**: LangChain-powered agent selection using Claude 3.5-Sonnet
- **Circuit Breaker**: Automatic failover and recovery for agent communication
- **Rate Limiting**: Built-in protection against request flooding
- **Async Performance**: Full async/await support for high concurrency
- **Clean Architecture**: Modular design with clear separation of concerns
- **Lightweight**: Minimal dependencies focused on core functionality

## Quick Start

### Prerequisites

- Python 3.11+
- Access to Anthropic API (Claude)
- Communication and Analysis agents running (for full functionality)

### Installation

1. **Clone and navigate**:
   ```bash
   git clone <repository-url>
   cd luceron-ai-manager-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

The service will start on `http://localhost:8081`

## Configuration

### Required Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-your-api-key-here
BACKEND_URL=http://your-backend-service:8080
BACKEND_API_KEY=your-backend-api-key
```

### Optional Settings

```bash
MANAGER_PORT=8081
COMMUNICATIONS_AGENT_URL=http://communications:8082
ANALYSIS_AGENT_URL=http://analysis:8083
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60
```

## API Endpoints

### Chat Interface

**POST /chat**

Primary endpoint for user interactions. Intelligently routes requests to appropriate agents.

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Send reminder email to client about pending documents",
    "conversation_id": "optional-uuid"
  }'
```

Response:
```json
{
  "response": "Communications Agent Response: Email reminder sent successfully",
  "conversation_id": "generated-or-provided-uuid",
  "agent_used": "llm-orchestrator",
  "execution_time": 2.34
}
```

### Health Checks

**GET /** - Basic health status
**GET /status** - Simple running status for load balancers

### Agent Task Interface

**POST /agent/task**

Internal endpoint for inter-agent communication in multi-agent workflows.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Request  │───▶│  Manager Agent   │───▶│ Specialized     │
│                 │    │                  │    │ Agents          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ LangChain +      │
                       │ Claude 3.5       │
                       │ Decision Engine  │
                       └──────────────────┘
```

### Core Components

- **Agent Manager**: LangChain-based intelligent routing
- **HTTP Client**: Async communication with connection pooling
- **Circuit Breaker**: Reliability and failover handling
- **Security Middleware**: Rate limiting and request validation
- **Configuration**: Environment-based settings management

## Agent Selection Logic

The Manager Agent uses Claude 3.5-Sonnet to analyze incoming requests and determine the most appropriate specialized agent:

- **Communications Agent**: Email, client contact, reminders, notifications
- **Analysis Agent**: Document review, legal analysis, content extraction
- **Fallback**: Graceful degradation when AI routing fails

## Examples

### Communication Tasks
```bash
# Send client email
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Email John Smith about the contract review deadline"}'

# Schedule reminder
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Set up reminder for client call tomorrow at 2pm"}'
```

### Analysis Tasks
```bash
# Document analysis
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze the merger agreement for potential risks"}'

# Content extraction
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Extract key dates from the contract documents"}'
```

## Development

### Project Structure

```
src/
├── config/          # Environment and settings
├── models/          # Data validation and schemas
├── agents/          # Agent communication and AI logic
├── middleware/      # Security and request handling
├── routes/          # API endpoints
└── core/           # Application factory and HTTP client
```

### Adding New Agents

1. Update `src/agents/manager.py` with new delegation tool
2. Add agent client configuration in settings
3. Implement routing logic in the LangChain tools

### Testing

```bash
# Validate structure
ANTHROPIC_API_KEY=test BACKEND_URL=http://test BACKEND_API_KEY=test \
  python -c "from src.core.app import create_app; print('✅ Validated')"

# Test endpoints
curl http://localhost:8081/status
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8081
CMD ["python", "main.py"]
```

### Container Build

```bash
docker build -t manager-agent:latest .
docker run -p 8081:8081 --env-file .env manager-agent:latest
```

### Production Considerations

- Set appropriate `RATE_LIMIT_REQUESTS` for your traffic
- Configure `CORS_ORIGINS` for security
- Use container orchestration for scaling
- Monitor agent availability for circuit breaker health

## Dependencies

- **FastAPI**: Modern async web framework
- **HTTPX**: Async HTTP client for agent communication
- **Pydantic**: Data validation and serialization
- **LangChain**: AI agent framework and orchestration
- **LangChain-Anthropic**: Claude 3.5-Sonnet integration

## License

[Your License Here]

## Support

For issues and questions:
- Create GitHub issues for bugs
- Check logs for agent communication errors
- Verify environment variables are properly set
- Ensure dependent agents are running and accessible
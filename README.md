# Travel Agency Backend

A Python backend server powered by **LangGraph** and **FastAPI** for intelligent travel planning workflows. This API is designed to be consumed by a frontend application hosted in a separate repository.

## Project Structure

```
TravelAgencyBE/
├── src/
│   ├── api/
│   │   ├── routes/          # API endpoint definitions
│   │   │   ├── health.py    # Health check endpoint
│   │   │   └── ...          # Add more route files here (e.g., travel.py, bookings.py)
│   │   └── __init__.py
│   ├── core/
│   │   ├── config.py        # Application configuration & environment variables
│   │   └── __init__.py
│   ├── models/
│   │   ├── schemas.py       # Pydantic models for request/response validation
│   │   └── __init__.py
│   ├── services/
│   │   └── __init__.py      # Business logic & LangGraph workflows go here
│   ├── utils/
│   │   └── __init__.py      # Helper functions and utilities
│   ├── main.py              # FastAPI application entry point
│   └── __init__.py
├── tests/                   # Unit and integration tests
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore patterns
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Modern Python project configuration
└── README.md
```

## Folder Descriptions

### `src/api/routes/`
**Purpose**: Define your REST API endpoints here.

Each file should contain related endpoints using FastAPI routers:
- `health.py` - Health check endpoint (already implemented)
- Add files like `travel.py`, `bookings.py`, `chat.py` for different features

**Example**:
```python
# src/api/routes/travel.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/plan")
async def plan_trip(request: TripRequest):
    # Your endpoint logic here
    pass
```

### `src/core/`
**Purpose**: Core application configuration and settings.

- `config.py` - Centralized configuration using Pydantic Settings
  - Environment variables
  - API keys
  - CORS settings
  - LangGraph parameters

### `src/models/`
**Purpose**: Data models and schemas for validation.

- `schemas.py` - Pydantic models for:
  - API request bodies (what the frontend sends)
  - API response bodies (what you return)
  - Internal data structures
  - LangGraph state objects

**Example**:
```python
class TripRequest(BaseModel):
    destination: str
    budget: float
    preferences: list[str]
```

### `src/services/` ⭐ **LangGraph Lives Here**
**Purpose**: Business logic and LangGraph workflow implementations.

This is where you'll implement your **LangGraph agents and workflows**:
- Create graph-based workflows using LangGraph
- Define nodes (processing steps)
- Define edges (transitions between steps)
- Implement agent logic with LLMs

**Example structure**:
```python
# src/services/travel_planner.py
from langgraph.graph import StateGraph
from src.models.schemas import TravelState

class TravelPlannerGraph:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(TravelState)
        workflow.add_node("analyze_request", self._analyze)
        workflow.add_node("search_options", self._search)
        # ... more nodes
        return workflow.compile()

    async def plan_trip(self, user_input: str):
        return await self.graph.ainvoke(initial_state)
```

### `src/utils/`
**Purpose**: Reusable helper functions and utilities.

- Date/time helpers
- Data formatters
- Custom validators
- Third-party API clients

### `src/main.py`
**Purpose**: FastAPI application entry point.

- Creates and configures the FastAPI app
- Sets up CORS middleware
- Registers route handlers
- Configures logging
- Defines startup/shutdown events

### `tests/`
**Purpose**: Test files using pytest.

- Unit tests for individual functions
- Integration tests for API endpoints
- LangGraph workflow tests

## Getting Started

### 1. Setup Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and add your API keys (e.g., OPENAI_API_KEY)
```

### 4. Run the Server

**Development mode (with auto-reload)**:
```bash
python -m src.main
# or
uvicorn src.main:app --reload
```

**Production mode**:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 5. Test the API

Health check:
```bash
curl http://localhost:8000/v1/health
```

## Development Workflow

### Adding a New Feature

1. **Define the data models** in `src/models/schemas.py`
2. **Implement business logic** in `src/services/` (LangGraph workflows)
3. **Create API endpoints** in `src/api/routes/`
4. **Register the router** in `src/main.py`
5. **Write tests** in `tests/`

### Code Quality

**Format code**:
```bash
black src/ tests/
```

**Lint code**:
```bash
ruff check src/ tests/
```

**Run tests**:
```bash
pytest
```

## LangGraph Integration

### Where LangGraph Fits

LangGraph workflows are implemented in `src/services/` and exposed through FastAPI endpoints in `src/api/routes/`.

**Typical flow**:
1. Frontend sends request to `/v1/travel/plan`
2. Route handler in `src/api/routes/travel.py` receives request
3. Calls LangGraph workflow in `src/services/travel_planner.py`
4. LangGraph processes through multiple nodes/steps
5. Results returned to frontend

### Example Integration

```python
# src/services/travel_planner.py
class TravelPlanner:
    async def plan(self, request: TripRequest) -> TripPlan:
        # LangGraph workflow implementation
        pass

# src/api/routes/travel.py
from src.services.travel_planner import TravelPlanner

planner = TravelPlanner()

@router.post("/plan")
async def plan_trip(request: TripRequest):
    result = await planner.plan(request)
    return result
```

## Environment Variables

Key environment variables (see `.env.example`):

- `OPENAI_API_KEY` - Your OpenAI API key for LangChain/LangGraph
- `OPENAI_MODEL` - Model to use (default: gpt-4o-mini)
- `DEBUG` - Enable debug mode and verbose logging
- `ALLOWED_ORIGINS` - CORS allowed origins (your frontend URL)
- `RECURSION_LIMIT` - Max recursion depth for LangGraph

## API Documentation

Once running, visit:
- **Interactive docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative docs**: http://localhost:8000/redoc (ReDoc)

## Tech Stack

- **FastAPI** - Modern async Python web framework for building APIs. Provides automatic data validation, API documentation, and better performance than Flask
- **Uvicorn** - ASGI server that runs the FastAPI application and handles HTTP requests
- **LangGraph** - Workflow orchestration framework for building stateful, multi-step LLM applications
- **LangChain** - Framework for developing applications powered by language models
- **Pydantic** - Data validation using Python type hints. Automatically validates request/response data
- **pytest** - Testing framework for Python

## Best Practices

1. **Keep routes thin** - Put logic in services, not route handlers
2. **Use Pydantic models** - Validate all inputs/outputs
3. **Type hints everywhere** - Improves IDE support and catches bugs
4. **Log important events** - Use the configured logger
5. **Write tests** - Especially for LangGraph workflows
6. **Environment-based config** - Never hardcode secrets

## Contributing

1. Create feature branch
2. Implement changes following project structure
3. Add tests
4. Run linting and formatting
5. Submit pull request

## License

MIT

# Development Guide

## Install Dev Dependencies

```bash
# Linux/macOS
./setup.sh --dev

# Windows PowerShell
.\setup.ps1 -Dev
```

## Run Tests

```bash
pytest tests/ -v
```

## Run with Auto-Reload

```bash
# Linux/macOS
./run.sh --dev

# Windows PowerShell
.\run.ps1 -Dev
```

## Run as Daemon (Background Service)

### Linux/macOS

```bash
./daemon.sh start     # Start as daemon
./daemon.sh status    # Check status
./daemon.sh logs      # View logs
./daemon.sh stop      # Stop daemon
./daemon.sh restart   # Restart daemon
```

### Windows (PowerShell)

```powershell
.\daemon.ps1 -Action start     # Start as background job
.\daemon.ps1 -Action status    # Check status
.\daemon.ps1 -Action logs      # View logs
.\daemon.ps1 -Action stop      # Stop daemon
.\daemon.ps1 -Action restart   # Restart daemon
```

## Project Structure

```
reranker-serve/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── routes.py        # API routes
│   │   ├── lb_routes.py     # Load balancer routes
│   │   └── health.py        # Health endpoints
│   ├── config/
│   │   └── settings.py      # Configuration
│   ├── engine/
│   │   ├── async_engine.py  # Async batch processing
│   │   ├── request_queue.py # Request queue
│   │   └── handlers/        # Request handlers
│   ├── models/
│   │   └── reranker.py      # Reranker model
│   ├── schemas/
│   │   └── rerank.py        # Pydantic schemas
│   └── load_balancer/
│       ├── config.py        # YAML config loader
│       ├── client.py        # HTTP client
│       └── router.py        # Load balancer router
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_async_engine.py
│   ├── test_config.py
│   ├── test_handlers.py
│   ├── test_load_balancer.py
│   ├── test_models.py
│   └── test_reranker.py
├── models/                  # Model cache directory
├── logs/                    # Log files
├── setup.sh / setup.ps1     # Setup scripts
├── run.sh / run.ps1         # Run scripts
├── daemon.sh / daemon.ps1   # Daemon scripts
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

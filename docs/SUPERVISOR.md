# Supervisor Usage Guide

## Installation

Install supervisor (if not already installed):

```bash
# Ubuntu/Debian
sudo apt-get install supervisor

# macOS
brew install supervisor

# Python pip (any OS)
pip install supervisor
```

## Basic Commands

### Start Supervisor
```bash
# Start supervisord daemon
supervisord -c supervisord.conf
```

### Control the Service
```bash
# Start the reranker service
supervisorctl -c supervisord.conf start reranker-serve

# Stop the reranker service
supervisorctl -c supervisord.conf stop reranker-serve

# Restart the reranker service
supervisorctl -c supervisord.conf restart reranker-serve

# Check status
supervisorctl -c supervisord.conf status

# View logs
supervisorctl -c supervisord.conf tail reranker-serve

# Follow logs in real-time
supervisorctl -c supervisord.conf tail -f reranker-serve
```

### Stop Supervisor
```bash
# Stop all services and supervisord
supervisorctl -c supervisord.conf shutdown
```

## Configuration

The supervisor configuration is in `supervisord.conf`. Key settings:

- **Command**: Uses `.venv/bin/uvicorn` from the project directory
- **Config**: Reads from `config.yml` via `RERANKER_CONFIG_PATH` environment variable
- **Logs**: Stored in `./logs/` directory
- **Auto-restart**: Enabled by default
- **Working Directory**: Project root directory

## Troubleshooting

### Check if supervisor is running
```bash
ps aux | grep supervisord
```

### Kill stuck supervisor process
```bash
pkill supervisord
# or find and kill by PID
ps aux | grep supervisord
kill <PID>
```

### View all logs
```bash
tail -f logs/supervisord.log
tail -f logs/reranker-serve.log
tail -f logs/reranker-serve-error.log
```

## Windows Note

Supervisor doesn't officially support Windows. For Windows, consider:

1. **Use Windows Subsystem for Linux (WSL)** - Recommended
2. **Use NSSM (Non-Sucking Service Manager)** - Windows service wrapper
3. **Use Docker** with supervisor inside the container
4. **Use PowerShell scripts** with Windows Task Scheduler

### Windows Alternative: NSSM

Download NSSM and install service:
```cmd
nssm install reranker-serve "D:\Project\AIProject\reranker-serve\.venv\Scripts\uvicorn.exe" "src.main:app --host 0.0.0.0 --port 8000"
nssm set reranker-serve AppDirectory D:\Project\AIProject\reranker-serve
nssm set reranker-serve AppEnvironmentExtra RERANKER_CONFIG_PATH=D:\Project\AIProject\reranker-serve\config.yml
nssm start reranker-serve
```

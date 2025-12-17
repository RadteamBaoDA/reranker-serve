# Configuration Quick Reference

## Files Created

1. **config.yml** - Main YAML configuration file
2. **supervisord.conf** - Supervisor daemon configuration
3. **test_config_loading.py** - Test script to verify config loading
4. **CONFIG_GUIDE.md** - Comprehensive configuration documentation
5. **SUPERVISOR.md** - Supervisor usage guide

## Quick Start with config.yml

### 1. Use the default config.yml

```bash
# Already created in project root
# Modify settings as needed
nano config.yml
```

### 2. Run with config.yml

**Linux/macOS:**
```bash
./run.sh
```

**Windows:**
```powershell
.\run.ps1
```

The scripts automatically detect and use `config.yml` if present.

### 3. Test configuration

```bash
python test_config_loading.py
```

## Configuration Priority

```
Environment Variables (highest)
    ↓
config.yml
    ↓
.env file
    ↓
Default values (lowest)
```

## Common Settings

### Change Port
```yaml
server:
  port: 9000
```

### Change Model
```yaml
model:
  name: BAAI/bge-reranker-v2-m3
```

### Force CPU
```yaml
device:
  name: cpu
```

### Enable Authentication
```yaml
api:
  key: your-secret-key-here
```

### Production Settings
```yaml
server:
  workers: 4
  
async_engine:
  max_concurrent_batches: 4
  max_batch_size: 64
  
logging:
  level: info
  json_logs: true
```

## Supervisor Usage (Linux/macOS)

### Start Service
```bash
supervisord -c supervisord.conf
supervisorctl -c supervisord.conf start reranker-serve
```

### Check Status
```bash
supervisorctl -c supervisord.conf status
```

### View Logs
```bash
supervisorctl -c supervisord.conf tail -f reranker-serve
```

### Stop Service
```bash
supervisorctl -c supervisord.conf stop reranker-serve
supervisorctl -c supervisord.conf shutdown
```

## Windows Note

Supervisor doesn't support Windows. Alternatives:
- **WSL** - Use Windows Subsystem for Linux
- **NSSM** - Windows service wrapper (see SUPERVISOR.md)
- **Task Scheduler** - Use with run.ps1 script

## Verify Configuration

```bash
# Test config loading
python test_config_loading.py

# Check what config is being used
grep "model_name\|port" config.yml

# Test API
curl http://localhost:8000/health
```

## Environment Variables Override

To override config.yml temporarily:

```bash
# Single command
RERANKER_PORT=9000 ./run.sh

# Export for session
export RERANKER_PORT=9000
./run.sh
```

## More Information

- Full config options: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- Supervisor details: [SUPERVISOR.md](SUPERVISOR.md)
- API docs: [docs/api-reference.md](docs/api-reference.md)
- Deployment: [docs/deployment.md](docs/deployment.md)

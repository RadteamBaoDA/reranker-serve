# Deployment Guide

## Docker

### Using Docker Compose

```bash
docker-compose up -d
```

### Manual Build

```bash
docker build -t reranker-serve .
docker run -p 8000:8000 reranker-serve
```

### Docker with GPU

```bash
docker run --gpus all -p 8000:8000 reranker-serve
```

---

## Offline Model Usage

### Linux/macOS

1. **Download the model:**
   ```bash
   ./download_model.sh BAAI/bge-reranker-v2-m3 ./models
   ```

2. **Configure for offline use:**
   ```bash
   export RERANKER_MODEL_PATH=./models/BAAI_bge-reranker-v2-m3
   export RERANKER_USE_OFFLINE_MODE=true
   ```

3. **Run the server:**
   ```bash
   ./run.sh
   ```

### Windows

1. **Download the model (PowerShell):**
   ```powershell
   .\download_model.ps1 -ModelName "BAAI/bge-reranker-v2-m3" -OutputDir "./models"
   ```

2. **Configure for offline use:**
   ```powershell
   $env:RERANKER_MODEL_PATH="./models/BAAI_bge-reranker-v2-m3"
   $env:RERANKER_USE_OFFLINE_MODE="true"
   ```

3. **Run the server:**
   ```powershell
   .\run.ps1
   ```

---

## Apple Silicon (MPS) Optimization

The service automatically detects and uses MPS on Apple Silicon Macs:

- Automatic MPS fallback to CPU for unsupported operations
- Optimized memory management for MPS
- FP32 precision for stability on MPS

Verify MPS support:
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

## Troubleshooting

### Windows: PyTorch DLL Error

If you encounter `[WinError 1114] A dynamic link library (DLL) initialization routine failed`:

1. **Install Visual C++ Redistributable:**
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart your terminal

2. **Run the fix script:**
   ```powershell
   .\fix-pytorch.ps1
   ```

### Verify PyTorch Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
```

### Verify Device Support

```bash
# CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# MPS (Apple Silicon)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

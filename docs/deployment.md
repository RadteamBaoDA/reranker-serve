# Deployment Guide

## Docker

Two image variants ship in the repo:

| Variant | Base | Architectures | Dockerfile | Compose profile |
|---|---|---|---|---|
| CPU | `python:3.11-slim-bookworm` | `linux/amd64`, `linux/arm64` | `Dockerfile` | default |
| CUDA | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | `linux/amd64` only | `Dockerfile.cuda` | `cuda` |

### CPU (multi-arch)

```bash
# Default compose service runs the CPU image
docker compose up -d

# Local multi-arch build with buildx
docker buildx build --platform linux/amd64,linux/arm64 -t reranker-serve:cpu .
```

### NVIDIA CUDA

```bash
# Compose (requires nvidia-container-toolkit on the host)
docker compose --profile cuda up reranker-cuda

# Or build/run by hand
docker build -f Dockerfile.cuda -t reranker-serve:cuda .
docker run --gpus all -p 8000:8000 reranker-serve:cuda
```

The CUDA image preinstalls CUDA-12.4 torch wheels and defaults `RERANKER_DEVICE=cuda` + `RERANKER_USE_FP16=true`.

### Continuous integration

`.github/workflows/docker.yml` builds both variants with `docker buildx` and pushes to GHCR on every push to `main` and on tag pushes. Tagged images are published as `ghcr.io/<owner>/reranker-serve:<tag>-cpu` and `:<tag>-cuda`.

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

# Offline / Air-Gapped Install (PyTorch GPU + project deps)

When the target machine has no internet — common for production GPU boxes
behind a firewall — you build a wheel bundle on a connected machine, copy it
across, and install with `pip install --no-index`. The repo ships a helper
at `scripts/offline-install.sh` that does both halves.

## Prerequisites

**On the internet-connected host** (the "builder"):
- Same Python major.minor as the target (e.g. both 3.11 or both 3.12).
- Same OS/CPU architecture (Windows x64 → `win_amd64`; Linux x64 →
  `linux_x86_64`; macOS arm64 → `macosx_11_0_arm64`).
- `pip >= 22`.

**On the target machine:**
- NVIDIA driver installed; `nvidia-smi` works. The PyTorch CUDA wheel bundles
  the CUDA runtime and cuDNN, so a separate CUDA Toolkit install is **not**
  required.
- Python venv activated.

## Step 1 — Pick the CUDA channel

Run `nvidia-smi` on the **target** and read the top-right "CUDA Version"
field. That number is the maximum CUDA the driver supports. Map to a channel:

| Driver supports | Channel  |
|-----------------|----------|
| CUDA 12.6+      | `cu126`  |
| CUDA 12.4       | `cu124`  |
| CUDA 12.1       | `cu121`  |
| CUDA 11.8       | `cu118`  |

Pick the highest channel your driver supports. The script defaults to
`cu124`; override with `--cuda cuXXX`.

## Step 2 — Build the wheel bundle (on the connected host)

```bash
./scripts/offline-install.sh download
# or with overrides:
./scripts/offline-install.sh download --cuda cu126 --python 3.12 --platform win_amd64
```

The script runs `pip download` twice:
1. `torch` from the CUDA index (`https://download.pytorch.org/whl/<channel>`),
   pulling its transitive deps.
2. The project's `requirements.txt` from regular PyPI.

Output lands in `./offline-wheels/`. Expect ~2.5–3 GB total (the torch CUDA
wheel itself is most of that). The script prints a summary at the end.

## Step 3 — Transfer to the target

Copy the entire `offline-wheels/` directory across along with the repo
checkout. USB, SCP, rsync, internal artifact store — whichever your ops
policy allows.

```bash
# Example over scp
scp -r offline-wheels reranker.internal:/srv/reranker/
scp -r reranker-serve  reranker.internal:/srv/
```

## Step 4 — Install on the target

```bash
cd /srv/reranker-serve
source .venv/bin/activate         # or .venv\Scripts\activate on Windows
./scripts/offline-install.sh install --dir /srv/reranker/offline-wheels
```

Equivalent manual command:
```bash
pip install --no-index --find-links=/srv/reranker/offline-wheels torch
pip install --no-index --find-links=/srv/reranker/offline-wheels -r requirements.txt
```

The `--no-index` flag stops pip from reaching out to PyPI; `--find-links`
makes it resolve everything from the local folder.

## Step 5 — Verify

The script ends with a verification step that prints torch version, CUDA
status, and device list. You can also run it standalone:

```bash
./scripts/offline-install.sh verify
```

Expected output (your versions will vary):
```
torch:              2.5.1+cu124
torch.version.cuda: 12.4
cuda_available:     True
device_count:       1
  [0] NVIDIA RTX A6000
```

If `cuda_available: False` after a clean install, something is wrong with
either the wheel selection or the driver. See troubleshooting below.

## Windows PowerShell equivalents

The script is bash; on Windows run it via Git Bash, WSL, or translate to
PowerShell:

```powershell
# Download (on connected host)
pip download torch `
  --index-url https://download.pytorch.org/whl/cu124 `
  --extra-index-url https://pypi.org/simple `
  --python-version 3.12 `
  --platform win_amd64 `
  --only-binary=:all: `
  -d .\offline-wheels

pip download `
  -r requirements.txt `
  --python-version 3.12 --platform win_amd64 --only-binary=:all: `
  -d .\offline-wheels

# Install (on target)
.\.venv\Scripts\activate
pip install --no-index --find-links=.\offline-wheels torch
pip install --no-index --find-links=.\offline-wheels -r requirements.txt

# Verify
python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available(), '| cuda_ver:', torch.version.cuda)"
```

## Troubleshooting

**`cuda_available: False` after install**
1. Confirm `nvidia-smi` works on the target. If not, the driver isn't
   installed correctly — fix that first.
2. Run `python -c "import torch; print(torch.__file__)"`. If the path ends
   in `+cpu`, the bundle accidentally grabbed the CPU wheel. Rebuild on the
   connected host with explicit `--cuda cuXXX` and confirm the wheel
   filenames in `offline-wheels/` contain `+cuXXX`.
3. Check no env var is masking the GPU: `echo $CUDA_VISIBLE_DEVICES`
   (or `$env:CUDA_VISIBLE_DEVICES` in PowerShell). It should be unset or
   contain a valid index (e.g. `0`).

**`ModuleNotFoundError: Could not import module 'PreTrainedModel'` from
sentence-transformers**
That's the transformers package failing to expose torch-dependent classes
because torch isn't importable. Install torch first (see step 4), then
re-run the project install — the error will disappear.

**Platform tag mismatch**
`pip download` will refuse if the target platform doesn't match the local
platform. Pass the right `--platform` explicitly (e.g. `--platform win_amd64`
when the builder is Linux but the target is Windows). Same for `--python`.

**Wheel index changes**
PyTorch occasionally drops older minor versions from the index. If you need
a specific torch version, pin it: `pip download torch==2.5.1+cu124 ...`.

## How the script picks defaults

- `--python` — auto-detected from `python --version` on the builder. Override
  if the builder and target use different Python versions.
- `--platform` — auto-detected from `uname -s`. Override when
  cross-targeting (e.g. building on Linux for a Windows target).
- `--cuda` — defaults to `cu124`. Override based on the target's driver.
- `--dir` — defaults to `./offline-wheels`. Override if you need to land
  the bundle elsewhere.

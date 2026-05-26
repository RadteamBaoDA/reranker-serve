#!/usr/bin/env bash
# offline-install.sh — install PyTorch (CUDA) + project deps without internet
# on the target machine.
#
# Workflow:
#   1. On a machine WITH internet:
#        ./scripts/offline-install.sh download
#      Wheels land in ./offline-wheels.
#   2. Copy the offline-wheels/ folder (and this repo) to the air-gapped target.
#   3. On the TARGET (no internet):
#        ./scripts/offline-install.sh install
#
# Usage:
#   ./scripts/offline-install.sh download [--cuda cu124] [--python 3.12] \
#                                          [--platform win_amd64] [--dir ./offline-wheels]
#   ./scripts/offline-install.sh install  [--dir ./offline-wheels]
#   ./scripts/offline-install.sh verify
#
# Defaults: cuda=cu124, python=auto-detected from `python --version`,
#           platform=auto-detected (linux_x86_64 / win_amd64 / macosx).

set -euo pipefail

WHEELS_DIR="./offline-wheels"
CUDA_CHANNEL="cu124"
PY_VERSION=""
PLATFORM=""

usage() {
    sed -n '2,18p' "$0"
    exit 1
}

detect_python_version() {
    python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
}

detect_platform() {
    case "$(uname -s)" in
        Linux*)   echo "linux_x86_64" ;;
        Darwin*)  echo "macosx_11_0_arm64" ;;
        MINGW*|MSYS*|CYGWIN*) echo "win_amd64" ;;
        *)        echo "linux_x86_64" ;;
    esac
}

cmd_download() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cuda)      CUDA_CHANNEL="$2"; shift 2 ;;
            --python)    PY_VERSION="$2";   shift 2 ;;
            --platform)  PLATFORM="$2";     shift 2 ;;
            --dir)       WHEELS_DIR="$2";   shift 2 ;;
            *) echo "unknown flag: $1" >&2; exit 2 ;;
        esac
    done

    [[ -z "$PY_VERSION" ]] && PY_VERSION=$(detect_python_version)
    [[ -z "$PLATFORM"  ]] && PLATFORM=$(detect_platform)

    echo ">>> Downloading wheels"
    echo "    cuda channel: $CUDA_CHANNEL"
    echo "    python:       $PY_VERSION"
    echo "    platform:     $PLATFORM"
    echo "    dest:         $WHEELS_DIR"
    echo

    mkdir -p "$WHEELS_DIR"

    # 1) torch + its transitive deps from the CUDA channel (with PyPI as backstop)
    pip download torch \
        --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}" \
        --extra-index-url https://pypi.org/simple \
        --python-version "$PY_VERSION" \
        --platform "$PLATFORM" \
        --only-binary=:all: \
        -d "$WHEELS_DIR"

    # 2) project deps (transformers, sentence-transformers, fastapi, ...)
    if [[ -f requirements.txt ]]; then
        pip download \
            -r requirements.txt \
            --python-version "$PY_VERSION" \
            --platform "$PLATFORM" \
            --only-binary=:all: \
            -d "$WHEELS_DIR"
    else
        echo "(no requirements.txt found; skipping project deps)" >&2
    fi

    echo
    echo ">>> Done. Wheels: $(ls -1 "$WHEELS_DIR" | wc -l) files, total $(du -sh "$WHEELS_DIR" 2>/dev/null | cut -f1)"
    echo "    Copy '$WHEELS_DIR' (and this repo) to the air-gapped target."
}

cmd_install() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dir) WHEELS_DIR="$2"; shift 2 ;;
            *) echo "unknown flag: $1" >&2; exit 2 ;;
        esac
    done

    if [[ ! -d "$WHEELS_DIR" ]]; then
        echo "ERROR: wheels dir not found: $WHEELS_DIR" >&2
        echo "Run './scripts/offline-install.sh download' on an internet host first," >&2
        echo "then copy the directory to this machine." >&2
        exit 1
    fi

    echo ">>> Installing torch from $WHEELS_DIR"
    pip install --no-index --find-links="$WHEELS_DIR" torch

    if [[ -f requirements.txt ]]; then
        echo ">>> Installing project deps from $WHEELS_DIR"
        pip install --no-index --find-links="$WHEELS_DIR" -r requirements.txt
    fi

    cmd_verify
}

cmd_verify() {
    echo
    echo ">>> Verifying torch install"
    python - <<'PY'
import sys
try:
    import torch
except ImportError as e:
    print(f"FAIL torch not importable: {e}", file=sys.stderr)
    sys.exit(1)

cuda_ok = torch.cuda.is_available()
print(f"torch:          {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"cuda_available: {cuda_ok}")
if cuda_ok:
    print(f"device_count:   {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
else:
    print("WARNING: torch is installed but CUDA is not available.", file=sys.stderr)
    print("Check (a) you grabbed a cuXXX wheel, not the CPU wheel,", file=sys.stderr)
    print("      (b) NVIDIA driver is installed and `nvidia-smi` works,", file=sys.stderr)
    print("      (c) no CUDA_VISIBLE_DEVICES='' env var is masking the GPU.", file=sys.stderr)
    sys.exit(2)
PY
}

main() {
    [[ $# -lt 1 ]] && usage
    cmd="$1"; shift
    case "$cmd" in
        download) cmd_download "$@" ;;
        install)  cmd_install  "$@" ;;
        verify)   cmd_verify ;;
        -h|--help|help) usage ;;
        *) echo "unknown command: $cmd" >&2; usage ;;
    esac
}

main "$@"

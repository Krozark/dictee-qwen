#!/usr/bin/env bash
set -euo pipefail

echo "== Installing system dependencies =="
# python-xlib is pure Python (no C extension), so no X11 dev headers are
# needed for it. python3-gi / gir1.2-gtk-3.0 provide the PyGObject runtime
# used via `uv venv --system-site-packages` below, since PyGObject itself
# is not uv-managed (building it from source needs root for girepository
# dev headers).
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    xdotool \
    xclip \
    portaudio19-dev \
    python3-gi \
    gir1.2-gtk-3.0

echo "== Checking for uv =="
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first, then re-run this script."
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "See https://docs.astral.sh/uv/getting-started/installation/ for other options."
    exit 1
fi

echo "== Creating Python virtual environment (with system site packages for PyGObject) =="
uv venv --system-site-packages

echo "== Installing Python dependencies =="
uv sync --dev

echo "== Verifying PyGObject is importable =="
uv run python3 -c "import gi; print('PyGObject OK')"

echo "== Verifying CUDA availability =="
uv run python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

read -rp "Build whisper.cpp with CUDA support as a fallback backend? [y/N] " build_whisper
if [[ "${build_whisper:-N}" =~ ^[Yy]$ ]]; then
    echo "== Cloning and building whisper.cpp with CUDA =="
    git clone https://github.com/ggerganov/whisper.cpp.git vendor/whisper.cpp
    (
        cd vendor/whisper.cpp
        GGML_CUDA=1 make -j"$(nproc)"
        ./models/download-ggml-model.sh large-v3-turbo
    )
    echo "Start the backend with:"
    echo "  vendor/whisper.cpp/build/bin/whisper-server -m vendor/whisper.cpp/models/ggml-large-v3-turbo.bin --port 8080"
fi

echo "== Done. Activate with: source .venv/bin/activate (or just use 'uv run <command>' directly) =="

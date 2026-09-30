#!/usr/bin/env bash
# One command to set up and run shortform-notes. Safe to re-run; each run refreshes yt-dlp.
#   ./start.sh            first run: asks whether to set up in the browser or in this terminal
#   ./start.sh web        open the browser setup and import page
#   ./start.sh setup      run the terminal setup wizard
#   ./start.sh <url>      import a link from the terminal
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager, one time)."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing Python 3.12 and shortform-notes dependencies. The first run takes about a minute."
uv python install 3.12 --quiet 2>/dev/null || true
# cuda: cuBLAS/cuDNN wheels for GPU Whisper on Linux; NVIDIA publishes none for macOS.
extras="--extra all --extra local --extra ocr"
[ "$(uname -s)" = "Linux" ] && extras="$extras --extra cuda"
# shellcheck disable=SC2086
uv sync --python 3.12 $extras --upgrade-package yt-dlp --quiet

if [ "$#" -gt 0 ]; then
  exec uv run shortform-notes "$@"
fi

CONFIG="${SHORTFORM_NOTES_CONFIG:-$HOME/.config/shortform-notes/config.env}"
if [ -f "$CONFIG" ] || [ ! -t 0 ]; then
  exec uv run shortform-notes web
fi

echo
echo "How do you want to set up shortform-notes?"
echo "  1) In the browser (recommended if you are not used to the terminal)"
echo "  2) In this terminal"
read -r -p "Choose 1 or 2 [1]: " choice
case "${choice:-1}" in
  2) exec uv run shortform-notes setup ;;
  *) exec uv run shortform-notes web ;;
esac

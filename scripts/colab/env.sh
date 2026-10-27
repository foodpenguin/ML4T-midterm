# Sourced by the helpers: repo root from this file's location, colab CLI from COLAB_BIN or PATH.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${COLAB_BIN:-$(command -v colab)}"
S="${COLAB_SESSION:-ml4t-deep}"
if [ -z "$C" ] || [ ! -x "$C" ]; then
  echo "colab CLI not found: install google-colab-cli or set COLAB_BIN" >&2
  exit 1
fi
quiet() { grep -v -i "update\|available\|silence" || true; }

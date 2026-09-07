#!/usr/bin/env bash
# Launch the local learning guide without installing inference dependencies.
set -euo pipefail

DOCS_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCS_PYTHON="${MINISGL_DOCS_PYTHON:-python3}"

if ! command -v "$DOCS_PYTHON" >/dev/null 2>&1; then
  printf '找不到 Python 3。请先安装 Python 3，或设置 MINISGL_DOCS_PYTHON。\n' >&2
  exit 1
fi

exec "$DOCS_PYTHON" "$DOCS_ROOT/scripts/serve_docs.py" "$@"

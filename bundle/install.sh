#!/usr/bin/env bash
set -euo pipefail
P2P_PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$P2P_PACKAGE_ROOT/installer.py" "$@"

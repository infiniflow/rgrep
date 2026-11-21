#!/usr/bin/env bash
# Fetch the onnxruntime shared library that matches the onnxruntime_go headers.
#
#   ./scripts/fetch_onnxruntime.sh            # default version
#   RGREP_PROXY=socks5h://127.0.0.1:7891 ./scripts/fetch_onnxruntime.sh
#
# The version must match ORT_API_VERSION in onnxruntime_go/onnxruntime_c_api.h.
set -euo pipefail

VER="${1:-1.29.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime/lib"
mkdir -p "$DEST"

CURL=(curl -sSL)
if [[ -n "${RGREP_PROXY:-}" ]]; then
  CURL+=(--proxy "$RGREP_PROXY")
fi

echo "resolving onnxruntime $VER wheel from PyPI"
URL="$("${CURL[@]}" "https://pypi.org/pypi/onnxruntime/$VER/json" | python3 - <<'PY'
import json, sys
d = json.load(sys.stdin)
urls = [u["url"] for u in d["urls"] if u["filename"].endswith(".whl")]
prefs = ("macosx", "arm64") if sys.platform == "darwin" else ("manylinux", "x86_64")
for want in prefs:
    hit = [u for u in urls if want in u]
    if hit:
        print(sorted(hit)[-1]); break
else:
    sys.exit("no matching wheel")
PY
)"
echo "downloading $URL"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
"${CURL[@]}" -o "$TMP/ort.whl" "$URL"

case "$(uname -s)" in
  Darwin) LIB="libonnxruntime.$VER.dylib" ;;
  Linux)  LIB="libonnxruntime.so.$VER" ;;
  *) echo "unsupported platform"; exit 1 ;;
esac

unzip -o -j "$TMP/ort.whl" "onnxruntime/capi/$LIB" -d "$DEST" >/dev/null
echo "installed $DEST/$LIB"
"$ROOT/bin/rgrep" version >/dev/null 2>&1 || true
echo "done — restart the daemon so it picks up the new library"

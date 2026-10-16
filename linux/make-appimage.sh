#!/usr/bin/env bash
# Packs build-linux/shadps4 + linux/play.sh (as AppRun) + the SotC config into an AppImage.
# Usage: bash linux/make-appimage.sh [output.AppImage]
# Downloads appimagetool into build-linux/ the first time.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
BIN="$ROOT/build-linux/shadps4"
OUT="${1:-$ROOT/build-linux/TT-shadPS4-SotC-Linux-x86_64.AppImage}"
[ -x "$BIN" ] || { echo "Build first: bash linux/build.sh"; exit 1; }
TOOL="$ROOT/build-linux/appimagetool"
if [ ! -x "$TOOL" ]; then
  curl -fL -o "$TOOL" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$TOOL"
fi
APPDIR="$ROOT/build-linux/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/sotc-shadps4" "$APPDIR/usr/share/applications" \
  "$APPDIR/usr/share/icons/hicolor/scalable/apps"
strip --strip-all -o "$APPDIR/usr/bin/shadps4" "$BIN"
cp "$ROOT/linux/play.sh" "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun" "$APPDIR/usr/bin/shadps4"
cp "$ROOT/linux/sotc/CUSA08809.workarounds.json" "$ROOT/linux/sotc/sotc-60fps.xml" "$APPDIR/usr/share/sotc-shadps4/"
cp "$ROOT/linux/sotc/sotc-shadps4.desktop" "$APPDIR/"
cp "$ROOT/linux/sotc/sotc-shadps4.desktop" "$APPDIR/usr/share/applications/"
cp "$ROOT/linux/sotc/sotc-shadps4.svg" "$APPDIR/"
cp "$ROOT/linux/sotc/sotc-shadps4.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/"
ARCH=x86_64 "$TOOL" --appimage-extract-and-run --comp zstd \
  --mksquashfs-opt -Xcompression-level --mksquashfs-opt 19 "$APPDIR" "$OUT"
echo "AppImage: $OUT"

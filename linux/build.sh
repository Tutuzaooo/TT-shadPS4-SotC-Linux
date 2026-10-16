#!/usr/bin/env bash
# Builds this fork for Linux with clang + ThinLTO into build-linux/.
# Usage: bash linux/build.sh            portable build (any x86-64-v3 CPU: Haswell / Zen 1 or newer)
#        bash linux/build.sh --native   tuned for this machine's CPU only (a bit faster)
# Needs: clang, lld, cmake, ninja or make, git, and the usual shadPS4 dependencies
# (Arch/CachyOS: sudo pacman -S clang lld cmake ninja sdl3 vulkan-headers vulkan-icd-loader)
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
ARCH_FLAGS="-march=x86-64-v3"
[ "${1:-}" = --native ] && ARCH_FLAGS="-march=native"
git -C "$ROOT" submodule update --init --recursive
GEN=()
command -v ninja >/dev/null 2>&1 && GEN=(-G Ninja)
cmake -S "$ROOT" -B "$ROOT/build-linux" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_FLAGS="$ARCH_FLAGS -flto=thin" -DCMAKE_CXX_FLAGS="$ARCH_FLAGS -flto=thin" \
  -DCMAKE_EXE_LINKER_FLAGS="-flto=thin -fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-flto=thin -fuse-ld=lld"
cmake --build "$ROOT/build-linux" -j"$(nproc)"
echo
echo "Done: $ROOT/build-linux/shadps4"
echo "Play: bash linux/play.sh --game \"/path/CUSA08809/eboot.bin\""
echo "AppImage: bash linux/make-appimage.sh"

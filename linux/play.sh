#!/usr/bin/env bash
# Shadow of the Colossus launcher for this build: every performance option of the final build on.
# Also used as AppRun inside the AppImage (linux/make-appimage.sh).
#
# Usage:  bash linux/play.sh --game "/path/CUSA08809/eboot.bin"   (or -g, or just the path)
#
# Any option can be changed from the environment, for example:
#   PINK_RB_ASYNC=0          no asynchronous readbacks
#   PINK_60FPS=0             no 60 FPS patch (illusion)
#   PINK_SOMBRA=original     original sun shadow distance (default here: 50)
#   PINK_GAME_ARGS=0         leave the game's CommandLineArgs.txt alone
#   PINK_KEEP_CONFIG=1       do not write the SotC config into the user folder
#   PINK_PERF=1              performance stats every 10 s in the log
#   SHADPS4_BIN=/path        another shadps4 binary
HERE="$(dirname "$(readlink -f "$0")")"
if [ -x "$HERE/usr/bin/shadps4" ]; then          # inside the AppImage
  BIN="$HERE/usr/bin/shadps4"; SHARE="$HERE/usr/share/sotc-shadps4"
else                                              # from the repository
  BIN="${SHADPS4_BIN:-$HERE/../build-linux/shadps4}"; SHARE="$HERE/sotc"
fi

# ---- Emulator options (code) ----------------------------------------------------------
export PINK_VRAM_LIMIT_MB="${PINK_VRAM_LIMIT_MB:-4500}"
export PINK_DRAW_PIPE="${PINK_DRAW_PIPE:-7}"            # GPU commands on two threads, mode 7
export PINK_PIPE_EXTRA="${PINK_PIPE_EXTRA-all}"         # WAIT_REG_MEM/RELEASE_MEM/EOS without draining
export PINK_LABEL_SPIN_US="${PINK_LABEL_SPIN_US:-100}" PINK_REC_SPIN_US="${PINK_REC_SPIN_US:-20}"
export PINK_RB_BATCH="${PINK_RB_BATCH:-0}"
export PINK_FILL_HLE="${PINK_FILL_HLE:-1}" PINK_FILL_HLE_MAX_KB="${PINK_FILL_HLE_MAX_KB:-64}"
export PINK_FILL_MIRROR="${PINK_FILL_MIRROR:-1}"
export PINK_STREAM_REUSE="${PINK_STREAM_REUSE:-1}" PINK_STREAM_REUSE_TICKS="${PINK_STREAM_REUSE_TICKS:-16}"
export PINK_VERTEX_MEMO="${PINK_VERTEX_MEMO:-1}" PINK_SPEC_MEMO="${PINK_SPEC_MEMO:-1}"
export PINK_DIRECT_DRAWS="${PINK_DIRECT_DRAWS:-1}" PINK_FAST_BIND="${PINK_FAST_BIND:-1}"
export PINK_RB_ASYNC="${PINK_RB_ASYNC:-1}" PINK_FAULT_WINDOW_KB="${PINK_FAULT_WINDOW_KB:-256}"
[ "${PINK_PERF:-0}" = 1 ] && export SHADPS4_PERF_STATS=1

# ---- Arguments: find the game, see whether a patch is given ---------------------------------------
GAME="" HAS_PATCH=0 prev=""
for a in "$@"; do
  case "$prev" in -g|--game) GAME="$a" ;; esac
  case "$a" in
    -p|--patch|--patch=*|-p=*) HAS_PATCH=1 ;;
    --game=*) GAME="${a#--game=}" ;;
    -*) ;;
    *) [ -z "$GAME" ] && [ -e "$a" ] && GAME="$a" ;;
  esac
  prev="$a"
done

# ---- SotC config (same user folder rule as the emulator) ----------------------
if [ -d "$PWD/user" ]; then UDIR="$PWD/user"
elif [ -n "${XDG_DATA_HOME:-}" ]; then UDIR="$XDG_DATA_HOME/shadPS4"
else UDIR="$HOME/.local/share/shadPS4"; fi
if [ "${PINK_KEEP_CONFIG:-0}" != 1 ]; then
  mkdir -p "$UDIR/custom_configs"
  # Applied after the user's CUSA08809.json: only the SotC options change
  cp -f "$SHARE/CUSA08809.workarounds.json" "$UDIR/custom_configs/CUSA08809.workarounds.json"
fi

# ---- 60 FPS patch (illusion) ---------------------------------------------------------------
PATCH=()
if [ "${PINK_60FPS:-1}" = 1 ] && [ "$HAS_PATCH" = 0 ]; then
  PATCH=(--patch "$SHARE/sotc-60fps.xml")
fi

# ---- Game engine arguments (CommandLineArgs.txt, restored on exit) -------------------
restore_args() { :; }
if [ "${PINK_GAME_ARGS:-1}" = 1 ] && [ -n "$GAME" ]; then
  if [ -d "$GAME" ]; then GDIR="$GAME"; else GDIR="$(dirname "$GAME")"; fi
  ARGS="$(find "$GDIR" -maxdepth 1 -iname 'commandlineargs.txt' 2>/dev/null | head -1)"
  [ -n "$ARGS" ] || ARGS="$GDIR/CommandLineArgs.txt"
  BAK="$ARGS.original-pink"
  restore_args() {
    if [ -f "$BAK" ]; then mv -f "$BAK" "$ARGS"
    elif [ -f "$ARGS.criado-pink" ]; then rm -f "$ARGS" "$ARGS.criado-pink"; fi
  }
  restore_args   # left over from an interrupted run
  if [ -d "$GDIR" ] && [ -w "$GDIR" ]; then
    if [ -f "$ARGS" ]; then cp -p "$ARGS" "$BAK"; else : > "$ARGS.criado-pink"; fi
    EXTRA=""
    if [ "${PINK_60FPS:-1}" = 1 ]; then EXTRA="+targetFramesPerSeconds=60 +fixedTimeStep=false"; fi
    SOMBRA="${PINK_SOMBRA:-50}"   # sun shadows up to 50: near shadows unchanged
    [ "$SOMBRA" = original ] || EXTRA="+sunShadowMaxDistance=$SOMBRA $EXTRA"
    KEYS='renderErrorMsgs|doIndexBufferCulling|doSunShadowIndexBufferCulling|fog_enableParticles|particleWaterFogScale|targetFramesPerSeconds|fixedTimeStep|sunShadowMaxDistance'
    {
      [ -f "$BAK" ] && grep -v -E "^\\+($KEYS)=" "$BAK"
      printf '%s\n' '+renderErrorMsgs=0' '+doIndexBufferCulling=0' \
        '+doSunShadowIndexBufferCulling=0' '+fog_enableParticles=0' '+particleWaterFogScale=0'
      for a in $EXTRA; do printf '%s\n' "$a"; done
    } > "$ARGS.novo" && mv -f "$ARGS.novo" "$ARGS"
    trap restore_args EXIT
  else
    echo "SotC shadPS4: game folder is read-only; running without the engine arguments" >&2
  fi
fi

[ -x "$BIN" ] || { echo "shadps4 not found at $BIN (build it: bash linux/build.sh)" >&2; exit 1; }
"$BIN" "${PATCH[@]}" "$@"
STATUS=$?
restore_args
trap - EXIT
exit $STATUS

# TT-shadPS4 SotC Linux

A Linux build of [Pink-shadPS4](https://github.com/luizgustavs/Pink-shadPS4) tuned for
**Shadow of the Colossus** (CUSA08809, v01.01), made by Tutuzaooo together with Claude (Anthropic).

Tested on CachyOS (kernel 7.2), KDE Wayland, RTX 5060 8 GB, i5-12400F, 1440p:
heavy areas went from 18–26 fps to around 30 fps or more; light and medium areas reach 50–60 fps.

## Quick start

```bash
bash linux/build.sh                 # or: bash linux/build.sh --native (this CPU only, a bit faster)
bash linux/play.sh --game "/path/to/CUSA08809/eboot.bin"
bash linux/make-appimage.sh         # optional: build-linux/TT-shadPS4-SotC-Linux-x86_64.AppImage
```

`play.sh` (also the AppImage's launcher) turns on every option of the final build, writes the SotC
settings to `custom_configs/CUSA08809.workarounds.json` in the emulator's user folder, applies the
60 FPS patch and adjusts the game's `CommandLineArgs.txt` (restored when the game closes).
The user folder is `./user` in the current folder if it exists, otherwise `~/.local/share/shadPS4`.

Each piece can be turned off from the environment, e.g. `PINK_RB_ASYNC=0 bash linux/play.sh ...`
(the full list is at the top of `linux/play.sh`).

## What this fork adds

- **Two-thread GPU command processing ("draw pipe")**: one thread decodes the PS4 command stream and
  picks pipelines, another records the Vulkan commands.
- **"Present when ready" frame pacing**: no more 66–100 ms frametime spikes.
- **VRAM-aware texture garbage collection**: decides how hard to collect from the free VRAM the
  driver reports (no more out-of-memory crashes, textures no longer reload on every camera turn).
- **Filtered hot readbacks**: windows the game reads again and again come back together.
- **Fill HLE in command order**: a per-frame table-fill compute shader runs on the CPU, with the GPU
  copy written at the right point of the command stream; large fills stay on the GPU and are also
  mirrored to guest memory.
- **Constant-buffer copy reuse**: identical small constant buffers are not copied again
  (20–30% less work per draw).
- **Binding memos**: vertex/index bindings, shader specialization lookups and written-buffer
  texture checks are skipped when nothing changed.
- **Asynchronous readbacks**: the game thread waits for the GPU, the command thread keeps working.
- **DMA off without black blocks**, plus measurement tooling (per-draw timings, readback stats).

## Built on (thanks!)

- [shadPS4](https://github.com/shadps4-emu/shadPS4): the emulator
- [Pink-shadPS4](https://github.com/luizgustavs/Pink-shadPS4) by luizgustavs: the SotC-focused base
- [ShadowOfTheColossus-PC](https://github.com/AjncSystem/ShadowOfTheColossus-PC) by AjncSystem:
  direct draws, the fill HLE and async readback ideas, two bug fixes from its pull requests
- [AjncSystem's SotC shadPS4 build](https://github.com/AjncSystem/ShadPS4-ShadowoftheColossus-Build-customizado-97-corrigida):
  the runtime-indexed constant window fix (`609b4dd2`) that makes DMA off usable
- [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc) by deadinside28: VRAM ideas
  (texture collector, page-fault window)
- illusion's 60 FPS patch for Shadow of the Colossus

Licensed under GPL-2.0, like shadPS4.

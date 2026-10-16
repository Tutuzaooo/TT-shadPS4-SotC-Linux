# Shadow of the Colossus on Linux: building a faster shadPS4 for it, together with Claude

Over the last couple of weeks I worked with Claude (Anthropic's AI) on getting Shadow of the
Colossus (CUSA08809, v01.01) to run well on Linux through shadPS4. Here's the whole process: what we
built ourselves and what we brought in from other projects, with credit where it's due.

**My setup:** CachyOS (kernel 7.2), KDE Wayland, RTX 5060 8 GB, i5-12400F (6c/12t), 1440p.

## Where it started

Official shadPS4 ran SotC badly and with visual bugs. **Pink-shadPS4** (by luizgustavs) ran much
better, but it's Windows-only. We started from there. The first versions crashed with
`ErrorOutOfDeviceMemory` within seconds in the opening area, and even once they stopped crashing,
they ran at only 18–26 fps with the GPU about 25% busy. The emulator was bottlenecked on the CPU,
not the GPU.

## What we built ourselves

- **Linux build of Pink-shadPS4:** recompiled and fixed for Linux (Fedora, then CachyOS), with
  launcher scripts and a PGO + LTO build setup.
- **Two-thread GPU command processing ("draw pipe"):** one thread decodes the PS4 command stream
  and picks pipelines, while a second thread records the Vulkan commands. We tested seven modes and
  kept the fastest stable one.
- **"Present when ready" frame pacing:** fixed constant frametime spikes that reached 66–100 ms.
- **VRAM-aware texture garbage collector:** it decides how hard to collect from the free VRAM the
  driver reports. This ended the VRAM crashes and stopped textures from reloading every time the
  camera turned.
- **Filtered hot readbacks:** windows the game reads repeatedly are brought back together, but only
  when read 3+ times in a row and up to 4 MB, which removed the glitches and stutters the
  unfiltered version caused.
- **Fill HLE done in command order:** SotC fills small tables every frame with a compute shader,
  and we now do that fill on the CPU. Our part was writing the GPU copy at the exact point in the
  command stream, which fixed the texture flicker, and keeping large fills on the GPU, which kept
  the frame rate.
- **Constant-buffer copy reuse:** profiling showed each draw spent about 2.2 µs copying small
  constant buffers that were byte-for-byte identical to the previous copy almost 100% of the time.
  Reusing the copy cut per-draw cost by 20–30%.
- **Binding memos:** vertex/index buffer bindings, shader specialization lookups and written-buffer
  texture checks are skipped when nothing changed.
- **Asynchronous readbacks:** our own implementation of the native port's idea. When the game reads
  GPU results, the game thread waits for the GPU and the command thread keeps working.
- **Measurement tooling:** per-thread CPU, per-draw timing broken down by stage, readback
  statistics (how many bring new data, how long they wait), and FPS by scene load. Almost every
  decision came from these numbers.

## What we brought in from other projects

- **ShadowOfTheColossus-PC** (AjncSystem's native port): direct draws for CPU-written indirect
  arguments, the fill HLE idea, the async readback idea, and two bug fixes from its open PRs (a
  command pool that kept growing, and compute-queue padding packets read wrong).
- **AjncSystem's SotC shadPS4 custom build:** commit `609b4dd2`, which lets the game run with direct
  memory access (DMA) off without black blocks from the depth of field. With DMA off, the final
  build runs noticeably faster.
- **bloodborne_pc / bbport** (deadinside28): the texture collector and the page-fault window for
  VRAM.
- **Pink-shadPS4 0.4.0** (luizgustavs): we rebased everything onto it.
- illusion's **60 FPS patch** for SotC.

## Final build

It has everything above: draw pipe, fill HLE, copy reuse and binding memos, async readbacks, DMA off
with the depth-of-field fix, plus the bug fixes.

**Results:** heavy areas went from 18–26 fps to around 30 fps or more, and light and medium areas
reach 50–60 fps, with no visual bugs in my testing. The remaining limit is CPU work per draw (SotC
issues 4–5k+ draws per frame in its heaviest views), not the GPU.

## Thanks

- **shadPS4** (shadps4-emu)
- **Pink-shadPS4** (luizgustavs)
- **ShadowOfTheColossus-PC** and the SotC shadPS4 build (AjncSystem)
- **bloodborne_pc** (deadinside28)
- **LNDF's gc2 branch** (now GC part 2 in upstream shadPS4), which we studied for VRAM
- **illusion** for the 60 FPS patch

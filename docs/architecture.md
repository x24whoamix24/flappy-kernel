# Architecture

Flappy Kernel adds co-tenant work to Mirage's persistent megakernel (MPK). Mirage
compiles a model into a task graph and runs it in one long-lived kernel: worker
blocks execute tasks (matmuls, attention, norms), and scheduler warps release tasks
as their dependencies complete. This project appends extra thread blocks to that
launch. They run a game, a trainer and an arcade of up to 7 players, sharing the GPU's SMs,
memory and lifetime with the LLM.

## Processes and memory

```
┌──────────────────────── GPU (one launch) ─────────────────────────┐
│ Mirage workers ── decode Qwen3-8B ──► tokens[] (device memory)    │
│ arcade block ──── reads tokens[], weights; steps 8 games; renders │
│ learner blocks ── train the policy ──► weights (device memory)    │
└──────────────┬──────────────────────────────────────▲─────────────┘
               │ writes frames, counters, tokens       │ reads flaps, slot controls
         ┌─────▼───────────────────────────────────────┴─────┐
         │ HostShared: pinned shared memory (/dev/shm/flappy)│
         └─────▲───────────────────────────────────────▲─────┘
               │ maps it (CUDA host-registered)        │ maps it (mmap)
   Mirage process (deploy/serve.sh)            web server (server/app.py)
   one launch, all questions   ◄── queue/ ───  players' questions
```

- **`HostShared`** (`kernel/core/state.cuh`) is one struct in a POSIX shared-memory
  segment. The Mirage process registers it with `cudaHostRegister(...Mapped)`, so the
  kernel reads and writes it directly over PCIe. The web server maps the same segment
  with `mmap` and never touches CUDA. Field offsets come from `kernel/standalone layout`,
  which `deploy/serve.sh` writes to `server/layout.json` at start. Rebuild `kernel/`
  after changing `kernel/core/` so that binary matches the headers Mirage compiles.
- **Device state** (weights, Adam moments, environments, game slots) lives in CUDA
  allocations owned by the Mirage process, so it would also survive a relaunch.
- **Request queue:** the server writes questions as JSON files to `queue/`. The Mirage
  process submits the oldest one to the running kernel, names it in `current.json`
  while it is decoded, and writes the answer to `done/<n>.json`. With no question
  queued, the LLM is idle and the game and training keep running.

## Thread-block roles

With `MPK_COTENANT_FUSED=1` (what `serve.sh` uses), the co-tenant blocks are appended
to Mirage's worker kernel in its default two-kernel launch (worker kernel plus scheduler
kernel). With only `MPK_COTENANT_BLOCKS` set, everything runs in Mirage's single
`persistent_kernel`: one kernel launch in total, about 4% slower for the LLM because
Mirage's single-kernel mode is slower by itself.

| Blocks | Role | Code |
|---|---|---|
| `LLM_WORKERS` | Mirage workers | Mirage |
| 1 | arcade (`COTENANT_MODE=4`) or single-game renderer | `run_arcade()`, `run_showcase()` |
| `COTENANT_BLOCKS-1` | A2C learners | `run_learner()` |
| scheduler kernel | Mirage schedulers | Mirage |

In the fused launch the roles are not fixed by block index: once every block has
started, each looks up which SM it landed on, and the blocks in the schedulers' GPCs
become the co-tenants (see "Block placement" below).

Every block must be resident at the same time: learners meet at a grid barrier, and
Mirage workers wait on each other. That is why the defaults add up to exactly the
H100's 132 SMs (96 workers, 4 scheduler blocks, 32 co-tenants).

## The trainer

Each learner block owns 256 environments, one per thread. One update:

1. **Rollout:** 16 steps per environment. Each thread runs the policy forward pass
   (6 → 64 → 64 → 2, weights in shared memory) and samples a flap, then steps its game.
   Observations, actions, rewards and done flags go to a global trajectory buffer.
2. **Returns:** bootstrapped discounted returns (γ = 0.99) from the value head.
3. **Backward pass:** each thread recomputes its activations and backpropagates the A2C
   loss (policy gradient with advantage, value MSE, entropy bonus). Gradients are summed
   across the warp with a reduce-scatter butterfly, then added into shared memory.
4. **Synchronous update:** a grid barrier across learner blocks. Each block then sums
   one slice of the parameters over all blocks' partial gradients and applies Adam. A
   second barrier follows before the next rollout.

The last block to reach the barrier also samples the stop flag, so every learner leaves
at the same update. Mirage's workers set that flag when they receive their terminate
task.

**Constraint learned the hard way:** co-tenant code must stay out of local memory. One
runtime-indexed array in the reduction put 750 GB of traffic through L2 in 0.6 s and
slowed Mirage's latency-bound decode by 30%. The butterfly stages are now templated so
every index is a compile-time constant. Check `ptxas -v` stack sizes and Nsight Compute's
`l1tex__t_bytes_pipe_lsu_mem_local_*` before adding co-tenant code. The one remaining use
is each learner thread's 40-byte `Env` (its pipe arrays are indexed by the nearest pipe's
number), touched a few times per game step and served from L1.

## The arcade

The arcade block wakes every 33.3 ms of GPU time (`%globaltimer`). Each tick:

1. Copies the current weights into shared memory.
2. Thread 0 advances every assigned slot. It compares the slot's `start_seq` and
   `flap_seq` counters with the values it last saw; the host only ever increments them,
   so neither side needs a lock. The player's bird and the AI's bird share one pipe course.
   The AI flies greedily on the live weights.
3. Copies newly decoded LLM tokens from Mirage's `tokens` buffer into `HostShared`.
4. Renders each assigned slot into the back half of its double buffer, then publishes
   the buffer index and bumps `frame_seq`. A frame holds the game and its score only;
   the live numbers are on the page.

Slot 0 is the spectator view: the AI playing alone, restarting after each crash. The
server assigns slots 1–7 to connecting players and frees them on disconnect.

## The serving loop

`mirage/megakernel.patch` adds `serve_requests()` to Mirage's Qwen3 demo. It builds the
megakernel in Mirage's `online_pinned` mode and launches it once; the launch lasts until
`<serve_dir>/stop` exists. For each question it:

1. tokenizes the prompt (Qwen3 chat template, thinking disabled);
2. submits it through Mirage's pinned request ring (`OnlinePinnedRuntime`);
3. waits for its completion, reads the answer's tokens and writes `done/<n>.json`.

The arcade block follows the question in token row 0 by its request id and copies
each token to `HostShared` as soon as the scheduler publishes it. While no question is
queued, Mirage's scheduler waits for the next one instead of running empty iterations,
which would slow the co-tenants.

**Block placement.** Which SM runs which block is up to the hardware, and decode is
~5% slower when Mirage's workers share a GPC with its schedulers than when the
co-tenants do. So in the fused launch every block reads its SM id and roles are
assigned after placement: blocks in the schedulers' GPCs become co-tenants, the rest
workers. The SM-to-GPC map comes from a short probe with thread-block clusters at
startup. Before launching, the patch also checks that workers, schedulers and
co-tenants fit on the GPU's SMs at once, since they wait on each other.

## The web server

`server/app.py` (aiohttp) has one WebSocket per browser:

- **Out:** JPEG frames of that player's slot, or slot 0 for spectators, sent when
  `frame_seq` changes. Each slot's frame is encoded once and shared by everyone watching
  it. Four times a second it also sends a JSON state message with live counters, the answer
  being decoded, and recent answers to players. The page fits a phone screen and can be
  added to the home screen as a full-screen app.
- **In:** `join` (race, solo), `flap`, `restart`, `leave`, `prompt`. Questions are
  limited to 280 characters and 20 in the queue; `--question-interval` can also limit how
  often one client address may ask (off by default).

## Design decisions

| Decision | Why | Alternative considered |
|---|---|---|
| Co-tenants as extra blocks, not tasks in Mirage's graph | Mirage pushes every task of an iteration into fixed per-worker queues up front, so independent tasks would wait behind decode tasks | Adding training tasks to the task graph |
| Spatial split (fixed SMs per role) | Batch-1 decode saturates at ~96 SMs for this graph; the rest were idle | Interleaving training into decode gaps (needs a work-stealing queue) |
| Fused into Mirage's worker launch by default | Keeps Mirage's faster two-kernel mode | Single `persistent_kernel` (available; ~4% slower LLM) |
| Non-worker code behind `__noinline__` calls | Inlined next to the worker path, ptxas serialized every wgmma in the kernel (C7510) | — |
| Host communication through pinned shared memory | The kernel never stops; the server needs no CUDA and no GIL contention | Device copies on a side stream |
| One launch for the whole run (`online_pinned`) | Relaunching per request restarted the co-tenants twice a second and redrew block placement each time | One request per launch (`MPK_RELAUNCH=1`, still available) |
| Frames rendered on the GPU and streamed as JPEG | The game really is in the kernel; any browser can play | Sending game state and drawing in the browser |

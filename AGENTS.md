# Working on Flappy Kernel

Guidance for coding agents (and people) changing this repository. Read
[README.md](README.md) for what the project is and
[docs/architecture.md](docs/architecture.md) for how it fits together.

## Map

| Path | What it is | Who depends on it |
|---|---|---|
| `kernel/cotenant.cuh` | Entry point: `cotenant::run`, `signal_stop`, `host_*` | `mirage/megakernel.patch` calls these by name |
| `kernel/core/` | Settings (`config.h`), shared state (`state.cuh`), device helpers | everything in `kernel/` |
| `kernel/training/` | Policy network, hand-written A2C gradient, learner blocks | `reference/check_grad.py` checks the gradient |
| `kernel/game/` | Physics, frame state, renderer, 30 Hz loop, showcase, arcade | the frames players see |
| `kernel/host/` | Host setup, shared memory, CUDA IPC, recording | Mirage's host code, `kernel/tools/` |
| `kernel/tools/` | `standalone` and `separate`: the co-tenants without the LLM | tests, benchmarks, `deploy/serve.sh` (`standalone layout`) |
| `mirage/megakernel.patch` | Changes to Mirage @ `ca58a02` | applied by the Dockerfile and `deploy/install.sh` (which `deploy/setup_pod.sh` runs in a pod) |
| `server/` | Web server (`app.py`) and page (`static/index.html`) | players |
| `reference/` | PyTorch version of the game, network and gradient | the gradient check |
| `bench/`, `results/` | Benchmarks and the raw results behind `docs/results.md` | docs |

## Build

The CUDA tools compile without a GPU; running them needs an H100 (`sm_90a`).

```bash
make -C kernel                       # needs nvcc (CUDA 12.8)
# without a local CUDA toolkit:
docker run --rm -v "$PWD/kernel":/k -w /k nvidia/cuda:12.8.1-devel-ubuntu22.04 make
```

The full arcade (Mirage + LLM + server) builds from the `Dockerfile`; see the
README's Quick start.

## Verify a change

Run the checks that cover what you touched. All of them must pass before a
change is done.

Without a GPU (CI runs all of these):

```bash
make -C kernel                                   # no warnings
kernel/standalone layout                         # runs without a GPU
python -m py_compile server/app.py bench/*.py reference/*.py
for f in deploy/*.sh bench/*.sh; do bash -n "$f"; done; node --check tests/e2e.test.mjs
clang-format --dry-run -Werror kernel/cotenant.cuh kernel/*/*.cu*  kernel/*/*.h
git -C <mirage @ ca58a02> apply --check "$PWD/mirage/megakernel.patch"
```

With an H100:

| You changed | Check |
|---|---|
| `kernel/core/state.cuh` (`HostShared`, `SlotIO`) | `kernel/standalone layout` before and after; `server/app.py` reads the shared block at these offsets and assumes their types (`frame_seq` 64-bit, other slot fields 32-bit, tokens 64-bit), so update it in the same change. `deploy/serve.sh` rebuilds and re-exports the layout on every start |
| `kernel/training/` | `python reference/check_grad.py kernel/standalone <tmpdir>`: it passes (max relative error under 1e-3; about 1e-6 today); `COTENANT_MODE=3 kernel/standalone run 32 20` training updates/s not lower than before (about 395 in this training-only mode on an H100 SXM; `docs/results.md` reports 377 with the game running too) |
| code that runs in a co-tenant block | `nvcc -Xptxas -v`: no spills, stack frame not larger than before (40 bytes); no new `LDL`/`STL` in `cuobjdump -sass` (see below) |
| `kernel/game/` | `kernel/standalone run 32 10` holds 30 frames/s; look at the frames (`COTENANT_OUT=<dir>` records video) |
| `server/`, `mirage/` | serve with `deploy/serve.sh`, then `curl localhost:8080/healthz` returns `{"stalled": false}` and `node tests/e2e.test.mjs http://localhost:8080 --llm` passes every check (needs Chrome: `npx playwright install chrome`) |

## Rules that are easy to break

- **No new local memory in co-tenant code.** Co-tenant blocks share the GPU with
  the LLM. A per-thread array indexed at run time is placed in local memory, and
  its traffic through L2 slowed LLM decoding by about 30% once. Keep array
  indices compile-time constants (fully unrolled loops, template parameters) and
  compare `ptxas -v` and `LDL`/`STL` counts before and after. The one accepted
  exception is each learner thread's `Env` (40 bytes: the pipe arrays are
  indexed by the nearest pipe's number), read and written a few times per game
  step and cached in L1; it is the whole 40-byte stack frame today.
- **The shared-memory layout is an interface.** `HostShared` and `SlotIO` are read
  by `server/app.py` by byte offset. Add fields at the end; never reorder.
- **Entry points are an interface.** Keep `cotenant::run`, `signal_stop`,
  `host_init`, `host_start`, `host_stop` and `host_shared` as they are, or update
  `mirage/megakernel.patch` in the same change.
- **Includes are file-relative** (`"../core/state.cuh"`). Mirage compiles this
  code with its own long include path, where generic names could resolve to
  someone else's header.
- **One Frame per game block.** Only thread 0 changes game state; all threads
  render. Every `__syncthreads()` must be reached by all threads of the block.
- **Keep the reference in sync.** The physics, observation, network and update
  mirror `reference/a2c_flappy.py`; change both or neither.
- **Every block must be resident at once.** Mirage's workers + its schedulers +
  the co-tenant blocks must not exceed the GPU's SM count (132 on an H100 SXM);
  otherwise the learners' grid barrier deadlocks.
- **Who writes what in the shared block.** The host only increments
  `start_seq`/`flap_seq` and sets `assigned`/`mode`; only the kernel writes
  `frame_seq`/`frame_buf`/`phase` and the counters. A frame's number is published
  after it is drawn.
- **Co-tenant roles come from placement, not block index.** In the fused launch,
  `place_block()` in the patch assigns roles once all blocks have started; every
  block must reach the same decision (it is published once, atomically).
- **The idle online scheduler waits, it does not spin.** Spinning on an empty
  request ring slowed training by about 5%.
- **`kLlmMaxTokens` must be at least Mirage's maximum sequence length**
  (`MAX_SEQ`); a `static_assert` checks it when Mirage compiles the kernel.

## Don't

- Run another CUDA process (or MPS) on the GPU while the arcade is serving:
  Mirage's persistent kernel stops producing valid tokens.
- Regenerate `mirage/megakernel.patch` against another Mirage commit without
  re-running the benchmarks in `bench/`.
- Edit `results/` by hand; it holds raw outputs behind `docs/results.md`.
- Push, tag or publish without the maintainer's go-ahead.

## Style

- C++/CUDA: `clang-format -i` with the repository's `.clang-format`; constants
  are `kName`, functions and variables `snake_case`.
- Comments say why, not what; no TODOs or commented-out code.
- Python: standard library first, small functions, no new dependencies without a
  reason in the commit message.

## Never commit

Credentials, kubeconfigs, `.env` files, or anything specific to one cluster or
company (node names, registries, internal hostnames or paths). Deployment
examples in `deploy/` stay generic.

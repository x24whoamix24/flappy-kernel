# Results

All measurements: one H100 80GB HBM3 (SM clock locked at 1290 MHz), Qwen3-8B, batch 1,
greedy decoding, Mirage at `ca58a02` with `mirage/megakernel.patch`. Raw logs, per-frame
counters and token dumps are in `results/`; [results/README.md](../results/README.md)
maps each row below to its directory.

These runs serve one request per launch (`bench/run_mpk.sh`), the way the arcade
served before it moved to a single persistent launch. The arcade's current serving
path (`online_pinned`, `deploy/serve.sh`) measured on the same H100: 130–134 tokens/s
per answer, 386–391 training updates/s and frames every 32.8–34.1 ms, against 126–135
tokens/s and 373 updates/s for the one-request-per-launch arcade.

Two layouts appear (see the end of "Versus running the three jobs separately"):

- **Single-kernel mode** (`MPK_COTENANT_BLOCKS=32`): one `persistent_kernel` launch
  holds the LLM workers, Mirage's schedulers and the co-tenants. The overhead tables,
  the split sweep and the Nsight trace use it.
- **Fused mode** (`MPK_COTENANT_FUSED=1`, what `deploy/serve.sh` serves): one Mirage
  launch of two kernels, the worker kernel with the co-tenant blocks appended and
  Mirage's scheduler kernel. The comparison table has both.

## Megakernel overhead (single-kernel mode)

Short context (prompt 39 tokens, 985 generated):

| Configuration | ms / token | Training | Game |
|---|---|---|---|
| Stock MPK (128 workers, split worker+scheduler kernels) | 7.53 | – | – |
| Single persistent kernel, 128 workers | 7.94 | – | – |
| Single persistent kernel, 96 workers, no co-tenants | 7.86 | – | – |
| **Megakernel: 96 LLM workers + game + 31 learners** | **7.98** | ~360 updates/s | 30 fps |
| Co-tenants alone (32 blocks, no LLM) | – | 377 updates/s | 30 fps |

Long context (same story prompt, 6,038 generated tokens, one 70 s launch):

| Configuration | ms / token |
|---|---|
| Stock MPK, 128 workers | 11.23 |
| LLM alone, 96 workers | 11.22 |
| **Megakernel (LLM + training + game)** | **11.42** (+1.7%) |

In that one launch the trainer ran 27,278 updates (~390/s, ~49M env steps/s); the mean
training episode went from 0 to 498 pipes, and the game rendered 2,105 frames.

Split sweep (1k context; each megakernel against the same LLM config without
co-tenants). The co-tenant overhead stays ~1.5%. What changes is the LLM's own speed
below 96 workers: most per-layer task grids in this Qwen3-8B graph are 64–96 wide,
which is also why stock MPK's extra 32 workers barely help.

| Co-tenant blocks | LLM workers | LLM alone | Megakernel | Env steps / s |
|---|---|---|---|---|
| 32 | 96 | 7.86 | 7.98 | 49M |
| 48 | 80 | 10.53 | 10.67 | 73M |
| 64 | 64 | 10.79 | 10.96 | 96M |

At 112 workers, MPK's LM-head grid produces an invalid TMA descriptor even without
co-tenants.

Single launch: an Nsight Systems trace of a single-kernel-mode megakernel run shows one
`persistent_kernel` (grid 132 × 256 threads, 8.1 s) with no other kernel or copy
overlapping it. The only kernels before it are MPK's `init_kernel` and `prepare_kernel`
setup launches (`results/nsys/gpu_trace_summary.csv` lists every kernel and copy and
how many of its launches overlap `persistent_kernel`: none).

Correctness:

- **LLM:** the megakernel's 6,144 tokens are bit-identical to the same 96-worker LLM
  without co-tenants. Going from 128 to 96 workers changes the floating-point reduction
  order, so that pair first differs at token 563; both outputs are coherent.
- **Trainer:** the hand-written backward pass matches PyTorch autograd to 8.8e-7
  relative error (`reference/check_grad.py`).

## Versus running the three jobs separately

`bench/compare.sh` runs one Qwen3-8B request (1,940 greedy tokens), the A2C trainer and
the 30 Hz game in each setup. `bench/compare.py` scores each run inside the LLM's decode
window, and `bench/compare_video.py` renders `docs/media/comparison.mp4` side by side.

| Setup | LLM ms/token | Training upd/s | Game | Outcome |
|---|---|---|---|---|
| LLM alone, stock Mirage (128 workers) | 8.64 | – | – | reference |
| Separate processes, default time-slicing | failed | 331 | 30 fps | Mirage aborts after a few positions, no valid tokens |
| One process, separate kernels, stock Mirage | 8.68 | 0 | frozen 16.8 s | Mirage holds every SM, co-tenants never start |
| Same, Mirage capped at 96 workers | 8.48 | 0 | frozen 16.4 s | co-tenant kernel still waits for Mirage to finish |
| Separate processes, MPS split tuned by hand | 8.56 | 369 (30 learners) | 30 fps | works once the split is sized in whole TPCs |
| **Co-located in Mirage (`MPK_COTENANT_FUSED=1`)** | **8.58** | **382 (31 learners)** | 30 fps | works with no tuning |
| Single-launch megakernel | 8.97 | 384 | 30 fps | Mirage's single-kernel mode is ~4% slower on its own |

The rows for LLM alone, MPS and co-located are means of two runs (spread ~1.5%).

- **Time-slicing breaks Mirage.** Its persistent kernels abort, writing no valid
  tokens, when any other CUDA process shares the GPU. A plain PyTorch matmul loop is
  enough to trigger it, in both its split and single-kernel modes (`results/compare_r2/dbg_*`).
  Alone, it is correct.
- **Separate kernels starve.** Launched next to Mirage from the same process, the
  co-tenant kernel got 1 update and 1 frame during the whole request, even with 32 SMs
  left free.
- **MPS ties co-location, but needs tuning.** The device then reports only the client's
  share of SMs, and shares are whole TPCs. The first split deadlocked the trainer, and the
  working one had to drop a learner.
- **Co-location matches MPS on throughput** and needs neither a daemon nor any tuning.

Two layouts co-locate inside Mirage:
- `MPK_COTENANT_BLOCKS=32` puts everything in one `persistent_kernel` launch: Mirage's
  single-kernel mode.
- Adding `MPK_COTENANT_FUSED=1` appends the co-tenant blocks to Mirage's worker kernel
  in its default two-kernel launch.

In both, co-tenancy itself costs ~0%: single-kernel Mirage alone runs at 8.96 ms/token,
and with training and the game 8.97. The single-launch layout is slower only because
Mirage's single-kernel mode is. Part of that came from ptxas serializing every wgmma in
`persistent_kernel` (C7510) when the scheduler path was inlined next to the worker path.
The patch moves the non-worker paths behind `__noinline__` calls.

## What made the difference: keep co-tenant traffic off L2

The first megakernel decoded at 10.3 ms/token, 30% slower, even though a streaming
probe showed the learners cost the other SMs only 3% of HBM bandwidth.
Nsight Compute found the cause:

- One reduction indexed a `float r[32]` with a runtime index, so the array was placed in
  local memory.
- That produced 754 GB of local traffic and ~900 GB/s through L2.
- Batch-1 decode is latency-bound on L2: MPK's event atomics and TMA loads go through
  it, so the extra queueing slowed every decode step.

Making the reduction stages compile-time cut L2 traffic 122×. Training got 70% faster
and decode returned to within 2% of the LLM-only baseline.

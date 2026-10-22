# Configuration

## Serving (`deploy/serve.sh`)

| Variable | Default | Effect |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-8B` | Qwen3 checkpoint, from the Hugging Face cache |
| `MAX_SEQ` | `1024` | Mirage's compiled maximum sequence length: prompt plus answer. Longer answers are cut here |
| `LLM_WORKERS` | `96` | Mirage worker blocks (SMs) for the LLM |
| `COTENANT_BLOCKS` | `32` | Co-tenant blocks: 1 arcade + (N−1) learners. `LLM_WORKERS + 4 + COTENANT_BLOCKS` must not exceed the GPU's SM count |
| `PORT` | `8080` | Web server port |
| `FK`, `SERVE_DIR`, `MIRAGE` | `/root/fk`, `/var/lib/flappy-kernel`, `/root/mirage` | Install, queue and Mirage locations |

## Runtime switches (environment of the Mirage process)

| Variable | Effect |
|---|---|
| `MPK_COTENANT_BLOCKS=N` | Compile co-tenant code into Mirage's kernel and append N blocks |
| `MPK_COTENANT_DIR` | Directory with `cotenant.cuh` (added to the kernel's include path) |
| `MPK_COTENANT_FUSED=1` | Append the co-tenants to Mirage's worker kernel (default two-kernel launch) instead of the single persistent kernel |
| `MPK_COTENANT_SEPARATE=1` | Launch the co-tenants as their own kernel next to Mirage (comparison baseline) |
| `MPK_SINGLE_KERNEL=1` | Mirage's single-kernel launch without co-tenants (baseline) |
| `MPK_NUM_WORKERS=W` | Mirage worker count (default: 128 on H100) |
| `MPK_RESERVED_SMS=N` | SMs to leave out when sizing Mirage's schedulers (defaults to `MPK_COTENANT_BLOCKS`) |
| `MPK_RELAUNCH=1` | Reset request state on every launch, to serve one request per launch in offline mode (`deploy/serve.sh` no longer uses it: `MPK_SERVE_DIR` serves every request in one launch) |
| `MPK_SERVE_DIR=dir` | Serve requests from `dir/queue` until `dir/stop` exists |
| `MPK_THINKING=0` | Disable Qwen3 thinking for the demo's single-prompt mode |
| `MPK_DUMP_RUN=file` | Write the run's tokens and timing to `file` (benchmarks) |
| `COTENANT_MODE` | `0` game + training, `1` game only, `2` idle, `3` training only, `4` arcade + training |
| `COTENANT_SHM=/name` | Put `HostShared` in POSIX shared memory (needed by the web server) |
| `COTENANT_OUT=dir` | Record the single-game view to `dir/game_N.mp4` plus per-frame CSVs |
| `COTENANT_MAX_UPDATES=N` | Stop training after N updates (profiling) |
| `COTENANT_QUIET=1` | No per-launch summary line |

## Compile-time constants (`kernel/core/config.h`)

Changing these changes the shared-memory layout. Rebuild `kernel/` and regenerate
`server/layout.json` (`serve.sh` does both).

| Constant | Value | Meaning |
|---|---|---|
| `kSlots` | 8 | Arcade games: slot 0 spectator view, slots 1..7 players |
| `kLlmMaxTokens` | 8192 | Token feed capacity; must be ≥ `MAX_SEQ` |
| `kTickNs` | 1/30 s | Game tick |
| `kRolloutLen` | 16 | Rollout length per update |
| `kEnvsPerLearner` | 256 | Environments per learner block (one per thread) |
| `kObs`, `kHidden` | 6, 64 | Network shape: 6 → 64 → 64 → (flap logit, value) |
| `kLearningRate`, `kAdamBeta1`, `kAdamBeta2` | 1e-3, 0.9, 0.999 | Adam |
| `kDiscount`, `kEntropyCoef`, `kValueCoef` | 0.99, 0.01, 0.5 | A2C loss |
| `kPipeGap`, `kPipeSpeed`, `kPipeSpacing`, `kGravity`, `kFlapSpeed` | 100, 4, 160, 1, −9 | Game difficulty and physics (same as `reference/a2c_flappy.py`) |

The trainer's batch is `(COTENANT_BLOCKS − 1) × 256 × 16` steps per update: 126,976 at
the default 32 blocks.

## Server (`server/app.py`)

| Setting | Default | Where |
|---|---|---|
| Question length | 280 characters | `PROMPT_MAX_CHARS` |
| Seconds between questions per client address | 0 (no limit) | `--question-interval` (server option) |
| Queue length | 20 | `QUEUE_MAX` |
| JPEG quality | 80 | `JPEG_QUALITY` |

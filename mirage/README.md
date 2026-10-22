# Mirage patch

`megakernel.patch` applies to [Mirage](https://github.com/mirage-project/mirage) at
commit `ca58a02`:

```bash
git clone --recursive https://github.com/mirage-project/mirage.git
cd mirage && git checkout ca58a02 && git submodule update --init --recursive
git apply /path/to/flappy-kernel/mirage/megakernel.patch
pip install -e .
```

The header changes are compiled into the megakernel when it is built (JIT), so
editing them needs no rebuild. The `task_register.cc` fix needs `pip install -e .`.

## What it changes

| File | Change |
|---|---|
| `include/mirage/persistent_kernel/persistent_kernel.cuh` | Co-tenant blocks in `persistent_kernel` (single launch) or in `worker_kernel` (`MPK_COTENANT_FUSED`), or as their own kernel (`MPK_COTENANT_SEPARATE`); workers signal co-tenants to stop on `TASK_TERMINATE`; in the fused launch, roles chosen after block placement so the co-tenants, not the workers, share GPCs with the schedulers (SM-to-GPC map from a cluster probe at init); a startup check that all blocks fit on the GPU's SMs; online serving (`MODE_ONLINE_PINNED`) with co-tenants: the launch call returns at once and `wait_persistent_kernel` ends the co-tenants, the co-tenants follow the request in token row 0, the idle scheduler waits for the next request instead of running empty iterations, and host-memory reads stay off the per-token path; non-worker paths behind `__noinline__` calls so ptxas keeps the workers' wgmma pipelines; `MPK_SINGLE_KERNEL`; `MPK_RELAUNCH` re-runs `init_kernel` per launch |
| `python/mirage/mpk/persistent_kernel.py` | Passes `-DMPK_COTENANT_BLOCKS` and the co-tenant include path to the kernel compile |
| `python/mirage/utils.py` | `MPK_NUM_WORKERS` and `MPK_RESERVED_SMS` override worker and scheduler sizing |
| `demo/qwen3/demo.py` | `serve_requests()` (`MPK_SERVE_DIR`): one `online_pinned` launch answers every queued question through `OnlinePinnedRuntime`; `MPK_THINKING`, `MPK_DUMP_RUN` |
| `src/kernel/task_register.cc` | Fixes two Hopper paged-attention codegen bugs: a template with one placeholder fewer than its arguments (aborts code generation), and the split-KV path passing the page stride as `ROTARY_DIM` |

Without any of the new environment variables set, Mirage behaves as before, apart from
the codegen fix, the `__noinline__` change, and two changes to the `online_pinned`
scheduler (waiting while idle, and skipping host-memory reads for rows that cannot be
awaiting release).

The two `task_register.cc` fixes are independent of this project and would be worth
sending upstream.

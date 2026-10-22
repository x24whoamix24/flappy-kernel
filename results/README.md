# Raw results

Logs, per-frame counters and token dumps behind [docs/results.md](../docs/results.md).
Each run directory holds some of: `log.txt` (Mirage demo output; its `per-token latency`
line is the ms/token reported in the docs), `run.json` (tokens and wall time of the whole
launch), `frames_0.csv` (one row per rendered frame: training updates, game score, LLM
position), `train.csv` and `game.log` (separate trainer and game processes).

## Megakernel overhead (Mirage's single-kernel mode)

| docs/results.md row | Directory |
|---|---|
| Single persistent kernel, 128 workers (7.94) | `single128/` |
| Single persistent kernel, 96 workers, no co-tenants (7.86); split sweep 32/96 "LLM alone" | `single96_s16/` |
| Megakernel: 96 LLM workers + game + 31 learners (7.98); split sweep 32/96 "Megakernel" | `mk_full_v2/` |
| The first megakernel, before the L2 fix (10.3) | `mk_full/` |
| Long context: stock MPK, 128 workers (11.23) | `long_split128/` |
| Long context: LLM alone, 96 workers (11.22) | `long_single96/` |
| Long context: megakernel (11.42) | `showcase/` |
| Split sweep 48/80 (10.53 / 10.67) | `sweep_llm_w80/`, `sweep_mk_c48/` |
| Split sweep 64/64 (10.79 / 10.96) | `sweep_llm_w64/`, `sweep_mk_c64/` |
| 112 workers: invalid TMA descriptor without co-tenants | `sweep_llm_w112/` (failed run, kept as evidence) |
| Single launch, nothing overlapping it | `nsys/gpu_trace_summary.csv`: one kernel or copy group per row, with how many of its launches overlap `persistent_kernel` (all 0) |

`docs/media/race.gif` is a race recorded next to a running arcade with
`bench/race_record.py` and laid out by `bench/compose.py --race`.
`video_2026-10-22/` holds a showcase recording (`showcase/`: single-kernel
megakernel, 11.35 ms/token; `bench/compose.py` lays it out) and the runs behind
`docs/media/comparison.mp4` (`cmp_*/` and `compare.json`); its numbers match the
tables above within run-to-run noise.

No raw data here: the short-context stock MPK row (7.53) and "co-tenants alone" (377
updates/s, from `kernel/standalone run`).

## Versus running the jobs separately (Mirage's default two-kernel launch)

`compare_r1/` and `compare_r2/` are two rounds of `bench/compare.sh`; `compare.json` in
each is `bench/compare.py`'s scoring. The table's LLM-alone, MPS and co-located rows are
the mean of both rounds. The other setups ran once: their data is in `compare_r1/`, and
`compare_r2/` keeps only their Mirage logs.

| docs/results.md row | Directories |
|---|---|
| LLM alone, stock Mirage | `compare_r*/cmp_llm_alone/` |
| Separate processes, default time-slicing | `compare_r1/cmp_procs/`, `compare_r2/cmp_procs/` (log), `compare_r2/dbg_*` |
| One process, separate kernels, stock Mirage | `compare_r1/cmp_streams_128/` |
| Same, Mirage capped at 96 workers | `compare_r1/cmp_streams_96/` |
| Separate processes, MPS split | `compare_r*/cmp_mps/` |
| Co-located in Mirage (`MPK_COTENANT_FUSED=1`), what `deploy/serve.sh` runs | `compare_r*/cmp_fused/` |
| Single-launch megakernel | `compare_r1/cmp_mega/` |
| Single-kernel Mirage alone (8.96) | `compare_r1/cmp_single128/` |
| Co-tenants alone | `compare_r1/cmp_cotenants_alone/` |

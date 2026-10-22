# Contributing

Thanks for helping. [AGENTS.md](AGENTS.md) is the working guide for this repository,
for people and coding agents alike: the map, the build, the checks each kind of change
needs, and the rules that are easy to break.

## Before you open a pull request

- **Build:** `make -C kernel` (needs CUDA 12.8; no GPU needed to compile).
- **Checks without a GPU:** what CI runs (`.github/workflows/ci.yml`): Python compiles,
  shell scripts parse, `clang-format --dry-run -Werror` on `kernel/`, the kernel builds,
  and `kernel/standalone layout` runs.
- **Checks on an H100:** the ones in AGENTS.md for the code you changed. For
  `kernel/training/` that is the gradient check; for code in a co-tenant block, the
  register, stack and local-memory comparison; for `kernel/game/`, a look at the frames.
  Say in the pull request which you ran and what they showed.
- **Mirage patch:** every release carries a prebuilt Mirage for its patch (AGENTS.md
  says how); without one, installs build Mirage from source.
- **Benchmarks:** if a change can affect speed, run the relevant `bench/` script before
  and after, and put the numbers in the pull request.

## Style

- C++/CUDA: `clang-format -i` with the repository's `.clang-format` (version 18).
- Comments explain why; no TODOs or commented-out code.
- Commit messages say what changed and why.

## Keep it generic

Do not add anything specific to one cluster or company (node names, registries,
internal hostnames or paths), and never commit credentials. Deployment examples stay
generic.

## License

By contributing you agree that your contributions are licensed under the Apache
License 2.0 ([LICENSE](LICENSE)).

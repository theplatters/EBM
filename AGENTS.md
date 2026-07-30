# Repository Guidelines

## Project Structure & Module Organization

EBM is a mixed-language agent-based modeling project. The Julia package entry point is `src/EBM.jl`; most active code lives under `src/Traffic/`, split into `components/`, `core/`, `systems/`, `simulation/`, and `analysis/`. Keep new Julia files in the matching subsystem and include them from `src/Traffic/Traffic.jl`. Rust prototypes live in `src/*.rs`; Cargo currently builds `src/main.rs`. Exploratory Pluto work belongs in `notebooks/`. Typst sources and bibliography files live in `paper/`, while generated figures and animations belong in `plots/` or alongside their document source.

## Build, Test, and Development Commands

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` installs the dependencies pinned by `Manifest.toml`.
- `julia --project=. -e 'using EBM'` performs a quick package-load smoke test.
- `julia --project=. -t auto` starts a threaded Julia REPL; run `using EBM` before experiments.
- `cargo build` compiles the Rust executable, and `cargo run` runs `src/main.rs`.
- `cargo test` runs Rust unit and integration tests.
- `cargo fmt --check` and `cargo clippy --all-targets` check Rust formatting and common mistakes.
- `typst compile paper/main.typ` rebuilds the paper when Typst is installed.

## Coding Style & Naming Conventions

Use four spaces in Julia and Rust source. Follow Julia conventions: `snake_case` for variables and functions, `PascalCase` for types, and a trailing `!` for mutating functions such as `step!`. Keep simulation state in components/resources and system behavior in `systems/`. Let `cargo fmt` define Rust layout; use `snake_case` for functions/modules and `PascalCase` for structs and enums. Prefer focused files and explicit `include` ordering.

## Testing Guidelines

There is no formal Julia test suite or coverage threshold yet. Add Julia tests under `test/runtests.jl` using `Test`, with deterministic seeds for stochastic models. Place Rust unit tests beside the implementation in `#[cfg(test)]` modules or integration tests in `tests/`. Before submitting, run the Julia smoke test plus `cargo test`; add assertions for regressions rather than relying only on plots.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, lowercase summaries such as `rename for clarity` and `fixed grid rebuilding`. Keep each commit focused and state the behavioral change. Pull requests should describe the model or subsystem affected, list validation commands, and link relevant issues. Include before/after plots or screenshots for visualization, paper, or presentation changes. Do not commit incidental generated media or manifest changes unless they are intentional and explained.

# Repository Guidelines

## Project Structure & Module Organization

EBM is a mixed-language agent-based modeling project. The Julia package entry point is `src/EBM.jl`; traffic-model code lives in `src/Traffic/`, organized into `components/`, `core/`, `systems/`, `simulation/`, and `analysis/`. Add Julia files to the matching subsystem and include them from `src/Traffic/Traffic.jl` in dependency order. Rust prototypes live directly under `src/`, and Cargo currently builds `src/main.rs`. Put exploratory Pluto notebooks in `notebooks/`, Typst sources and bibliographies in `paper/`, and generated figures or animations in `plots/` or beside their document source.

## Build, Test, and Development Commands

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` installs dependencies pinned by `Manifest.toml`.
- `julia --project=. -e 'using EBM'` performs a quick package-load smoke test.
- `julia --project=. -t auto` starts a threaded Julia REPL; run `using EBM` before experiments.
- `cargo build` compiles the Rust executable; `cargo run` runs `src/main.rs`.
- `cargo test` runs Rust unit and integration tests.
- `cargo fmt --check` and `cargo clippy --all-targets` check formatting and common mistakes.
- `typst compile paper/main.typ` rebuilds the paper when Typst is installed.

Run commands from the repository root, two levels above this directory.

## Coding Style & Naming Conventions

Use four-space indentation in Julia and Rust. Follow Julia conventions: `snake_case` for variables and functions, `PascalCase` for types, and a trailing `!` for mutating functions such as `step!`. Keep simulation state in components or resources and behavior in `systems/`. Let `cargo fmt` define Rust layout; use `snake_case` for functions and modules and `PascalCase` for structs and enums. Prefer focused files and explicit `include` ordering.

## Testing Guidelines

There is no formal Julia test suite or coverage threshold yet. Add deterministic Julia regression tests under `test/runtests.jl` using `Test`, including fixed seeds for stochastic behavior. Place Rust unit tests beside implementations in `#[cfg(test)]` modules or integration tests in `tests/`. Before submitting, run the Julia smoke test and `cargo test`; verify behavior with assertions rather than plots alone.

## Commit & Pull Request Guidelines

Use short, imperative commit summaries consistent with project history, such as `update plotting` or `rename for clarity`. Keep commits focused on one behavioral change. Pull requests should identify the affected model or subsystem, summarize the change, list validation commands, and link relevant issues. Include before-and-after plots or screenshots for visualization, paper, or presentation changes. Commit generated media or manifest updates only when intentional, and explain them in the PR.

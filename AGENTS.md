# Repository Guidelines

## Project Structure

EBM is a mixed-language agent-based modeling repository. `src/EBM.jl` is the
Julia package entry point and exports four Ark.jl-based models:

- `EBM.Traffic` lives in `src/Traffic/` and contains the traffic convention,
  capability, and sequential-reference models.
- `EBM.AssetMarket` lives in `AssetMarket/` and implements the Santa Fe
  artificial stock market.
- `EBM.ElFasol` lives in `ElFasol/` and implements the El Farol bar problem.
- `EBM.Sugarscape` lives in `Sugarscape/` and implements the single-resource
  Sugarscape model with optional reproduction and disease.

Each active Julia model separates `components/`, `core/`, `systems/`,
`simulation/`, and `analysis/` where applicable. Add a file to the narrowest
matching subsystem and include it from that model's module entry point in
dependency order. Shared package tests are in `test/runtests.jl`; the three
top-level model directories also contain focused test files that the root suite
includes.

Traffic experiment drivers and reports live in `notebooks/`. Despite the name,
these are primarily reproducible Julia scripts and Markdown reports, not only
Pluto notebooks. Traffic outputs belong in `plots/`; each other model keeps its
generated plots beside its implementation. Typst paper, abstract, presentation,
bibliography, and presentation assets live in `paper/`.

Rust files under `src/` are prototypes. Cargo currently builds only
`src/main.rs`; `src/sciencemodel.rs` is not wired into that binary. The loose
Julia prototypes `src/sir.jl`, `src/sfcio.jl`, and `src/repl.jl` are not exported
by `EBM`.

## Build, Test, and Development Commands

Run commands from the repository root.

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` installs dependencies
  from the checked-in project and manifest.
- `julia --project=. -e 'using EBM'` is the Julia package-load smoke test.
- `julia --project=. test/runtests.jl` runs the complete Julia regression suite,
  including Traffic, ElFasol, AssetMarket, and Sugarscape.
- `julia --project=. -t auto` starts a threaded REPL for experiments.
- `cargo test`, `cargo fmt --check`, and `cargo clippy --all-targets` validate the
  current Rust target.
- `typst compile paper/main.typ`, `typst compile paper/abstract.typ`, and
  `typst compile paper/presentation/presentation.typ` rebuild the documents.

Long-running Traffic studies default to 5,000 ticks and often 30 paired seeds.
Use their documented `TRAFFIC_*` environment overrides for a small development
run, then rerun the published configuration before updating results. Standard
plot generators are:

- `julia --project=. -t auto notebooks/run_social_habit_experiment.jl`
- `julia --project=. -t auto notebooks/run_mixture_ensemble_dynamics.jl`
- `julia --project=. -t auto notebooks/run_speed_sensitivity_experiment.jl`
- `julia --project=. AssetMarket/generate_plots.jl`
- `julia --project=. ElFasol/generate_plots.jl`
- `julia --project=. Sugarscape/generate_plots.jl`

## Coding and Model Conventions

Use four spaces in Julia and Rust. Follow Julia naming conventions:
`snake_case` for variables and functions, `PascalCase` for types, and `!` for
mutating functions. Let `cargo fmt` define Rust layout.

Keep persistent simulation state in components or resources and behavior in
systems. `ModelParams` contains structural/economic parameters; `ModelArgs`
contains run configuration such as seed and number of steps. Preserve explicit
system ordering in `simulation/step.jl`. Do not add or remove ECS components
while iterating a query; stage structural changes and commit them between
systems.

All stochastic behavior must use the seeded model RNG. Tests and experiments
must use deterministic seeds, and paired comparisons must reuse the same seeds
across treatments. Treat logger schemas and generated CSV column names as public
analysis interfaces.

Traffic-specific semantic and experiment constraints are documented in
`src/Traffic/AGENTS.md` and apply to every file under that directory.

## Testing and Generated Artifacts

Add behavioral regressions to `test/runtests.jl` or the relevant model-local
test file. Prefer assertions on state transitions, accounting identities,
reproducibility, and treatment differences; plots alone are not tests. For a
model change, run the smallest focused test during development and the complete
root suite before handoff. Also run `git diff --check`.

Generated figures, animations, CSV files, PDFs, and manifest changes should be
committed only when intentional. When a Traffic experiment changes its horizon,
lookahead, treatment semantics, or output schema, regenerate every affected
artifact and update `plots/README.md`. Do not relabel an old artifact as current.
Preserve unrelated generated files and dirty-worktree changes.

## Commit and Pull Request Guidelines

Use short, imperative, lowercase commit summaries consistent with project
history, such as `update plotting` or `rename for clarity`. Keep commits focused
on one behavioral or documentation change. Pull requests should identify the
affected model, explain any changed assumptions, list validation commands, and
link relevant issues. Include before/after figures for visualization or paper
changes and state whether generated artifacts or manifests changed deliberately.

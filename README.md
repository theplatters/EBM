# EBM

EBM is a research repository for agent-based economic and social models built
around Entity Component System architecture. The Julia package uses Ark.jl and
exports four active models; the repository also contains a small Rust/Bevy ECS
prototype area and Typst research documents.

## Active Julia models

| Module | Location | Subject |
|---|---|---|
| `EBM.Traffic` | `src/Traffic/` | Lane convention, habit formation, bounded observation, synchronous multi-speed capabilities, and a sequential reference model |
| `EBM.AssetMarket` | `AssetMarket/` | Santa Fe artificial stock market with heterogeneous evolving forecast rules |
| `EBM.ElFasol` | `ElFasol/` | El Farol attendance coordination with competing predictor families |
| `EBM.Sugarscape` | `Sugarscape/` | Wealth distribution, movement, lifecycle, optional reproduction, and disease |

`src/EBM.jl` is the package entry point and exports these four modules. Each
model has its own components, resources, systems, simulation schedule, logger,
plotting functions, and deterministic tests.

## Setup and validation

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using EBM'
julia --project=. test/runtests.jl
```

The root test file runs Traffic regressions and includes the focused ElFasol,
AssetMarket, and Sugarscape suites. Stochastic tests and published experiments
use explicit seeds.

The Rust target is currently separate from the Julia package:

```sh
cargo test
cargo fmt --check
cargo clippy --all-targets
```

Cargo presently builds the minimal `src/main.rs` executable. The larger
`src/sciencemodel.rs` Bevy ECS implementation is a prototype and is not wired
into that executable.

## Repository map

```text
src/EBM.jl             Julia package entry point
src/Traffic/           active Traffic module
AssetMarket/           active artificial stock-market module
ElFasol/               active El Farol module
Sugarscape/            active wealth-distribution module
test/runtests.jl       complete Julia regression suite
notebooks/             Traffic experiment scripts and result reports
plots/                 Traffic figures, animations, CSV data, and artifact audit
paper/                 Typst paper, abstract, presentation, bibliography, assets
src/*.rs               Rust prototypes; Cargo entry point is src/main.rs
src/sir.jl             unexported exploratory SIR prototype
src/sfcio.jl           unexported exploratory stock-flow prototype
```

The top-level model directories keep their own generated media and specialized
documentation. See [AssetMarket](AssetMarket/README.md),
[ElFasol](ElFasol/README.md), and [Sugarscape](Sugarscape/README.md).

## Traffic models

Traffic is the most extensively studied subsystem. It contains three related
execution paths:

- synchronous unit-speed traffic with eight occupancy-forecast strategies,
  including heterogeneous per-car strategy profiles;
- synchronous multi-speed `CapabilityModel` traffic with bounded local
  information and collision replacement;
- an Agents.jl sequential reference model used for matched contrasts.

### Capability semantics

Habit, Convention, and SocialHabit are separate capabilities that influence the
same lane-response value, LR:

- **Habit** is the Hodgson–Knudsen disposition reinforced by the time a driver
  has spent on its own realized side.
- **Convention** is learned from a history of locally observed side choices made
  by other drivers.
- **SocialHabit** is learned from a history of locally observed, geometrically
  vanishing traces deposited by successful drivers.

Capability shares are independent, so a driver may carry any mixture. Static
replacement redraws capabilities from entry shares. Evolutionary replacement
inherits stable capabilities and quantitative traits from a surviving driver
with mutation, while acquired dispositions reset for the newborn.

```julia
using EBM

model = EBM.Traffic.CapabilityModel(
    habit_share = 0.5,
    convention_share = 0.5,
    social_habit_share = 0.5,
    replacement_policy = EBM.Traffic.EvolutionaryReplacement(
        capability_mutation_rate = 0.02,
        trait_mutation_scale = 0.05,
    ),
)

args = EBM.Traffic.ModelArgs(
    seed = 42,
    params = EBM.Traffic.ModelParams(lookahead = 20),
    prediction_strategy = model,
    steps = 5_000,
)
logger = EBM.Traffic.main(args)
```

The low-level API defaults remain suitable for short interactive runs
(`ModelArgs.steps == 300`, `ModelParams.lookahead == 60`). Current capability
experiments explicitly use 5,000 ticks, a 1,000-tick burn-in, and lookahead 20.

### Traffic snapshots and visualization

Traffic can capture immutable snapshots, plot a state, plot analytical history,
or record an animation:

```julia
using EBM, CairoMakie

args = EBM.Traffic.ModelArgs(
    seed = 42,
    prediction_strategy = EBM.Traffic.DecisionAwareStrategy(),
    steps = 100,
)
history = EBM.Traffic.traffic_history(args; every = 2)

save("plots/traffic_overview.png", EBM.Traffic.plot_traffic(last(history)))
save("plots/traffic_history.png", EBM.Traffic.plot_traffic_history(history))
EBM.Traffic.record_traffic(history, "plots/traffic.mp4"; framerate = 12)
```

Capability histories add speed, convention, acquired disposition, replacement
pressure, and capability-composition diagnostics. Heterogeneous occupancy
strategies can be displayed with `color_by = :strategy`.

### Traffic experiments and reports

The current studies are reproducible Julia scripts under `notebooks/`:

| Study | Generator | Report |
|---|---|---|
| Habit, Convention, SocialHabit, mixtures, replacement, and sequential contrast | `run_social_habit_experiment.jl` | [social_habit_experiments.md](notebooks/social_habit_experiments.md) |
| Aggregated mixture dynamics with mean ± SD | `run_mixture_ensemble_dynamics.jl` | [social_habit_experiments.md](notebooks/social_habit_experiments.md) |
| Speed-choice sensitivity | `run_speed_sensitivity_experiment.jl` | [speed_sensitivity_results.md](notebooks/speed_sensitivity_results.md) |
| Capability mechanism diagnostics | `generate_capability_scenario.jl` | [capability_model_results.md](notebooks/capability_model_results.md) |
| No-convention comparison | `generate_no_convention_experiments.jl` | [no_convention_results.md](notebooks/no_convention_results.md) |
| Heterogeneous occupancy strategies | `generate_heterogeneous_strategy.jl` | [heterogeneous_strategy_report.md](notebooks/heterogeneous_strategy_report.md) |
| Eight-strategy benchmark | `strategy_analysis.jl` | [strategy_analysis.md](notebooks/strategy_analysis.md) |

For example:

```sh
julia --project=. -t auto notebooks/run_social_habit_experiment.jl
julia --project=. -t auto notebooks/run_mixture_ensemble_dynamics.jl
julia --project=. -t auto notebooks/run_speed_sensitivity_experiment.jl
```

These studies can be expensive: the principal comparisons use 30 paired seeds
and 5,000 ticks. Most scripts accept `TRAFFIC_REPLICATES`, `TRAFFIC_STEPS`,
`TRAFFIC_BURN_IN`, `TRAFFIC_LOOKAHEAD`, and `TRAFFIC_OUTPUT_DIR` overrides.
The generated-artifact inventory and relic policy are recorded in
[plots/README.md](plots/README.md).

## Other model entry points

AssetMarket, ElFasol, and Sugarscape share the `ModelParams`, `ModelArgs`,
`setup_world`, `step!`, `run_model`, and `main` interface. `run_model` returns
the final Ark world; `main` returns the aggregate logger. Traffic exposes
`setup_world`, `step!`, and `main`, with `traffic_history` providing the
snapshot-oriented runner used by its plotting tools. Each model also exposes a
plotting suite:

```julia
using EBM

bar = EBM.ElFasol.main(EBM.ElFasol.ModelArgs(seed = 2026, steps = 500))
market = EBM.AssetMarket.main(EBM.AssetMarket.ModelArgs(seed = 2026, steps = 2_500))
sugar = EBM.Sugarscape.main(EBM.Sugarscape.ModelArgs(seed = 2026, steps = 250))
```

Regenerate their checked-in figures with:

```sh
julia --project=. ElFasol/generate_plots.jl
julia --project=. AssetMarket/generate_plots.jl
julia --project=. Sugarscape/generate_plots.jl
julia --project=. AssetMarket/run_scenarios.jl
```

## Paper

The manuscript, abstract, and presentation are maintained as Typst sources:

```sh
typst compile paper/main.typ
typst compile paper/abstract.typ
typst compile paper/presentation/presentation.typ
```

Generated PDFs are stored beside their corresponding Typst source.

# Sugarscape wealth-distribution model in Ark.jl

This directory contains an Entity Component System implementation of the canonical
single-resource Sugarscape wealth-distribution model. Citizens move over a toroidal
two-hill resource landscape, harvest sugar, pay heterogeneous metabolic costs, age,
die from starvation or old age, and are optionally replaced by newly initialized
citizens. Optional sexual reproduction and disease transmission add nominal sex types
and dynamic structural heterogeneity. The resulting unequal wealth distribution can be
summarized by its Gini coefficient.

The implementation follows the repository's model conventions: `Sugarscape.jl` is the
module entry point, state is split into focused components and resources, model processes
are systems, `ModelParams` and `ModelArgs` define configuration, and `main` returns a
time-series logger. `run_model` returns the final Ark world for component-level analysis.

## Module structure

`Sugarscape.jl` is included by `src/EBM.jl` and exposed as
`EBM.Sugarscape`:

```text
components/       citizen identity, traits, wealth, sex tags, and infection state
core/             model parameters, landscape, occupancy, RNG, clock, and logger resources
systems/          growback, movement, disease, lifecycle, and reproduction
simulation/       world setup and ordered period schedule
analysis/         snapshots, inequality diagnostics, runner, and plots
agent_oriented/   outcome-equivalent sequential and synchronous Agents.jl models
test/runtests.jl  landscape, movement, lifecycle, disease, reproduction, and plot tests
```

## Running the model

```julia
using EBM

args = EBM.Sugarscape.ModelArgs(seed = 2026, steps = 250)
logger = EBM.Sugarscape.main(args)
logger.gini
```

The default uses seeded shuffled-sequential movement: earlier citizens move and harvest
before later citizens choose. A staged synchronous alternative makes observation,
proposal, conflict resolution, and commitment explicit:

```julia
params = EBM.Sugarscape.ModelParams(
    movement_mode = EBM.Sugarscape.SynchronousMovement,
)
world = EBM.Sugarscape.run_model(
    EBM.Sugarscape.ModelArgs(seed = 2026, params = params, steps = 250),
)
```

The semantics-matched Agents.jl implementations use the same `ModelParams` and
`ModelArgs`:

```julia
sequential = EBM.Sugarscape.AgentSequential.run_model(args)
synchronous = EBM.Sugarscape.AgentSynchronous.run_model(
    EBM.Sugarscape.ModelArgs(
        seed = 2026,
        params = EBM.Sugarscape.ModelParams(
            movement_mode = EBM.Sugarscape.SynchronousMovement,
        ),
        steps = 250,
    ),
)
```

See [`agent_oriented/COMPARISON.md`](agent_oriented/COMPARISON.md) for measured
performance, source-line counts, architecture trade-offs, and reproduction commands.

When several citizens propose the same initially empty cell, one is selected using the
simulation RNG and the others remain at their original positions. Synchronous movement
does not allow a citizen to target a cell that was occupied at the beginning of the
period, even if its occupant intends to leave. This rule preserves single occupancy and
keeps conflict semantics explicit.

The wealth-distribution baseline uses replacement and leaves reproduction and initial
infection disabled. Enable the heterogeneous extension with:

```julia
params = EBM.Sugarscape.ModelParams(
    replace_dead = false,
    reproduction_enabled = true,
    initial_infection_probability = 0.05,
)
```

Female and male tag components define the two nominal reproductive types. Infection is
represented by an optional `Infection` component: transmission adds it, recovery removes
it, and the recovered strain is stored in `ImmuneProfile`. Reproduction and replacement
are mutually exclusive population-regeneration regimes.

The bounded extension makes its operational choices explicit. A fertile female and an
adjacent fertile male may produce at most one child each per period when an adjacent cell
is empty. Each parent contributes half its initial endowment, and the child independently
inherits vision, metabolism, and maximum age from either parent. The disease subsystem
supports one active `UInt64` strain per citizen, transmission across cardinal neighbors,
an additive sugar cost, fixed-duration recovery, and exact-strain immunity. It is a
structural-heterogeneity experiment, not yet a reproduction of Sugarscape's full
multi-disease bit-substring immune adaptation.

## ECS mapping

Citizens are entities with these core components:

- `CitizenId`, `Position`, and `ProposedPosition`;
- parametric traits `Vision`, `Metabolism`, `MaximumAge`, and `InitialEndowment`;
- changing state `Sugar` and `Age`;
- one nominal `Female` or `Male` tag;
- persistent `ImmuneProfile` state and an optional, dynamically changing `Infection`.

The landscape, occupancy index, seeded RNG, simulation clock, per-step event counters,
and logger are resources. One period is decomposed into:

```text
grow back patch sugar
    -> rebuild occupancy
    -> move and harvest, or propose -> resolve -> commit
    -> transmit and progress infections
    -> metabolize and age
    -> remove dead citizens
    -> create replacements
    -> create offspring when reproduction is enabled
    -> log aggregate outcomes
```

`initial_capacity` in `ModelArgs` accepts a nonnegative `width × height` matrix for
replication or controlled experiments. Without it, setup generates a deterministic
two-hill landscape scaled to the requested grid.

## Plotting

```julia
result = EBM.Sugarscape.generate_plot_suite(
    EBM.Sugarscape.ModelArgs(seed = 2026, steps = 250);
    output_dir = "Sugarscape/plots",
)
result.paths
```

This writes a final landscape/wealth visualization and a diagnostic figure containing
wealth, inequality, resource stocks, movement, conflicts, births, infections, and deaths. Run
`julia --project=. Sugarscape/generate_plots.jl` to generate both figures.

For a live view in Pluto, Jupyter, or another browser-backed Julia display, activate
WGLMakie and create an interactive dashboard:

```julia
using EBM
using WGLMakie

WGLMakie.activate!()
visualization = EBM.Sugarscape.interactive_sugarscape(
    EBM.Sugarscape.ModelArgs(seed = 2026, steps = 250),
)
visualization.figure
```

The dashboard has reset, single-step, run/pause, and playback-speed controls. It updates
the resource landscape, citizen positions and wealth, infection markers, inequality
history, the current age–wealth distribution, and event counts. `ModelArgs.steps` is the
playback horizon. A dashboard can also be driven from Julia with
`EBM.Sugarscape.step!(visualization, 10)`, played with
`EBM.Sugarscape.play!(visualization)`, reset with
`EBM.Sugarscape.reset!(visualization)`, and stopped with
`EBM.Sugarscape.stop!(visualization)`.

## Testing

From the repository root:

```sh
julia --project=. Sugarscape/test/runtests.jl
```

The complete package suite, including this model, is
`julia --project=. test/runtests.jl`.

## Extension boundary

The core component signature is intentionally small. Reproduction and infection are
already focused subsystems. Culture, multi-resource trade, and credit should follow the
same pattern instead of being folded into one citizen component. Structural changes must
be committed between systems so queries are never invalidated during iteration.

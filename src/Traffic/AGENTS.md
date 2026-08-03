# Traffic Model Guidelines

This file extends the repository-level `AGENTS.md` for all work under
`src/Traffic/`.

## Structure and Execution Paths

`Traffic.jl` includes files in dependency order:

- `components/` defines cars, spatial state, occupancy strategies, capability
  components, proposals, and replacement policies.
- `core/` defines model parameters, resources, world construction helpers, and
  utilities.
- `systems/` contains observation, LR calculation, habit learning, social-trace
  learning, movement, collision, spawning, and replacement behavior.
- `simulation/` defines setup and the ordered tick schedules.
- `analysis/` contains loggers, snapshots, plotting, sweeps, regressions, and
  experiment runners.
- `sequential_model/` is the Agents.jl reference implementation used to contrast
  the synchronous ECS model.

The module supports three related but distinct paths:

1. Occupancy-strategy traffic is synchronous and normally unit-speed. Strategies
   include per-entity habitus, mean habitus, naive, two-frame naive, unsure,
   random, switch, decision-aware, and heterogeneous per-car compositions.
2. `CapabilityModel` is synchronous and multi-speed. Cars make bounded local
   observations, calculate one LR value, submit lane and speed paths, resolve
   conflicts, learn from committed outcomes, and replace collided cars.
3. `SequentialModel` updates Agents.jl cars in activation order or with its
   explicit simultaneous treatment. It is a reference model, not another
   `CapabilityModel` policy.

Do not silently share state or decision information between these paths merely
to make their outcomes agree.

## Habit, Convention, and SocialHabit

These capabilities have different information sources and must remain separate:

- **Habit** is the Hodgson–Knudsen disposition reinforced by how long the driver
  has occupied its own realized side. It uses the driver's lane history.
- **Convention** is a private history of locally observed lane-side choices made
  by other drivers.
- **SocialHabit** is a private history built by observing spatial traces left by
  successful drivers. Traces decay geometrically on the road and are learned
  only through bounded local observation.

All three capabilities serve the same downstream function: their weighted
dispositions contribute additively to the car's LR value. Do not redefine one
in terms of another, use global lane shares for local convention learning, or
let SocialHabit inspect driver success directly instead of the vanishing trace.
Acquired habitus, perceived convention, and social habit state reset for newborn
cars; only stable capability presence and quantitative traits are eligible for
evolutionary inheritance.

## Capability Tick and Replacement Semantics

Maintain the ordered capability tick:

```text
store previous positions
    -> rebuild occupancy and bounded observations
    -> calculate capability LR and lane proposals
    -> choose speed/path from current observable motion
    -> resolve synchronous micro-step conflicts
    -> decay and deposit successful-driver traces
    -> replace collided cars
    -> update acquired state and aggregate logging
```

`EntryDrawReplacement` samples new capability combinations from configured
shares. `EvolutionaryReplacement` selects a surviving parent, inherits stable
capabilities and traits with mutation, and preserves the replaced car's travel
direction. Replacement pressure is a model outcome and must not be smoothed or
clipped inside the simulation.

The historical speed treatment is speed-first: choose the fastest exactly safe
path and use the LR-selected lane to break equal-speed choices. The optional
`prefer_lane_over_speed=true` treatment exhausts slower speeds on the selected
lane before trying the other lane; `speed_clearance` adds experimental graded
headway. Keep the treatment explicit in reports because it materially changes
coordination and throughput.

## Current Experiment Baseline

Current capability studies use 5,000 ticks, a 1,000-tick burn-in, lookahead 20,
120 cars, and a 2 × 300 periodic road unless a report explicitly states another
fixed design. Most inferential comparisons use 30 paired seeds and report run
points plus mean uncertainty or ensemble mean ± standard deviation. The base
`ModelParams` lookahead remains 60 and `ModelArgs` remains 300 ticks for quick
interactive use; do not confuse those API defaults with the published study
configuration.

Traffic experiment scripts and reports live in the repository-level
`notebooks/` directory. The main mappings are:

- `run_social_habit_experiment.jl` / `social_habit_experiments.md`: pure
  capability scenarios, mixtures, replacement regimes, and sequential contrast.
- `run_mixture_ensemble_dynamics.jl`: aggregated mixture trajectories with
  mean ± SD.
- `run_activation_habit_experiment.jl`: activation-order Habit treatment.
- `run_speed_sensitivity_experiment.jl` / `speed_sensitivity_results.md`:
  speed-first, lane-first, clearance, and cap-2 treatments.
- `generate_capability_scenario.jl`,
  `generate_no_convention_experiments.jl`, and
  `generate_heterogeneous_strategy.jl`: focused diagnostics.

Generated Traffic media and data belong in `plots/`. Keep `plots/README.md`
synchronized with generator, horizon, replication count, uncertainty display,
and relic status. Analytical histories are preferred over torus snapshots when
the question concerns dynamics.

## Testing

Traffic regressions are in the first part of `test/runtests.jl`. Add fixed-seed
tests for capability semantics, bounded information, tick ordering, collision
resolution, replacement inheritance, newborn resets, speed action ordering,
heterogeneous composition, and sequential matching as appropriate. For changes
under this directory, run:

```sh
julia --project=. test/runtests.jl
```

Run commands from the repository root. A regenerated plot is supporting
evidence, not a substitute for state-level assertions.

# EBM

EBM is a work-in-progress package for agent-based modeling with an Entity Component
System architecture. It contains several scientific and exploratory example models.

## SIR model

THe SIR model is a simple model modeling the spread of infectious diseases. The internal logic is directly ported from
<https://juliadynamics.github.io/Agents.jl/stable/examples/sir/>, the code can be found in <https://github.com/theplatters/EBM/blob/main/src/sir.jl>

## Science Model

Written in Rust + Bevy
<https://github.com/theplatters/EBM/blob/main/src/sciencemodel.rs>

## Sugarscape

`EBM.Sugarscape` implements the single-resource Sugarscape wealth-distribution model in
Ark.jl. It supports both shuffled-sequential movement and staged synchronous movement
with explicit destination-conflict resolution. See
[the Sugarscape documentation](Sugarscape/README.md) for its ECS mapping, parameters,
diagnostics, and plotting workflow.

## Traffic visualization

The Traffic module can capture immutable simulation snapshots and render either
a static state overview or an animation:

```julia
using EBM, CairoMakie

args = Traffic.ModelArgs(
    seed = 42,
    steps = 100,
    prediction_strategy = Traffic.DecisionAwareStrategy(),
)
history = Traffic.traffic_history(args; every = 2)

save("plots/traffic_overview.png", Traffic.plot_traffic(last(history)))
Traffic.record_traffic(history, "plots/traffic.mp4"; framerate = 12)
```

Cars are drawn individually on the two-lane ring. Color distinguishes travel
direction, while the side panels summarize lane use, direction, age, habitus,
and left-lane intent.

For `CapabilityModel`, the same interface automatically switches to speed
coloring and capability-specific diagnostics. A second figure summarizes the
speed distribution, coordination, replacement pressure, and changing
capability composition through time. The traffic-response mechanisms default
to independent 75% entry shares, while habit and convention perception default
to 50%, so none of the five composition lines is a universal marker:

```julia
args = Traffic.ModelArgs(
    seed = 42,
    prediction_strategy = Traffic.CapabilityModel(),
    steps = 100,
)
history = Traffic.traffic_history(args; every = 5)

save("plots/capability_state.png", Traffic.plot_traffic(last(history)))
save("plots/capability_history.png", Traffic.plot_traffic_history(history))
Traffic.record_traffic(history, "plots/capability_model.mp4"; framerate = 12)
```

See [the capability-model results](notebooks/capability_model_results.md) for
the bounded-information design, measured scenario, and generated figures.
The matched [experiments without convention perception](notebooks/no_convention_results.md)
isolate habit under both entry-draw and evolutionary replacement.

Capability replacement defaults to independent entry draws. To make capability
transmission evolutionary, select a surviving parent and configure presence
and quantitative-trait mutation:

```julia
evolutionary = Traffic.CapabilityModel(
    replacement_policy = Traffic.EvolutionaryReplacement(
        capability_mutation_rate = 0.02,
        trait_mutation_scale = 0.05,
    ),
)
args = Traffic.ModelArgs(seed = 42, prediction_strategy = evolutionary)
```

The default simulation length is 300 ticks. Evolution preserves the crashed
car's travel direction, inherits only stable capabilities from the selected
survivor, and resets newborn habitus and convention confidence.

See [the strategy analysis](notebooks/strategy_analysis.md) for a paired-seed
comparison of survival, replacements, lane switching, density robustness, and
behavioral-weight sensitivity across all eight occupancy strategies. For a
current-information-only improvement over Naive, use
`Traffic.TwoFrameNaiveStrategy()`: it accepts a lane change only when the
current and one-step-advanced decision frames agree. Recreate
the individual-car examples with:

```bash
julia --project=. notebooks/generate_traffic_examples.jl
```

Cars can also use different strategies in the same world. Their forecasts are
composed into one shared predicted-occupancy field, and collision replacements
inherit strategy and direction so the population mix does not drift:

```julia
mix = Traffic.HeterogeneousStrategy(
    Traffic.DecisionAwareStrategy() => 0.50,
    Traffic.TwoFrameNaiveStrategy() => 0.25,
    Traffic.NaiveStrategy() => 0.25,
)
args = Traffic.ModelArgs(seed = 42, prediction_strategy = mix, steps = 150)
history = Traffic.traffic_history(args; every = 5)
save("plots/heterogeneous.png", Traffic.plot_traffic(last(history); color_by = :strategy))
```

See [the heterogeneous-strategy report](notebooks/heterogeneous_strategy_report.md)
for a 50-replicate comparison and strategy-colored scenario visualization.

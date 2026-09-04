using EBM
using CairoMakie
using Statistics

const T = EBM.Traffic
const SEEDS = 20260730:20260734
const STEPS = 5_000
const BURN_IN = 1_000

# Historical diagnostic; current-semantics outputs must not overwrite retained evidence.
const OUTPUT_DIR = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots", "historical_regenerated"))

function simulate(model, seed; capture_every = nothing)
    args = T.ModelArgs(
        seed = seed,
        params = T.ModelParams(lookahead = 20),
        prediction_strategy = model,
        steps = STEPS,
    )
    world = T.setup_world(args)
    history = isnothing(capture_every) ? T.TrafficSnapshot[] :
              T.TrafficSnapshot[T.traffic_snapshot(world; step = 0)]
    for step in 1:STEPS
        T.step!(world, model)
        if !isnothing(capture_every) && (step % capture_every == 0 || step == STEPS)
            push!(history, T.traffic_snapshot(world; step = step))
        end
    end
    return world, history
end

function metrics(world)
    snapshot = T.traffic_snapshot(world; step = STEPS)
    logger = T.Ark.get_resource(world, T.Logger)
    cars = snapshot.cars
    capability_shares = T._capability_counts(snapshot) ./ length(cars)
    return (
        mean_speed = mean(car.speed for car in cars),
        speed_three_share = mean(car.speed == 3 for car in cars),
        replacements = sum(logger.deaths),
        replacement_rate = sum(logger.deaths[(BURN_IN + 1):STEPS]) /
                           ((STEPS - BURN_IN) * length(cars)),
        early_replacement_rate = sum(logger.deaths[1:BURN_IN]) /
                                 (BURN_IN * length(cars)),
        late_replacement_rate = sum(logger.deaths[(STEPS - BURN_IN + 1):STEPS]) /
                                (BURN_IN * length(cars)),
        realized_coordination = T._convention_strength(snapshot),
        mean_abs_habitus = T._mean_abs_habitus(snapshot),
        habit_share = capability_shares[4],
        convention_share = capability_shares[5],
    )
end

function experiment(model)
    observations = [metrics(first(simulate(model, seed))) for seed in SEEDS]
    names = propertynames(first(observations))
    return NamedTuple{names}(
        Tuple(mean(getproperty(observation, name) for observation in observations)
              for name in names),
    )
end

entry_habit = T.CapabilityModel(
    habit_share = 0.5,
    convention_share = 0.0,
)
entry_control = T.CapabilityModel(
    habit_share = 0.0,
    convention_share = 0.0,
)
evolutionary_policy = T.EvolutionaryReplacement(
    capability_mutation_rate = 0.02,
    trait_mutation_scale = 0.05,
)
evolutionary_habit = T.CapabilityModel(
    habit_share = 0.5,
    convention_share = 0.0,
    replacement_policy = evolutionary_policy,
)
evolutionary_control = T.CapabilityModel(
    habit_share = 0.0,
    convention_share = 0.0,
    replacement_policy = evolutionary_policy,
)

conditions = (
    ("Entry habit", entry_habit),
    ("Entry no habit", entry_control),
    ("Evolution habit", evolutionary_habit),
    ("Evolution no habit", evolutionary_control),
)
condition_list = collect(conditions)
observations = Vector{NamedTuple}(undef, length(condition_list) * length(SEEDS))
Threads.@threads for index in eachindex(observations)
    condition_index = (index - 1) ÷ length(SEEDS) + 1
    seed_index = (index - 1) % length(SEEDS) + 1
    model = condition_list[condition_index][2]
    observations[index] = metrics(first(simulate(model, SEEDS[seed_index])))
end
condition_observations = Dict(
    label => observations[
        ((condition_index - 1) * length(SEEDS) + 1):(condition_index * length(SEEDS))
    ]
        for (condition_index, (label, _)) in enumerate(condition_list)
)
for (label, _) in conditions
    condition_rows = condition_observations[label]
    names = propertynames(first(condition_rows))
    summary = NamedTuple{names}(Tuple(
        mean(getproperty(observation, name) for observation in condition_rows)
            for name in names
    ))
    println(replace(lowercase(label), ' ' => '_'), "=", summary)
end

mkpath(OUTPUT_DIR)
for (slug, model) in (
        ("entry", entry_habit),
        ("evolutionary", evolutionary_habit),
    )
    _, history = simulate(model, first(SEEDS); capture_every = 25)
    @assert all(T._capability_counts(snapshot)[5] == 0 for snapshot in history)
    save(
        joinpath(OUTPUT_DIR, "no_convention_$(slug)_dynamics.png"),
        T.plot_traffic_history(history),
    )
end

metrics_to_plot = (
    (:replacement_rate, "Post-burn-in replacement rate"),
    (:late_replacement_rate, "Final-1,000-tick replacement rate"),
    (:mean_speed, "Final mean speed"),
    (:realized_coordination, "Final realized convention"),
)
colors = (:seagreen3, :gray50, :firebrick2, :gray25)
figure = Figure(size = (1250, 760), fontsize = 15)
for (panel, (metric, ylabel)) in enumerate(metrics_to_plot)
    axis = Axis(
        figure[(panel - 1) ÷ 2 + 1, (panel - 1) % 2 + 1];
        ylabel = ylabel,
        xticks = (1:4, collect(first.(conditions))),
    )
    for (index, (label, _)) in enumerate(conditions)
        values = [getproperty(row, metric) for row in condition_observations[label]]
        scatter!(axis, fill(index, length(values)), values; color = (colors[index], 0.4))
        scatter!(axis, [index], [mean(values)]; color = colors[index], markersize = 14)
        errorbars!(
            axis,
            [index],
            [mean(values)],
            [std(values)],
            [std(values)];
            color = :black,
            whiskerwidth = 8,
        )
    end
end
Label(
    figure[0, :],
    "No-convention ablation — mean ± 1 SD across $(length(SEEDS)) paired runs";
    fontsize = 21,
    font = :bold,
)
save(joinpath(OUTPUT_DIR, "no_convention_comparison.png"), figure; px_per_unit = 2)
println("diagnostic outputs use current semantics: $OUTPUT_DIR")

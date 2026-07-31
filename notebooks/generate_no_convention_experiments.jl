using EBM
using CairoMakie
using Statistics

const T = EBM.Traffic
const SEEDS = 20260730:20260734
const STEPS = 300

function simulate(model, seed; capture_every = nothing)
    args = T.ModelArgs(
        seed = seed,
        params = T.ModelParams(),
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
        early_replacements = sum(logger.deaths[1:75]),
        late_replacements = sum(logger.deaths[226:300]),
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

println("entry_habit=", experiment(entry_habit))
println("entry_control=", experiment(entry_control))
println("evolutionary_habit=", experiment(evolutionary_habit))
println("evolutionary_control=", experiment(evolutionary_control))

mkpath("plots")
for (slug, model) in (
        ("entry", entry_habit),
        ("evolutionary", evolutionary_habit),
    )
    _, history = simulate(model, first(SEEDS); capture_every = 3)
    @assert all(T._capability_counts(snapshot)[5] == 0 for snapshot in history)
    save(
        "plots/no_convention_$(slug)_state.png",
        T.plot_traffic(last(history)),
    )
    save(
        "plots/no_convention_$(slug)_dynamics.png",
        T.plot_traffic_history(history),
    )
    T.record_traffic(
        history,
        "plots/no_convention_$(slug).mp4";
        framerate = 12,
    )
end

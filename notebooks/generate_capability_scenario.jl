using EBM
using CairoMakie
using Statistics

const T = EBM.Traffic

args = T.ModelArgs(
    seed = 20260730,
    params = T.ModelParams(),
    prediction_strategy = T.CapabilityModel(),
    steps = 300,
)

function run_history(args; every = 3)
    world = T.setup_world(args)
    history = T.TrafficSnapshot[T.traffic_snapshot(world; step = 0)]
    for step in 1:args.steps
        T.step!(world, args.prediction_strategy)
        if step % every == 0 || step == args.steps
            push!(history, T.traffic_snapshot(world; step = step))
        end
    end
    return world, history
end

world, history = run_history(args)
initial = first(history)
final = last(history)
logger = T.Ark.get_resource(world, T.Logger)
mean_speed(snapshot) = mean(car.speed for car in snapshot.cars)
convention(snapshot) = abs(mean(
    T.relative_lane_sign(car.lane, car.direction) for car in snapshot.cars
))

mkpath("plots")
save(
    "plots/capability_speed_scenario.png",
    T.plot_traffic(final),
)
save(
    "plots/capability_dynamics.png",
    T.plot_traffic_history(history),
)
T.record_traffic(
    history,
    "plots/capability_speed_scenario.mp4";
    framerate = 12,
)

evolutionary_args = T.ModelArgs(
    seed = args.seed,
    params = args.params,
    prediction_strategy = T.CapabilityModel(
        replacement_policy = T.EvolutionaryReplacement(
            capability_mutation_rate = 0.02,
            trait_mutation_scale = 0.05,
        ),
    ),
    steps = args.steps,
)
_, evolutionary_history = run_history(evolutionary_args)
save(
    "plots/capability_evolutionary_scenario.png",
    T.plot_traffic(last(evolutionary_history)),
)
save(
    "plots/capability_evolutionary_dynamics.png",
    T.plot_traffic_history(evolutionary_history),
)
T.record_traffic(
    evolutionary_history,
    "plots/capability_speed_evolutionary.mp4";
    framerate = 12,
)

ablation_args = T.ModelArgs(
    seed = args.seed,
    params = args.params,
    prediction_strategy = T.CapabilityModel(
        habit_share = 0.0,
        convention_share = 0.0,
    ),
    steps = args.steps,
)
_, ablation_history = run_history(ablation_args)
T.record_traffic(
    ablation_history,
    "plots/capability_speed_ablation.mp4";
    framerate = 12,
)

println("initial mean speed: ", round(mean_speed(initial); digits = 3))
println("final mean speed: ", round(mean_speed(final); digits = 3))
println("final speed-3 share: ", round(mean(car.speed == 3 for car in final.cars); digits = 3))
println("collision replacements: ", sum(logger.deaths))
println("final convention strength: ", round(convention(final); digits = 3))
println(
    "final capability shares: ",
    round.(T._capability_counts(final) ./ length(final.cars); digits = 3),
)
println("full animation: plots/capability_speed_scenario.mp4")
println("ablation animation: plots/capability_speed_ablation.mp4")
println("evolutionary animation: plots/capability_speed_evolutionary.mp4")

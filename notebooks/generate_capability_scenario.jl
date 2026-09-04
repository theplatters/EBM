using EBM
using CairoMakie
using Statistics

const T = EBM.Traffic

"""Set `TRAFFIC_PREFER_LANE=true` to run all scenarios lane-first."""
const PREFER_LANE_OVER_SPEED = get(ENV, "TRAFFIC_PREFER_LANE", "false") == "true"
const OUTPUT_DIR = get(ENV, "TRAFFIC_OUTPUT_DIR", "plots")

args = T.ModelArgs(
    seed = 20260730,
    params = T.ModelParams(lookahead = 20),
    prediction_strategy = T.CapabilityModel(
        social_habit_share = 0.5,
        prefer_lane_over_speed = PREFER_LANE_OVER_SPEED,
    ),
    steps = 5_000,
)

function run_history(args; every = 25)
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

mkpath(OUTPUT_DIR)
save(
    joinpath(OUTPUT_DIR, "capability_dynamics.png"),
    T.plot_traffic_history(history),
)

evolutionary_args = T.ModelArgs(
    seed = args.seed,
    params = args.params,
    prediction_strategy = T.CapabilityModel(
        prefer_lane_over_speed = PREFER_LANE_OVER_SPEED,
        replacement_policy = T.EvolutionaryReplacement(
            capability_mutation_rate = 0.02,
            trait_mutation_scale = 0.05,
        ),
    ),
    steps = args.steps,
)
_, evolutionary_history = run_history(evolutionary_args)
save(
    joinpath(OUTPUT_DIR, "capability_evolutionary_dynamics.png"),
    T.plot_traffic_history(evolutionary_history),
)

ablation_args = T.ModelArgs(
    seed = args.seed,
    params = args.params,
    prediction_strategy = T.CapabilityModel(
        habit_share = 0.0,
        convention_share = 0.0,
        social_habit_share = 0.0,
        prefer_lane_over_speed = PREFER_LANE_OVER_SPEED,
    ),
    steps = args.steps,
)
_, ablation_history = run_history(ablation_args)
save(
    joinpath(OUTPUT_DIR, "capability_ablation_dynamics.png"),
    T.plot_traffic_history(ablation_history),
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
println("static-entry dynamics: $(joinpath(OUTPUT_DIR, "capability_dynamics.png"))")
println("ablation dynamics: $(joinpath(OUTPUT_DIR, "capability_ablation_dynamics.png"))")
println("evolutionary dynamics: $(joinpath(OUTPUT_DIR, "capability_evolutionary_dynamics.png"))")

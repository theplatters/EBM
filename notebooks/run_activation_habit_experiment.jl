using CairoMakie
using Statistics

include(joinpath(@__DIR__, "activation_habit_common.jl"))
using .ActivationHabitExperiments

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

replicates = env_int("TRAFFIC_REPLICATES", 30)
first_seed = env_int("TRAFFIC_FIRST_SEED", 20260801)
steps = env_int("TRAFFIC_STEPS", 1_000)
burn_in = env_int("TRAFFIC_BURN_IN", steps ÷ 4)
config = ExperimentConfig(
    seeds = first_seed:(first_seed + replicates - 1),
    steps = steps,
    burn_in = burn_in,
    population = env_int("TRAFFIC_POPULATION", 120),
    ring_y = env_int("TRAFFIC_RING_Y", 300),
    lookahead = env_int("TRAFFIC_LOOKAHEAD", 60),
    error_rate = env_float("TRAFFIC_ERROR_RATE", 0.01),
    habit_weight = env_float("TRAFFIC_HABIT_WEIGHT", 0.5),
)

rows = run_experiment(config)
output_dir = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots"))
csv_path = write_results(joinpath(output_dir, "activation_habit_runs.csv"), rows)

conditions = [
    ("sequential", false, "Sequential\nno habit"),
    ("sequential", true, "Sequential\nhabit"),
    ("simultaneous", false, "Simultaneous\nno habit"),
    ("simultaneous", true, "Simultaneous\nhabit"),
]
palette = [:gray55, :seagreen3, :darkorange2, :dodgerblue3]

function condition_values(metric, timing, habit)
    return [
        getproperty(row, metric)
            for row in rows
            if row.timing == timing && row.habit == habit && isfinite(getproperty(row, metric))
    ]
end

function metric_panel!(axis, metric, ylabel)
    for (index, (timing, habit, _)) in enumerate(conditions)
        values = condition_values(metric, timing, habit)
        offsets = length(values) == 1 ? [0.0] : collect(range(-0.12, 0.12; length = length(values)))
        scatter!(
            axis,
            fill(index, length(values)) .+ offsets,
            values;
            color = (palette[index], 0.45),
            markersize = 7,
        )
        mean_value = mean(values)
        interval = ActivationHabitExperiments.bootstrap_interval(values)
        scatter!(axis, [index], [mean_value]; color = palette[index], markersize = 14)
        errorbars!(
            axis,
            [index],
            [mean_value],
            [mean_value - interval[1]],
            [interval[2] - mean_value];
            color = :black,
            whiskerwidth = 8,
        )
    end
    axis.xticks = (1:4, last.(conditions))
    axis.ylabel = ylabel
    return axis
end

figure = Figure(size = (1200, 760), fontsize = 15)
metric_panel!(Axis(figure[1, 1]), :compatibility_rate, "Compatible joint choices / encounters")
metric_panel!(Axis(figure[1, 2]), :precoordination_rate, "Pre-coordinated encounters")
metric_panel!(Axis(figure[2, 1]), :disposition_alignment, "Aligned dispositions before encounters")
metric_panel!(Axis(figure[2, 2]), :disposition_predictability, "Actions matching accumulated disposition")
Label(
    figure[0, :],
    "Activation timing × habit: paired stochastic replications";
    fontsize = 22,
    font = :bold,
)
plot_path = joinpath(output_dir, "activation_habit_results.png")
save(plot_path, figure; px_per_unit = 2)

println("wrote run-level data: $csv_path")
println("wrote visualization: $plot_path")
println("Next: julia --project=. notebooks/verify_activation_habit_results.jl")

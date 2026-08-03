using CairoMakie
using Statistics

include(joinpath(@__DIR__, "social_habit_common.jl"))
using .SocialHabitExperiments

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

replicates = env_int("TRAFFIC_REPLICATES", 30)
first_seed = env_int("TRAFFIC_FIRST_SEED", 20260901)
steps = env_int("TRAFFIC_STEPS", 5_000)
config = ExperimentConfig(
    seeds = first_seed:(first_seed + replicates - 1),
    steps = steps,
    burn_in = env_int("TRAFFIC_BURN_IN", steps ÷ 5),
    population = env_int("TRAFFIC_POPULATION", 120),
    ring_y = env_int("TRAFFIC_RING_Y", 300),
    lookahead = env_int("TRAFFIC_LOOKAHEAD", 20),
    error_rate = env_float("TRAFFIC_ERROR_RATE", 0.01),
    habit_weight = env_float("TRAFFIC_HABIT_WEIGHT", 0.5),
    convention_weight = env_float("TRAFFIC_CONVENTION_WEIGHT", 0.5),
    learning_rate = env_float("TRAFFIC_LEARNING_RATE", 0.2),
    observation_noise = env_float("TRAFFIC_OBSERVATION_NOISE", 0.05),
    trace_retention = env_float("TRAFFIC_TRACE_RETENTION", 0.9),
    trace_deposit = env_float("TRAFFIC_TRACE_DEPOSIT", 0.25),
    convention_target = env_float("TRAFFIC_CONVENTION_TARGET", 0.8),
    max_speed = env_int("TRAFFIC_MAX_SPEED", 3),
    mixture_share = env_float("TRAFFIC_MIXTURE_SHARE", 0.5),
    capability_mutation_rate = env_float("TRAFFIC_MUTATION_RATE", 0.02),
    trait_mutation_scale = env_float("TRAFFIC_MUTATION_SCALE", 0.05),
)

rows = run_experiment(config)
output_dir = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots"))
csv_path = write_results(joinpath(output_dir, "social_habit_runs.csv"), rows)

conditions = [
    ("synchronous_capability", :no_habit, "No habit"),
    ("synchronous_capability", :habit, "Personal\nhabit"),
    ("synchronous_capability", :convention, "Perceived\nconvention"),
    (
        "synchronous_capability",
        :socially_formed_habit,
        "Socially formed\nhabit",
    ),
    (
        "synchronous_capability",
        :mixture_static_replacement,
        "Mixed\nstatic entry",
    ),
    (
        "synchronous_capability",
        :mixture_evolutionary_replacement,
        "Mixed\nevolutionary",
    ),
    ("sequential_reference", :no_habit, "Sequential\nno habit"),
    ("sequential_reference", :habit, "Sequential\nhabit"),
]
palette = [
    :gray55,
    :seagreen3,
    :darkorange2,
    :dodgerblue3,
    :goldenrod2,
    :firebrick2,
    :gray30,
    :purple3,
]

function metric_panel!(axis, metric, ylabel)
    for (index, (implementation, scenario, _)) in enumerate(conditions)
        values = condition_values(rows, implementation, scenario, metric)
        offsets = length(values) == 1 ? [0.0] :
                  collect(range(-0.15, 0.15; length = length(values)))
        scatter!(
            axis,
            fill(index, length(values)) .+ offsets,
            values;
            color = (palette[index], 0.4),
            markersize = 7,
        )
        scatter!(
            axis,
            [index],
            [mean(values)];
            color = palette[index],
            markersize = 14,
        )
        interval = 1.96 * std(values) / sqrt(length(values))
        errorbars!(
            axis,
            [index],
            [mean(values)],
            [interval],
            [interval];
            color = :black,
            whiskerwidth = 8,
        )
    end
    axis.xticks = (1:length(conditions), last.(conditions))
    axis.ylabel = ylabel
    return axis
end

figure = Figure(size = (1720, 860), fontsize = 14)
metric_panel!(
    Axis(figure[1, 1]),
    :mean_convention_strength,
    "Mean post-burn-in convention strength",
)
metric_panel!(
    Axis(figure[1, 2]),
    :replacement_rate,
    "Replacements per car-step",
)
metric_panel!(
    Axis(figure[2, 1]),
    :coordinated_fraction,
    "Fraction of post-burn-in ticks at convention ≥ $(config.convention_target)",
)
metric_panel!(Axis(figure[2, 2]), :mean_speed, "Mean chosen speed")
Label(
    figure[0, :],
    "Habit and convention mechanisms: synchronous capabilities and sequential reference";
    fontsize = 21,
    font = :bold,
)
plot_path = joinpath(output_dir, "social_habit_comparison.png")
save(plot_path, figure; px_per_unit = 2)

mixture_conditions = [
    ("synchronous_capability", :mixture_static_replacement, "Static entry"),
    ("synchronous_capability", :mixture_evolutionary_replacement, "Evolutionary"),
]
composition_metrics = [
    (:final_habit_share, "Habit"),
    (:final_convention_share, "Convention"),
    (:final_social_habit_share, "Social habit"),
    (:final_all_three_share, "All three"),
]
composition_figure = Figure(size = (1050, 620), fontsize = 15)
composition_axis = Axis(
    composition_figure[1, 1];
    ylabel = "Final population share",
    xticks = (1:length(composition_metrics), last.(composition_metrics)),
    limits = (nothing, (0.0, 0.8)),
)
offsets = (-0.16, 0.16)
for (condition_index, (implementation, scenario, label)) in enumerate(mixture_conditions)
    for (metric_index, (metric, _)) in enumerate(composition_metrics)
        values = condition_values(rows, implementation, scenario, metric)
        x = metric_index + offsets[condition_index]
        scatter!(
            composition_axis,
            fill(x, length(values)),
            values;
            color = (palette[condition_index + 4], 0.28),
            markersize = 7,
        )
        interval = 1.96 * std(values) / sqrt(length(values))
        errorbars!(
            composition_axis,
            [x],
            [mean(values)],
            [interval],
            [interval];
            color = :black,
            whiskerwidth = 8,
        )
        scatter!(
            composition_axis,
            [x],
            [mean(values)];
            color = palette[condition_index + 4],
            markersize = 14,
            label = metric_index == 1 ? label : nothing,
        )
    end
end
scatter!(
    composition_axis,
    1:length(composition_metrics),
    [fill(config.mixture_share, 3); config.mixture_share^3];
    marker = :diamond,
    color = :transparent,
    strokecolor = :black,
    strokewidth = 2,
    markersize = 14,
    label = "Initial expectation",
)
axislegend(composition_axis; position = :rt)
Label(
    composition_figure[0, :],
    "Capability composition after static-entry and evolutionary replacement";
    fontsize = 21,
    font = :bold,
)
composition_path = joinpath(output_dir, "social_habit_mixture_composition.png")
save(composition_path, composition_figure; px_per_unit = 2)

println("wrote run-level data: $csv_path")
println("wrote visualization: $plot_path")
println("wrote mixture composition: $composition_path")

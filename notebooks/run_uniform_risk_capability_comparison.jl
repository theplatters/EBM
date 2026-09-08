# Current capability comparison for the presentation.
#
# The mixed evolutionary treatment deliberately intervenes after each capability
# tick: EvolutionaryReplacement evolves capabilities and their quantitative
# traits, but its temporary inherited risk value is discarded and newborn risk
# is replaced with an independent Uniform(0, 1) draw before the next decision.
# Risk is consequently non-heritable and fixed for each car's lifetime.
using CairoMakie
using EBM
using Statistics

include(joinpath(@__DIR__, "social_habit_common.jl"))
using .SocialHabitExperiments

const T = EBM.Traffic

const CAPABILITY_CONDITIONS = (
    (implementation = "synchronous_capability", scenario = :no_habit, label = "No habit", post_step_hook = nothing),
    (implementation = "synchronous_capability", scenario = :habit, label = "Personal\nhabit", post_step_hook = nothing),
    (implementation = "synchronous_capability", scenario = :convention, label = "Perceived\nconvention", post_step_hook = nothing),
    (implementation = "synchronous_capability", scenario = :socially_formed_habit, label = "SocialHabit", post_step_hook = nothing),
    (implementation = "synchronous_capability", scenario = :mixture_static_replacement, label = "Mixed\nstatic entry", post_step_hook = nothing),
    (implementation = "synchronous_capability", scenario = :mixture_evolutionary_replacement, label = "Mixed\nevolutionary", post_step_hook = uniform_newborn_risk!),
)
const REFERENCE_CONDITIONS = (
    (implementation = "sequential_reference", scenario = :no_habit, label = "Sequential\nno habit", post_step_hook = nothing),
    (implementation = "sequential_reference", scenario = :habit, label = "Sequential\nHabit", post_step_hook = nothing),
)
const CONDITIONS = (CAPABILITY_CONDITIONS..., REFERENCE_CONDITIONS...)

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

replicates = env_int("TRAFFIC_REPLICATES", 30)
first_seed = env_int("TRAFFIC_FIRST_SEED", 20260901)
steps = env_int("TRAFFIC_STEPS", 5_000)
config = ExperimentConfig(
    seeds = first_seed:(first_seed + replicates - 1),
    steps = steps,
    burn_in = env_int("TRAFFIC_BURN_IN", 1_000),
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
    capability_mutation_rate = env_float("TRAFFIC_CAPABILITY_MUTATION_RATE", 0.02),
    trait_mutation_scale = env_float("TRAFFIC_TRAIT_MUTATION_SCALE", 0.05),
)
SocialHabitExperiments.validate(config)

# Keep the treatment list auditable.  The core evolutionary policy is retained
# here for capability inheritance; the explicit hook discards only its
# temporary inherited risk value.
for condition in CAPABILITY_CONDITIONS
    model = capability_model(config, condition.scenario)
    if condition.scenario == :mixture_evolutionary_replacement
        @assert model.replacement_policy isa T.EvolutionaryReplacement
        @assert condition.post_step_hook === uniform_newborn_risk!
    else
        @assert model.replacement_policy isa T.EntryDrawReplacement
        @assert condition.post_step_hook === nothing
    end
end

jobs = [(seed, condition) for seed in config.seeds for condition in CONDITIONS]
rows = Vector{NamedTuple}(undef, length(jobs))
completed = Threads.Atomic{Int}(0)
progress_lock = ReentrantLock()
Threads.@threads for job_index in eachindex(jobs)
    seed, condition = jobs[job_index]
    rows[job_index] = condition.implementation == "synchronous_capability" ?
        run_capability_condition(
            config,
            seed,
            condition.scenario;
            post_step_hook = condition.post_step_hook,
        ) :
        run_sequential_condition(config, seed, condition.scenario)
    count = Threads.atomic_add!(completed, 1) + 1
    lock(progress_lock) do
        println("completed $count/$(length(jobs)): seed=$seed, scenario=$(condition.scenario)")
    end
end

output_dir = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots"))
SocialHabitExperiments.validate_results(rows)
csv_path = write_results(joinpath(output_dir, "uniform_risk_capability_runs.csv"), rows)

palette = (:gray55, :seagreen3, :darkorange2, :dodgerblue3, :goldenrod2, :firebrick2, :gray30, :purple3)
function metric_panel!(axis, metric, ylabel)
    for (index, condition) in enumerate(CONDITIONS)
        values = condition_values(rows, condition.implementation, condition.scenario, metric)
        values = filter(!isnan, values)
        isempty(values) && continue
        offsets = length(values) == 1 ? [0.0] : collect(range(-0.15, 0.15; length = length(values)))
        average = mean(values)
        interval = length(values) < 2 ? 0.0 : 1.96 * std(values) / sqrt(length(values))
        scatter!(axis, fill(index, length(values)) .+ offsets, values;
                 color = (palette[index], 0.4), markersize = 7)
        scatter!(axis, [index], [average]; color = palette[index], markersize = 14)
        errorbars!(axis, [index], [average], [interval], [interval];
                   color = :black, whiskerwidth = 8)
    end
    axis.xticks = (1:length(CONDITIONS), [condition.label for condition in CONDITIONS])
    axis.ylabel = ylabel
    axis.xticklabelrotation = π / 8
    return axis
end

figure = Figure(size = (1720, 860), fontsize = 14)
metric_panel!(Axis(figure[1, 1]), :mean_convention_strength,
              "Mean post-burn-in convention strength")
metric_panel!(Axis(figure[1, 2]), :replacement_rate,
              "Replacement pressure (per car-step)")
metric_panel!(Axis(figure[2, 1]), :coordinated_fraction,
              "Coordinated fraction (convention ≥ $(config.convention_target))")
metric_panel!(Axis(figure[2, 2]), :completed_cells_per_car_step,
              "Completed cells per car-step")
Label(figure[0, :],
      "Capability comparison — risk is Uniform(0,1), independently entry-drawn and non-heritable\ncapabilities may evolve in the mixed evolutionary treatment; replicate points, mean ± 95% mean interval";
      fontsize = 21, font = :bold)
plot_path = joinpath(output_dir, "uniform_risk_capability_comparison.png")
save(plot_path, figure; px_per_unit = 2)

mixture_conditions = [
    ("synchronous_capability", :mixture_static_replacement, "Mixed static entry"),
    ("synchronous_capability", :mixture_evolutionary_replacement, "Mixed evolutionary"),
]
composition_metrics = [
    (:final_habit_share, "Habit"),
    (:final_convention_share, "Convention"),
    (:final_social_habit_share, "SocialHabit"),
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
        scatter!(composition_axis, fill(x, length(values)), values;
                 color = (palette[condition_index + 4], 0.28), markersize = 7)
        interval = length(values) < 2 ? 0.0 : 1.96 * std(values) / sqrt(length(values))
        errorbars!(composition_axis, [x], [mean(values)], [interval], [interval];
                   color = :black, whiskerwidth = 8)
        scatter!(composition_axis, [x], [mean(values)];
                 color = palette[condition_index + 4], markersize = 14,
                 label = metric_index == 1 ? label : nothing)
    end
end
scatter!(composition_axis, 1:length(composition_metrics),
         [fill(config.mixture_share, 3); config.mixture_share^3];
         marker = :diamond, color = :transparent, strokecolor = :black,
         strokewidth = 2, markersize = 14, label = "Initial expectation")
axislegend(composition_axis; position = :rt)
Label(composition_figure[0, :],
      "Capability composition — uniformly entry-drawn, non-heritable risk\ncapabilities and quantitative traits may evolve; points, mean ± 95% mean interval";
      fontsize = 21, font = :bold)
composition_path = joinpath(output_dir, "uniform_risk_capability_composition.png")
save(composition_path, composition_figure; px_per_unit = 2)

println("wrote run-level data: $csv_path")
println("wrote visualization: $plot_path")
println("wrote composition visualization: $composition_path")

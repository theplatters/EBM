using CairoMakie
using EBM
using Statistics

include(joinpath(@__DIR__, "social_habit_common.jl"))
using .SocialHabitExperiments

const T = EBM.Traffic
const SCENARIO = :mixture_static_replacement

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

replicates = env_int("TRAFFIC_REPLICATES", 30)
first_seed = env_int("TRAFFIC_FIRST_SEED", 20260901)
steps = env_int("TRAFFIC_STEPS", 2_000)
sample_every = env_int("TRAFFIC_DYNAMIC_EVERY", 25)
burn_in = min(env_int("TRAFFIC_BURN_IN", 1_000), max(steps - 1, 0))
max_age = env_int("TRAFFIC_MAX_AVOIDANCE_DISABLE_AGE", 50)
config = ExperimentConfig(
    seeds = first_seed:(first_seed + replicates - 1),
    steps = steps,
    burn_in = burn_in,
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
replicates > 1 || error("TRAFFIC_REPLICATES must be at least 2 for confidence intervals")
sample_every > 0 || error("TRAFFIC_DYNAMIC_EVERY must be positive")
sample_every <= steps || error("TRAFFIC_DYNAMIC_EVERY must not exceed TRAFFIC_STEPS")
steps % sample_every == 0 ||
    error("TRAFFIC_STEPS must be divisible by TRAFFIC_DYNAMIC_EVERY")
max_age > 0 || error("TRAFFIC_MAX_AVOIDANCE_DISABLE_AGE must be positive")

function run_condition(config, seed; avoidance_disable_age)
    model = capability_model(
        config,
        SCENARIO;
        avoidance_disable_age = avoidance_disable_age,
    )
    @assert model.replacement_policy isa T.EntryDrawReplacement
    world = T.setup_world(T.ModelArgs(
        seed = seed,
        params = SocialHabitExperiments.model_params(config),
        prediction_strategy = model,
        steps = 0,
    ))
    conventions = Float64[]
    speeds = Float64[]
    deaths = 0
    for step in 1:config.steps
        T.step!(world, model)
        if step > config.burn_in
            deaths += last(T.Ark.get_resource(world, T.Logger).deaths)
            if step % sample_every == 0
                push!(conventions, SocialHabitExperiments.convention_strength(world))
                push!(speeds, SocialHabitExperiments.mean_speed(world))
            end
        end
    end
    isempty(conventions) && error("no post-burn samples; adjust TRAFFIC_STEPS or TRAFFIC_DYNAMIC_EVERY")
    return (
        mean_convention_strength = mean(conventions),
        replacement_rate = deaths / ((config.steps - config.burn_in) * config.population),
        mean_speed = mean(speeds),
    )
end

const COLUMNS = (
    :seed, :disable_age,
    :treatment_mean_convention_strength, :baseline_mean_convention_strength,
    :difference_mean_convention_strength,
    :treatment_replacement_rate, :baseline_replacement_rate,
    :difference_replacement_rate,
    :treatment_mean_speed, :baseline_mean_speed, :difference_mean_speed,
)

function paired_row(seed, age, treatment, baseline)
    return (
        seed = seed,
        disable_age = age,
        treatment_mean_convention_strength = treatment.mean_convention_strength,
        baseline_mean_convention_strength = baseline.mean_convention_strength,
        difference_mean_convention_strength = treatment.mean_convention_strength - baseline.mean_convention_strength,
        treatment_replacement_rate = treatment.replacement_rate,
        baseline_replacement_rate = baseline.replacement_rate,
        difference_replacement_rate = treatment.replacement_rate - baseline.replacement_rate,
        treatment_mean_speed = treatment.mean_speed,
        baseline_mean_speed = baseline.mean_speed,
        difference_mean_speed = treatment.mean_speed - baseline.mean_speed,
    )
end

ages = collect(1:max_age)
rows = Vector{NamedTuple}(undef, length(config.seeds) * length(ages))
completed = Threads.Atomic{Int}(0)
progress_lock = ReentrantLock()
Threads.@threads for seed_index in eachindex(config.seeds)
    seed = config.seeds[seed_index]
    baseline = run_condition(config, seed; avoidance_disable_age = nothing)
    offset = (seed_index - 1) * length(ages)
    for (age_index, age) in enumerate(ages)
        treatment = run_condition(config, seed; avoidance_disable_age = age)
        rows[offset + age_index] = paired_row(seed, age, treatment, baseline)
    end
    count = Threads.atomic_add!(completed, 1) + 1
    lock(progress_lock) do
        println("completed $count/$(length(config.seeds)): seed=$seed")
    end
end

output_dir = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots"))
mkpath(output_dir)
csv_path = joinpath(output_dir, "avoidance_age_robustness.csv")
open(csv_path, "w") do io
    println(io, join(string.(COLUMNS), ','))
    for row in rows
        println(io, join((getproperty(row, column) for column in COLUMNS), ','))
    end
end

function age_summary(metric)
    means = Float64[]
    intervals = Float64[]
    for age in ages
        values = [getproperty(row, metric) for row in rows if row.disable_age == age]
        push!(means, mean(values))
        push!(intervals, 1.96 * std(values) / sqrt(length(values)))
    end
    return means, intervals
end

figure = Figure(size = (1700, 650), fontsize = 16, backgroundcolor = :white)
Label(
    figure[0, 1:3],
    "Avoidance shutdown-age robustness — $replicates paired runs, post-burn metrics\n" *
    "Differences are treatment minus the same-seed no-shutdown baseline; ribbon is a 95% normal paired-mean CI",
    fontsize = 21,
    font = :bold,
    tellwidth = false,
)
for (column, (metric, title, ylabel)) in enumerate((
        (:difference_mean_convention_strength, "Convention strength", "treatment − baseline"),
        (:difference_replacement_rate, "Replacement rate", "treatment − baseline"),
        (:difference_mean_speed, "Mean speed", "treatment − baseline"),
    ))
    means, intervals = age_summary(metric)
    axis = Axis(figure[1, column]; title = title, xlabel = "shutdown age N", ylabel = ylabel)
    band!(axis, ages, means .- intervals, means .+ intervals; color = (:steelblue, 0.25))
    lines!(axis, ages, means; color = :steelblue4, linewidth = 3)
    hlines!(axis, [0.0]; color = :black, linewidth = 1)
    vlines!(axis, [50]; color = :firebrick, linewidth = 2, linestyle = :dash)
    xlims!(axis, 1, max_age)
end
rowgap!(figure.layout, 18)
plot_path = joinpath(output_dir, "avoidance_age_robustness.png")
save(plot_path, figure; px_per_unit = 2)
println("wrote $csv_path")
println("wrote $plot_path")

using CairoMakie
using EBM
using Statistics

include(joinpath(@__DIR__, "social_habit_common.jl"))
using .SocialHabitExperiments

const T = EBM.Traffic
const MIXTURE_SCENARIOS = (
    :mixture_static_replacement,
    :mixture_evolutionary_replacement,
)
const DYNAMIC_COLUMNS = (
    :seed,
    :scenario,
    :step,
    :convention_strength,
    :replacement_pressure,
    :mean_speed,
    :habit_share,
    :convention_share,
    :social_habit_share,
    :all_three_share,
    :effective_profiles,
)

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

replicates = env_int("TRAFFIC_REPLICATES", 30)
first_seed = env_int("TRAFFIC_FIRST_SEED", 20260901)
steps = env_int("TRAFFIC_STEPS", 5_000)
sample_every = env_int("TRAFFIC_DYNAMIC_EVERY", 25)
burn_in = min(env_int("TRAFFIC_BURN_IN", 1_000), max(steps - 1, 0))
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
replicates > 0 || error("TRAFFIC_REPLICATES must be positive")
sample_every > 0 || error("TRAFFIC_DYNAMIC_EVERY must be positive")
sample_every <= steps || error("TRAFFIC_DYNAMIC_EVERY must not exceed TRAFFIC_STEPS")
steps % sample_every == 0 ||
    error("TRAFFIC_STEPS must be divisible by TRAFFIC_DYNAMIC_EVERY")

for scenario in MIXTURE_SCENARIOS
    model = capability_model(config, scenario)
    hook = scenario == :mixture_evolutionary_replacement ? uniform_newborn_risk! : nothing
    if scenario == :mixture_evolutionary_replacement
        @assert model.replacement_policy isa T.EvolutionaryReplacement "evolutionary mixture must use EvolutionaryReplacement"
        @assert hook === uniform_newborn_risk! "evolutionary mixture must use uniform_newborn_risk!"
    else
        @assert model.replacement_policy isa T.EntryDrawReplacement "static mixture must use EntryDrawReplacement"
        @assert hook === nothing "static mixture must not use a post-step hook"
    end
end

function capability_share(cars, capability_index)
    isempty(cars) && return 0.0
    mask = UInt8(1) << (capability_index - 1)
    return count(car -> car.capabilities & mask != 0, cars) / length(cars)
end

function all_acquired_share(cars)
    isempty(cars) && return 0.0
    acquired_mask = sum(UInt8(1) << (index - 1) for index in 4:6)
    return count(
        car -> car.capabilities & acquired_mask == acquired_mask,
        cars,
    ) / length(cars)
end

function effective_profile_count(cars)
    isempty(cars) && return 0.0
    counts = Dict{UInt8, Int}()
    for car in cars
        counts[car.capabilities] = get(counts, car.capabilities, 0) + 1
    end
    entropy = -sum(
        (count / length(cars)) * log(count / length(cars)) for count in values(counts)
    )
    return exp(entropy)
end

function dynamic_row(world, seed, scenario, step, replacement_pressure)
    cars = T.traffic_snapshot(world; step = step).cars
    convention = isempty(cars) ? 0.0 : abs(
        mean(T.relative_lane_sign(car.lane, car.direction) for car in cars)
    )
    return (
        seed = seed,
        scenario = string(scenario),
        step = step,
        convention_strength = convention,
        replacement_pressure = replacement_pressure,
        mean_speed = isempty(cars) ? 0.0 : mean(car.speed for car in cars),
        habit_share = capability_share(cars, 4),
        convention_share = capability_share(cars, 5),
        social_habit_share = capability_share(cars, 6),
        all_three_share = all_acquired_share(cars),
        effective_profiles = effective_profile_count(cars),
    )
end

function run_dynamic_condition(config, seed, scenario)
    model = capability_model(config, scenario)
    world = T.setup_world(
        T.ModelArgs(
            seed = seed,
            params = SocialHabitExperiments.model_params(config),
            prediction_strategy = model,
            steps = 0,
        ),
    )
    hook = scenario == :mixture_evolutionary_replacement ? uniform_newborn_risk! : nothing
    rows = NamedTuple[dynamic_row(world, seed, scenario, 0, 0.0)]
    replacements = 0
    for step in 1:config.steps
        before = hook === nothing ? nothing : Set(
            entity
                for (entities, _) in T.Query(world, (T.Position,)) for entity in entities
        )
        T.step!(world, model)
        hook === nothing || hook(world, before, step)
        replacements += last(T.Ark.get_resource(world, T.Logger).deaths)
        if step % sample_every == 0
            push!(rows, dynamic_row(world, seed, scenario, step,
                                    replacements / (sample_every * config.population)))
            replacements = 0
        end
    end
    return rows
end

samples_per_condition = steps ÷ sample_every + 1
conditions_per_seed = length(MIXTURE_SCENARIOS)
rows = Vector{NamedTuple}(
    undef,
    length(config.seeds) * conditions_per_seed * samples_per_condition,
)
completed = Threads.Atomic{Int}(0)
progress_lock = ReentrantLock()
Threads.@threads for seed_index in eachindex(config.seeds)
    seed = config.seeds[seed_index]
    offset = (seed_index - 1) * conditions_per_seed * samples_per_condition
    for (scenario_index, scenario) in enumerate(MIXTURE_SCENARIOS)
        condition_rows = run_dynamic_condition(config, seed, scenario)
        first_index = offset + (scenario_index - 1) * samples_per_condition + 1
        rows[first_index:(first_index + samples_per_condition - 1)] = condition_rows
        count = Threads.atomic_add!(completed, 1) + 1
        lock(progress_lock) do
            println(
                "completed $count/$(length(config.seeds) * conditions_per_seed): " *
                "seed=$seed, scenario=$scenario",
            )
        end
    end
end

output_dir = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots"))
mkpath(output_dir)
csv_path = joinpath(output_dir, "uniform_risk_mixture_dynamics.csv")
open(csv_path, "w") do io
    println(io, join(string.(DYNAMIC_COLUMNS), ','))
    for row in rows
        println(io, join((getproperty(row, column) for column in DYNAMIC_COLUMNS), ','))
    end
end

function ensemble_summary(rows, scenario, metric)
    scenario_rows = filter(row -> row.scenario == string(scenario), rows)
    sample_steps = sort!(unique(row.step for row in scenario_rows))
    means = [
        mean([
            getproperty(row, metric) for row in scenario_rows if row.step == step
        ]) for step in sample_steps
    ]
    deviations = [
        std([
            getproperty(row, metric) for row in scenario_rows if row.step == step
        ]) for step in sample_steps
    ]
    return (; steps = sample_steps, means, standard_deviations = deviations)
end

conditions = (
    (:mixture_static_replacement, "Static entry", :darkorange2),
    (:mixture_evolutionary_replacement, "Evolutionary + uniform risk", :firebrick2),
)
function metric_panel!(axis, metric; limits = nothing, legend = false)
    for (scenario, label, color) in conditions
        summary = ensemble_summary(rows, scenario, metric)
        lower = summary.means .- summary.standard_deviations
        upper = summary.means .+ summary.standard_deviations
        if !isnothing(limits)
            lower = clamp.(lower, limits[1], limits[2])
            upper = clamp.(upper, limits[1], limits[2])
        end
        band!(axis, summary.steps, lower, upper; color = (color, 0.18))
        lines!(
            axis,
            summary.steps,
            summary.means;
            color = color,
            linewidth = 3,
            label = label,
        )
    end
    xlims!(axis, 0, steps)
    vlines!(axis, [config.burn_in]; color = (:black, 0.3), linestyle = :dash)
    !isnothing(limits) && ylims!(axis, limits...)
    legend && axislegend(axis; position = :rb, framevisible = false)
end

figure = Figure(size = (1850, 920), fontsize = 15, backgroundcolor = :white)
Label(
    figure[0, 1:4],
    "Mixed-capability dynamics — mean ± 1 SD across $replicates paired runs\n" *
    "Risk is independently entry-drawn and non-heritable; capabilities and quantitative traits may evolve",
    fontsize = 23,
    font = :bold,
    tellwidth = false,
)
for (column, (metric, title)) in enumerate((
        (:convention_strength, "Emergent lane convention"),
        (:replacement_pressure, "Collision-driven replacement pressure"),
        (:mean_speed, "Traffic throughput"),
        (:effective_profiles, "Effective strategy-profile diversity"),
    ))
    ylabel = column == 2 ? "replacements per car-step ($sample_every-tick window)" :
             column == 3 ? "mean chosen speed" :
             column == 4 ? "exp(Shannon entropy)" : "convention strength"
    limits = column == 1 ? (0.0, 1.0) :
             column == 2 ? (0.0, 0.35) :
             column == 3 ? (1.0, 3.05) : nothing
    metric_panel!(
        Axis(
            figure[1, column];
            title = title,
            xlabel = "step",
            ylabel = ylabel,
        ),
        metric;
        limits = limits,
        legend = column == 1,
    )
end
for (column, (metric, title)) in enumerate((
        (:habit_share, "Habit prevalence"),
        (:convention_share, "Convention prevalence"),
        (:social_habit_share, "SocialHabit prevalence"),
        (:all_three_share, "All-three prevalence"),
    ))
    metric_panel!(
        Axis(
            figure[2, column];
            title = title,
            xlabel = "step",
            ylabel = "population share",
        ),
        metric;
        limits = (0.0, 1.0),
    )
end
rowgap!(figure.layout, 16)
colgap!(figure.layout, 18)
plot_path = joinpath(output_dir, "uniform_risk_mixture_ensemble_dynamics.png")
save(plot_path, figure; px_per_unit = 2)
println("wrote uniform-risk mixture dynamics: $csv_path")
println("wrote mean ± 1 SD visualization: $plot_path")

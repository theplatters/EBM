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

replicates = env_int("TRAFFIC_REPLICATES", 30)
first_seed = env_int("TRAFFIC_FIRST_SEED", 20260901)
steps = env_int("TRAFFIC_STEPS", 5_000)
sample_every = env_int("TRAFFIC_DYNAMIC_EVERY", 25)
steps % sample_every == 0 || error("TRAFFIC_STEPS must be divisible by TRAFFIC_DYNAMIC_EVERY")

config = ExperimentConfig(
    seeds = first_seed:(first_seed + replicates - 1),
    steps = steps,
    burn_in = min(steps - 1, steps ÷ 5),
)

function capability_share(cars, capability_index)
    isempty(cars) && return 0.0
    mask = UInt8(1) << (capability_index - 1)
    return count(car -> car.capabilities & mask != 0, cars) / length(cars)
end

function all_acquired_share(cars)
    isempty(cars) && return 0.0
    acquired_mask = sum(UInt8(1) << (index - 1) for index in 4:6)
    return count(car -> car.capabilities & acquired_mask == acquired_mask, cars) /
           length(cars)
end

function effective_profile_count(cars)
    isempty(cars) && return 0.0
    counts = Dict{UInt8, Int}()
    for car in cars
        counts[car.capabilities] = get(counts, car.capabilities, 0) + 1
    end
    entropy = 0.0
    for count in values(counts)
        probability = count / length(cars)
        entropy -= probability * log(probability)
    end
    return exp(entropy)
end

function dynamic_row(world, seed, scenario, step, replacement_pressure)
    snapshot = T.traffic_snapshot(world; step = step)
    cars = snapshot.cars
    convention = isempty(cars) ? 0.0 : abs(mean(
        T.relative_lane_sign(car.lane, car.direction) for car in cars
    ))
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
    rows = NamedTuple[dynamic_row(world, seed, scenario, 0, 0.0)]
    replacements = 0
    for step in 1:config.steps
        T.step!(world, model)
        logger = T.Ark.get_resource(world, T.Logger)
        replacements += last(logger.deaths)
        if step % sample_every == 0
            pressure = replacements / (sample_every * config.population)
            push!(rows, dynamic_row(world, seed, scenario, step, pressure))
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
            println("completed $count/$(length(config.seeds) * conditions_per_seed): seed=$seed, scenario=$scenario")
        end
    end
end

output_dir = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots"))
mkpath(output_dir)
csv_path = joinpath(output_dir, "social_habit_mixture_dynamics.csv")
open(csv_path, "w") do io
    println(io, join(string.(DYNAMIC_COLUMNS), ','))
    for row in rows
        println(io, join((getproperty(row, column) for column in DYNAMIC_COLUMNS), ','))
    end
end

function ensemble_summary(rows, scenario, metric)
    scenario_rows = filter(row -> row.scenario == string(scenario), rows)
    sample_steps = sort!(unique(row.step for row in scenario_rows))
    means = Float64[]
    standard_deviations = Float64[]
    for step in sample_steps
        values = [
            getproperty(row, metric) for row in scenario_rows if row.step == step
        ]
        push!(means, mean(values))
        push!(standard_deviations, std(values))
    end
    return (; steps = sample_steps, means, standard_deviations)
end

conditions = (
    (:mixture_static_replacement, "Static entry", :darkorange2),
    (:mixture_evolutionary_replacement, "Evolutionary", :firebrick2),
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
    return axis
end

figure = Figure(size = (1850, 920), fontsize = 15, backgroundcolor = :white)
Label(
    figure[0, 1:4],
    "Monte Carlo dynamics of mixed capabilities — mean ± 1 SD across $replicates paired runs" *
    (SocialHabitExperiments.PREFER_LANE_OVER_SPEED ? " (lane-first)" : "");
    fontsize = 23,
    font = :bold,
    tellwidth = false,
)

metric_panel!(
    Axis(
        figure[1, 1];
        title = "Emergent lane convention",
        xlabel = "step",
        ylabel = "convention strength",
    ),
    :convention_strength;
    limits = (0.0, 1.0),
    legend = true,
)
metric_panel!(
    Axis(
        figure[1, 2];
        title = "Collision-driven replacement pressure",
        xlabel = "step",
        ylabel = "replacements per car-step ($sample_every-tick window)",
    ),
    :replacement_pressure;
    limits = (0.0, 0.35),
)
metric_panel!(
    Axis(
        figure[1, 3];
        title = "Traffic throughput",
        xlabel = "step",
        ylabel = "mean chosen speed",
    ),
    :mean_speed;
    limits = (1.0, 3.05),
)
metric_panel!(
    Axis(
        figure[1, 4];
        title = "Effective strategy-profile diversity",
        xlabel = "step",
        ylabel = "exp(Shannon entropy)",
    ),
    :effective_profiles,
)

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
plot_path = joinpath(output_dir, "social_habit_mixture_ensemble_dynamics.png")
save(plot_path, figure; px_per_unit = 2)

println("wrote ensemble dynamics: $csv_path")
println("wrote mean ± SD visualization: $plot_path")

using CairoMakie
using EBM
using Statistics

const T = EBM.Traffic
const SPEED_POLICIES = (
    (name = "speed-first", clearance = 0.0, prefer_lane = false, max_speed = 3),
    (name = "lane-first", clearance = 0.0, prefer_lane = true, max_speed = 3),
    (name = "lane-first + 0.5", clearance = 0.5, prefer_lane = true, max_speed = 3),
    (name = "lane-first cap 2", clearance = 0.0, prefer_lane = true, max_speed = 2),
)
const REPLACEMENT_TREATMENTS = (:static_entry, :evolutionary)
const RESULT_COLUMNS = (
    :seed,
    :replacement,
    :speed_policy,
    :speed_clearance,
    :steps,
    :burn_in,
    :lookahead,
    :mean_convention_strength,
    :coordinated_fraction,
    :replacement_rate,
    :mean_proposed_speed,
    :successful_progress,
    :speed_one_share,
    :speed_two_share,
    :speed_three_share,
)

env_int(name, default) = parse(Int, get(ENV, name, string(default)))

replicates = env_int("TRAFFIC_REPLICATES", 30)
first_seed = env_int("TRAFFIC_FIRST_SEED", 20261001)
steps = env_int("TRAFFIC_STEPS", 5_000)
burn_in = env_int("TRAFFIC_BURN_IN", steps ÷ 5)
population = env_int("TRAFFIC_POPULATION", 120)
ring_y = env_int("TRAFFIC_RING_Y", 300)
lookahead = env_int("TRAFFIC_LOOKAHEAD", 20)
seeds = first_seed:(first_seed + replicates - 1)

function treatment_model(replacement, policy)
    replacement_policy = replacement == :evolutionary ?
        T.EvolutionaryReplacement(
            capability_mutation_rate = 0.02,
            trait_mutation_scale = 0.05,
        ) : T.EntryDrawReplacement()
    return T.CapabilityModel(
        same_direction_share = 1.0,
        opposite_direction_share = 1.0,
        avoidance_share = 1.0,
        habit_share = 0.5,
        convention_share = 0.5,
        social_habit_share = 0.5,
        habit_weight = 0.5,
        convention_weight = 0.5,
        social_habit_weight = 0.5,
        convention_learning_rate = 0.2,
        convention_noise = 0.05,
        social_habit_learning_rate = 0.2,
        social_habit_noise = 0.05,
        social_trace_retention = 0.9,
        social_trace_deposit = 0.25,
        max_speed = policy.max_speed,
        speed_clearance = policy.clearance,
        prefer_lane_over_speed = policy.prefer_lane,
        replacement_policy = replacement_policy,
    )
end

function proposed_speed_statistics(world)
    counts = zeros(Int, 3)
    for (_, proposals) in T.Query(world, (T.SpeedProposal,))
        for proposal in proposals
            counts[proposal.value] += 1
        end
    end
    return counts
end

function survivor_progress(world, denominator)
    distance = 0
    for (_, speeds) in T.Query(world, (T.Speed,))
        distance += sum(speed.val for speed in speeds)
    end
    return distance / denominator
end

"""Mirror the capability step while measuring submitted and completed motion."""
function measured_step!(world)
    T.store_prev_positions!(world)
    T.rebuild_occupancy!(world)
    T.calculate_capability_proposals!(world)
    T.propose_speeds!(world)
    speed_counts = proposed_speed_statistics(world)

    replacements = T.resolve_capability_movement!(world)
    population = sum(speed_counts)
    progress = survivor_progress(world, population)
    T.update_success_traces!(world)
    T.spawn_new_entities!(world, replacements)

    habitus_task = Threads.@spawn T.update_habitus!(world)
    occupancy_task = Threads.@spawn T.rebuild_occupancy!(world)
    wait(habitus_task)
    T.update_mean_habitus!(world)
    wait(occupancy_task)
    T.logger!(world, T.Ark.get_resource(world, T.CapabilityModel))
    return length(replacements), speed_counts, progress
end

function convention_strength(world)
    total = 0.0
    count = 0
    for (_, positions, directions) in T.Query(world, (T.Position, T.Direction))
        for index in eachindex(positions)
            total += T.relative_lane_sign(positions[index].x, directions[index])
            count += 1
        end
    end
    return iszero(count) ? 0.0 : abs(total / count)
end

function run_condition(seed, replacement, policy)
    model = treatment_model(replacement, policy)
    params = T.ModelParams(
        δ = 0.2,
        ϵ = 0.01,
        init_agents = population,
        K = 10.0,
        lookahead = lookahead,
        ring_x = 2,
        ring_y = ring_y,
    )
    world = T.setup_world(
        T.ModelArgs(
            seed = seed,
            params = params,
            prediction_strategy = model,
            steps = 0,
        ),
    )
    conventions = Float64[]
    proposed_speeds = Float64[]
    successful_progress = Float64[]
    speed_counts = zeros(Int, 3)
    replacements = 0
    for step in 1:steps
        deaths, tick_speed_counts, progress = measured_step!(world)
        if step > burn_in
            push!(conventions, convention_strength(world))
            push!(proposed_speeds, sum(speed * tick_speed_counts[speed] for speed in 1:3) / population)
            push!(successful_progress, progress)
            speed_counts .+= tick_speed_counts
            replacements += deaths
        end
    end
    sampled_car_steps = (steps - burn_in) * population
    return (
        seed = seed,
        replacement = string(replacement),
        speed_policy = policy.name,
        speed_clearance = policy.clearance,
        steps = steps,
        burn_in = burn_in,
        lookahead = lookahead,
        mean_convention_strength = mean(conventions),
        coordinated_fraction = mean(conventions .>= 0.8),
        replacement_rate = replacements / sampled_car_steps,
        mean_proposed_speed = mean(proposed_speeds),
        successful_progress = mean(successful_progress),
        speed_one_share = speed_counts[1] / sampled_car_steps,
        speed_two_share = speed_counts[2] / sampled_car_steps,
        speed_three_share = speed_counts[3] / sampled_car_steps,
    )
end

jobs = [
    (seed, replacement, policy)
        for seed in seeds
        for replacement in REPLACEMENT_TREATMENTS
        for policy in SPEED_POLICIES
]
output_dir = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots"))
mkpath(output_dir)
partial_path = joinpath(output_dir, "speed_sensitivity_partial.csv")

function parse_result(parts)
    length(parts) == length(RESULT_COLUMNS) || error("invalid speed-sensitivity checkpoint row")
    return (
        seed = parse(Int, parts[1]),
        replacement = parts[2],
        speed_policy = parts[3],
        speed_clearance = parse(Float64, parts[4]),
        steps = parse(Int, parts[5]),
        burn_in = parse(Int, parts[6]),
        lookahead = parse(Int, parts[7]),
        mean_convention_strength = parse(Float64, parts[8]),
        coordinated_fraction = parse(Float64, parts[9]),
        replacement_rate = parse(Float64, parts[10]),
        mean_proposed_speed = parse(Float64, parts[11]),
        successful_progress = parse(Float64, parts[12]),
        speed_one_share = parse(Float64, parts[13]),
        speed_two_share = parse(Float64, parts[14]),
        speed_three_share = parse(Float64, parts[15]),
    )
end

row_key(row) = (row.seed, row.replacement, row.speed_policy)
job_key(job) = (job[1], string(job[2]), job[3].name)
rows_by_key = Dict{Tuple{Int,String,String},NamedTuple}()
if isfile(partial_path)
    lines = readlines(partial_path)
    Symbol.(split(first(lines), ',')) == collect(RESULT_COLUMNS) ||
        error("unexpected checkpoint schema in $partial_path")
    for line in Iterators.drop(lines, 1)
        isempty(line) && continue
        row = parse_result(split(line, ','))
        row.steps == steps && row.burn_in == burn_in && row.lookahead == lookahead ||
            error("checkpoint settings do not match the requested experiment")
        rows_by_key[row_key(row)] = row
    end
else
    open(partial_path, "w") do io
        println(io, join(RESULT_COLUMNS, ','))
    end
end

pending_indices = [index for index in eachindex(jobs) if job_key(jobs[index]) ∉ keys(rows_by_key)]
completed = Threads.Atomic{Int}(0)
progress_lock = ReentrantLock()
Threads.@threads for pending_index in eachindex(pending_indices)
    index = pending_indices[pending_index]
    seed, replacement, policy = jobs[index]
    row = run_condition(seed, replacement, policy)
    count = Threads.atomic_add!(completed, 1) + 1
    lock(progress_lock) do
        rows_by_key[row_key(row)] = row
        open(partial_path, "a") do io
            println(io, join((getproperty(row, column) for column in RESULT_COLUMNS), ','))
        end
        println(
            "completed $(length(rows_by_key))/$(length(jobs)) " *
            "(this run $count/$(length(pending_indices))): seed=$seed, " *
            "replacement=$replacement, speed_policy=$(policy.name)",
        )
    end
end

rows = [rows_by_key[job_key(job)] for job in jobs]
csv_path = joinpath(output_dir, "speed_sensitivity_runs.csv")
open(csv_path, "w") do io
    println(io, join(RESULT_COLUMNS, ','))
    for row in rows
        println(io, join((getproperty(row, column) for column in RESULT_COLUMNS), ','))
    end
end

metrics = (
    (:mean_convention_strength, "Mean convention strength"),
    (:coordinated_fraction, "Fraction of ticks at convention ≥ 0.8"),
    (:replacement_rate, "Replacements per car-step"),
    (:successful_progress, "Successfully completed cells per car-step"),
    (:mean_proposed_speed, "Mean proposed speed"),
    (:speed_three_share, "Share of proposals at speed 3"),
)
colors = Dict(:static_entry => :darkorange2, :evolutionary => :firebrick2)
figure = Figure(size = (1500, 1050), fontsize = 15)
for (panel, (metric, ylabel)) in enumerate(metrics)
    axis = Axis(
        figure[(panel - 1) ÷ 2 + 1, (panel - 1) % 2 + 1];
        xlabel = "speed-choice policy",
        ylabel = ylabel,
        xticks = (1:length(SPEED_POLICIES), collect(getproperty.(SPEED_POLICIES, :name))),
    )
    for replacement in REPLACEMENT_TREATMENTS
        means = Float64[]
        deviations = Float64[]
        for policy in SPEED_POLICIES
            values = [
                getproperty(row, metric) for row in rows
                    if row.replacement == string(replacement) &&
                       row.speed_policy == policy.name
            ]
            scatter!(axis, fill(length(means) + 1, length(values)), values; color = (colors[replacement], 0.22))
            push!(means, mean(values))
            push!(deviations, std(values))
        end
        x = collect(1:length(SPEED_POLICIES))
        errorbars!(axis, x, means, deviations, deviations; color = colors[replacement], whiskerwidth = 7)
        lines!(axis, x, means; color = colors[replacement], linewidth = 3, label = replace(string(replacement), '_' => ' '))
        scatter!(axis, x, means; color = colors[replacement], markersize = 12)
    end
    panel == 1 && axislegend(axis; position = :rt, framevisible = false)
end
Label(
    figure[0, :],
    "Speed sensitivity in mixed-capability traffic — mean ± 1 SD across $replicates paired runs";
    fontsize = 22,
    font = :bold,
)
plot_path = joinpath(output_dir, "speed_sensitivity_comparison.png")
save(plot_path, figure; px_per_unit = 2)
rm(partial_path)

println("wrote $csv_path")
println("wrote $plot_path")

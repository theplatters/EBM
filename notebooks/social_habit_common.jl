module SocialHabitExperiments

using Agents
using EBM
using Statistics

const T = EBM.Traffic
const S = T.SequentialModel

"""Set `TRAFFIC_PREFER_LANE=true` to run every capability condition lane-first."""
const PREFER_LANE_OVER_SPEED = get(ENV, "TRAFFIC_PREFER_LANE", "false") == "true"

const CAPABILITY_SCENARIOS = (
    :no_habit,
    :habit,
    :convention,
    :socially_formed_habit,
    :mixture_static_replacement,
    :mixture_evolutionary_replacement,
)
const SEQUENTIAL_SCENARIOS = (:no_habit, :habit)

Base.@kwdef struct ExperimentConfig
    seeds::UnitRange{Int} = 20260901:20260930
    steps::Int = 5_000
    burn_in::Int = 1_000
    population::Int = 120
    ring_y::Int = 300
    lookahead::Int = 20
    error_rate::Float64 = 0.01
    habit_weight::Float64 = 0.5
    convention_weight::Float64 = 0.5
    learning_rate::Float64 = 0.2
    observation_noise::Float64 = 0.05
    trace_retention::Float64 = 0.9
    trace_deposit::Float64 = 0.25
    convention_target::Float64 = 0.8
    max_speed::Int = 3
    mixture_share::Float64 = 0.5
    capability_mutation_rate::Float64 = 0.02
    trait_mutation_scale::Float64 = 0.05
end

const RESULT_COLUMNS = (
    :seed,
    :implementation,
    :scenario,
    :steps,
    :burn_in,
    :population,
    :ring_y,
    :lookahead,
    :error_rate,
    :max_speed,
    :mean_convention_strength,
    :final_convention_strength,
    :mean_disposition_strength,
    :mean_speed,
    :replacement_rate,
    :time_to_convention,
    :coordinated_fraction,
    :coordination_entries,
    :mean_coordinated_episode,
    :longest_coordinated_episode,
    :final_habit_share,
    :final_convention_share,
    :final_social_habit_share,
    :final_all_three_share,
)

function validate(config::ExperimentConfig)
    0 <= config.burn_in < config.steps ||
        throw(ArgumentError("burn_in must satisfy 0 <= burn_in < steps"))
    iseven(config.population) || throw(ArgumentError("population must be even"))
    config.population <= 2 * config.ring_y ||
        throw(ArgumentError("population exceeds the two-lane torus capacity"))
    0.0 <= config.convention_target <= 1.0 ||
        throw(ArgumentError("convention_target must be in [0, 1]"))
    0.0 <= config.trace_retention < 1.0 ||
        throw(ArgumentError("trace_retention must be in [0, 1)"))
    config.trace_deposit > 0.0 || throw(ArgumentError("trace_deposit must be positive"))
    1 <= config.max_speed <= 3 || throw(ArgumentError("max_speed must be in 1:3"))
    0.0 <= config.mixture_share <= 1.0 ||
        throw(ArgumentError("mixture_share must be in [0, 1]"))
    0.0 <= config.capability_mutation_rate <= 1.0 ||
        throw(ArgumentError("capability_mutation_rate must be in [0, 1]"))
    config.trait_mutation_scale >= 0.0 ||
        throw(ArgumentError("trait_mutation_scale must be nonnegative"))
    return config
end

function model_params(config)
    return T.ModelParams(
        δ = 0.2,
        ϵ = config.error_rate,
        init_agents = config.population,
        K = 10.0,
        lookahead = config.lookahead,
        ring_x = 2,
        ring_y = config.ring_y,
    )
end

function capability_model(config, scenario::Symbol)
    scenario in CAPABILITY_SCENARIOS ||
        throw(ArgumentError("unknown capability scenario: $scenario"))
    mixed = scenario in (
        :mixture_static_replacement,
        :mixture_evolutionary_replacement,
    )
    replacement_policy = scenario == :mixture_evolutionary_replacement ?
        T.EvolutionaryReplacement(
            capability_mutation_rate = config.capability_mutation_rate,
            trait_mutation_scale = config.trait_mutation_scale,
        ) : T.EntryDrawReplacement()
    return T.CapabilityModel(
        same_direction_share = 1.0,
        opposite_direction_share = 1.0,
        avoidance_share = 1.0,
        habit_share = scenario == :habit ? 1.0 : mixed ? config.mixture_share : 0.0,
        convention_share = scenario == :convention ? 1.0 : mixed ? config.mixture_share : 0.0,
        social_habit_share = scenario == :socially_formed_habit ?
            1.0 : mixed ? config.mixture_share : 0.0,
        habit_weight = config.habit_weight,
        convention_weight = config.convention_weight,
        social_habit_weight = config.habit_weight,
        convention_learning_rate = config.learning_rate,
        convention_noise = config.observation_noise,
        social_habit_learning_rate = config.learning_rate,
        social_habit_noise = config.observation_noise,
        social_trace_retention = config.trace_retention,
        social_trace_deposit = config.trace_deposit,
        max_speed = config.max_speed,
        prefer_lane_over_speed = PREFER_LANE_OVER_SPEED,
        replacement_policy = replacement_policy,
    )
end

function convention_strength(world)
    total = 0.0
    count = 0
    for (entities, positions, directions) in T.Query(
            world, (T.Position, T.Direction),
        )
        @inbounds for index in eachindex(entities)
            total += T.relative_lane_sign(positions[index].x, directions[index])
            count += 1
        end
    end
    return iszero(count) ? 0.0 : abs(total / count)
end

function component_mean_abs(world, ::Type{Component}, value) where {Component}
    total = 0.0
    count = 0
    for (_, components) in T.Query(world, (Component,))
        for component in components
            total += abs(value(component))
            count += 1
        end
    end
    return iszero(count) ? 0.0 : total / count
end

function disposition_strength(world, scenario)
    scenario == :habit &&
        return component_mean_abs(world, T.Habitus, component -> component.val)
    scenario == :convention &&
        return component_mean_abs(
            world,
            T.PerceivedConvention,
            component -> component.confidence * component.value,
        )
    scenario == :socially_formed_habit &&
        return component_mean_abs(world, T.SocialHabitus, component -> component.value)
    if scenario in (:mixture_static_replacement, :mixture_evolutionary_replacement)
        total = 0.0
        count = 0
        for (_, components) in T.Query(world, (T.Habitus,))
            for component in components
                total += abs(component.val)
                count += 1
            end
        end
        for (_, components) in T.Query(world, (T.PerceivedConvention,))
            for component in components
                total += abs(component.confidence * component.value)
                count += 1
            end
        end
        for (_, components) in T.Query(world, (T.SocialHabitus,))
            for component in components
                total += abs(component.value)
                count += 1
            end
        end
        return iszero(count) ? 0.0 : total / count
    end
    return 0.0
end

function final_capability_shares(world)
    query_count(query) = mapreduce(
        result -> length(first(result)),
        +,
        query;
        init = 0,
    )
    population = query_count(T.Query(world, (T.Position,)))
    population == 0 && return (habit = 0.0, convention = 0.0, social = 0.0, all = 0.0)
    share(component) = query_count(T.Query(world, (component,))) / population
    all_three = query_count(
        T.Query(
            world,
            (T.HabitFormation, T.ConventionPerception, T.SocialHabitFormation),
        ),
    ) / population
    return (
        habit = share(T.HabitFormation),
        convention = share(T.ConventionPerception),
        social = share(T.SocialHabitFormation),
        all = all_three,
    )
end

function mean_speed(world)
    total = 0
    count = 0
    for (_, speeds) in T.Query(world, (T.Speed,))
        total += sum(speed.val for speed in speeds)
        count += length(speeds)
    end
    return iszero(count) ? 0.0 : total / count
end

function summarize_samples(
        config, seed, implementation, scenario, conventions, dispositions,
        speeds, deaths, time_to_convention;
        max_speed,
        capability_shares = (habit = NaN, convention = NaN, social = NaN, all = NaN),
    )
    sampled_steps = config.steps - config.burn_in
    coordinated = conventions .>= config.convention_target
    episode_lengths = Int[]
    current_episode = 0
    for is_coordinated in coordinated
        if is_coordinated
            current_episode += 1
        elseif current_episode > 0
            push!(episode_lengths, current_episode)
            current_episode = 0
        end
    end
    current_episode > 0 && push!(episode_lengths, current_episode)
    coordination_entries = count(eachindex(coordinated)) do index
        coordinated[index] && (index == firstindex(coordinated) || !coordinated[index - 1])
    end
    return (
        seed = seed,
        implementation = implementation,
        scenario = string(scenario),
        steps = config.steps,
        burn_in = config.burn_in,
        population = config.population,
        ring_y = config.ring_y,
        lookahead = config.lookahead,
        error_rate = config.error_rate,
        max_speed = max_speed,
        mean_convention_strength = mean(conventions),
        final_convention_strength = last(conventions),
        mean_disposition_strength = mean(dispositions),
        mean_speed = mean(speeds),
        replacement_rate = deaths / (sampled_steps * config.population),
        time_to_convention = something(time_to_convention, -1),
        coordinated_fraction = mean(coordinated),
        coordination_entries = coordination_entries,
        mean_coordinated_episode = isempty(episode_lengths) ? 0.0 : mean(episode_lengths),
        longest_coordinated_episode = isempty(episode_lengths) ? 0 : maximum(episode_lengths),
        final_habit_share = capability_shares.habit,
        final_convention_share = capability_shares.convention,
        final_social_habit_share = capability_shares.social,
        final_all_three_share = capability_shares.all,
    )
end

function run_capability_condition(config, seed, scenario)
    validate(config)
    model = capability_model(config, scenario)
    world = T.setup_world(
        T.ModelArgs(
            seed = seed,
            params = model_params(config),
            prediction_strategy = model,
            steps = 0,
        ),
    )
    conventions = Float64[]
    dispositions = Float64[]
    speeds = Float64[]
    deaths = 0
    time_to_convention = nothing

    for step in 1:config.steps
        T.step!(world, model)
        strength = convention_strength(world)
        isnothing(time_to_convention) && strength >= config.convention_target &&
            (time_to_convention = step)
        if step > config.burn_in
            push!(conventions, strength)
            push!(dispositions, disposition_strength(world, scenario))
            push!(speeds, mean_speed(world))
            logger = T.Ark.get_resource(world, T.Logger)
            deaths += last(logger.deaths)
        end
    end

    return summarize_samples(
        config,
        seed,
        "synchronous_capability",
        scenario,
        conventions,
        dispositions,
        speeds,
        deaths,
        time_to_convention;
        max_speed = config.max_speed,
        capability_shares = final_capability_shares(world),
    )
end

function sequential_convention_strength(model)
    cars = collect(allagents(model))
    isempty(cars) && return 0.0
    return abs(mean(S.relative_lane_sign(car.pos[1], car.direction) for car in cars))
end

function run_sequential_condition(config, seed, scenario)
    validate(config)
    scenario in SEQUENTIAL_SCENARIOS ||
        throw(ArgumentError("unknown sequential scenario: $scenario"))
    weights = T.Weights(
        wₛ = 0.5,
        wₒ = 0.5,
        wₐ = 0.5,
        wₕ = scenario == :habit ? config.habit_weight : 0.0,
    )
    model = S.init_model(
        model_params(config),
        weights;
        seed = seed,
        timing = S.SequentialActivation(),
    )
    conventions = Float64[]
    dispositions = Float64[]
    speeds = Float64[]
    deaths = 0
    time_to_convention = nothing

    for step in 1:config.steps
        Agents.step!(model, 1)
        strength = sequential_convention_strength(model)
        isnothing(time_to_convention) && strength >= config.convention_target &&
            (time_to_convention = step)
        if step > config.burn_in
            cars = collect(allagents(model))
            push!(conventions, strength)
            push!(
                dispositions,
                scenario == :habit ? mean(abs(car.habitus) for car in cars) : 0.0,
            )
            push!(speeds, 1.0)
            deaths += last(model.diagnostics.deaths)
        end
    end

    return summarize_samples(
        config,
        seed,
        "sequential_reference",
        scenario,
        conventions,
        dispositions,
        speeds,
        deaths,
        time_to_convention;
        max_speed = 1,
    )
end

function run_experiment(
        config = ExperimentConfig();
        progress = true,
        threaded = Threads.nthreads() > 1,
    )
    validate(config)
    conditions_per_seed = length(CAPABILITY_SCENARIOS) + length(SEQUENTIAL_SCENARIOS)
    total = length(config.seeds) * conditions_per_seed
    rows = Vector{NamedTuple}(undef, total)
    completed = Threads.Atomic{Int}(0)
    progress_lock = ReentrantLock()

    function run_seed!(seed_index)
        seed = config.seeds[seed_index]
        row_index = (seed_index - 1) * conditions_per_seed
        for scenario in CAPABILITY_SCENARIOS
            row_index += 1
            rows[row_index] = run_capability_condition(config, seed, scenario)
            count = Threads.atomic_add!(completed, 1) + 1
            if progress
                lock(progress_lock) do
                    println(
                        "completed $count/$total: seed=$seed, " *
                        "implementation=synchronous_capability, scenario=$scenario",
                    )
                end
            end
        end
        for scenario in SEQUENTIAL_SCENARIOS
            row_index += 1
            rows[row_index] = run_sequential_condition(config, seed, scenario)
            count = Threads.atomic_add!(completed, 1) + 1
            if progress
                lock(progress_lock) do
                    println(
                        "completed $count/$total: seed=$seed, " *
                        "implementation=sequential_reference, scenario=$scenario",
                    )
                end
            end
        end
        return nothing
    end

    seed_indices = eachindex(config.seeds)
    if threaded
        Threads.@threads for seed_index in seed_indices
            run_seed!(seed_index)
        end
    else
        for seed_index in seed_indices
            run_seed!(seed_index)
        end
    end
    return rows
end

function write_results(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(RESULT_COLUMNS), ','))
        for row in rows
            println(io, join((getproperty(row, column) for column in RESULT_COLUMNS), ','))
        end
    end
    return path
end

function parse_result(parts)
    length(parts) == length(RESULT_COLUMNS) || error("unexpected result column count")
    return (
        seed = parse(Int, parts[1]),
        implementation = parts[2],
        scenario = parts[3],
        steps = parse(Int, parts[4]),
        burn_in = parse(Int, parts[5]),
        population = parse(Int, parts[6]),
        ring_y = parse(Int, parts[7]),
        lookahead = parse(Int, parts[8]),
        error_rate = parse(Float64, parts[9]),
        max_speed = parse(Int, parts[10]),
        mean_convention_strength = parse(Float64, parts[11]),
        final_convention_strength = parse(Float64, parts[12]),
        mean_disposition_strength = parse(Float64, parts[13]),
        mean_speed = parse(Float64, parts[14]),
        replacement_rate = parse(Float64, parts[15]),
        time_to_convention = parse(Int, parts[16]),
        coordinated_fraction = parse(Float64, parts[17]),
        coordination_entries = parse(Int, parts[18]),
        mean_coordinated_episode = parse(Float64, parts[19]),
        longest_coordinated_episode = parse(Int, parts[20]),
        final_habit_share = parse(Float64, parts[21]),
        final_convention_share = parse(Float64, parts[22]),
        final_social_habit_share = parse(Float64, parts[23]),
        final_all_three_share = parse(Float64, parts[24]),
    )
end

function read_results(path)
    lines = readlines(path)
    isempty(lines) && error("empty result file: $path")
    Symbol.(split(first(lines), ',')) == collect(RESULT_COLUMNS) ||
        error("unexpected result schema in $path")
    return [
        parse_result(split(line, ','))
            for line in Iterators.drop(lines, 1) if !isempty(line)
    ]
end

function validate_results(rows)
    isempty(rows) && error("no experiment rows")
    seeds = sort!(unique(row.seed for row in rows))
    for seed in seeds
        for scenario in string.(CAPABILITY_SCENARIOS)
            count(
                row -> row.seed == seed &&
                       row.implementation == "synchronous_capability" &&
                       row.scenario == scenario,
                rows,
            ) == 1 || error("missing capability row for seed=$seed, scenario=$scenario")
        end
        for scenario in string.(SEQUENTIAL_SCENARIOS)
            count(
                row -> row.seed == seed &&
                       row.implementation == "sequential_reference" &&
                       row.scenario == scenario,
                rows,
            ) == 1 || error("missing sequential row for seed=$seed, scenario=$scenario")
        end
    end
    all(row -> 0.0 <= row.replacement_rate <= 1.0, rows) ||
        error("replacement rates must be in [0, 1]")
    return seeds
end

function condition_values(rows, implementation, scenario, metric)
    return [
        getproperty(row, metric)
            for row in rows
            if row.implementation == implementation && row.scenario == string(scenario)
    ]
end

export CAPABILITY_SCENARIOS,
    ExperimentConfig,
    RESULT_COLUMNS,
    SEQUENTIAL_SCENARIOS,
    capability_model,
    condition_values,
    read_results,
    run_capability_condition,
    run_experiment,
    run_sequential_condition,
    validate_results,
    write_results

end

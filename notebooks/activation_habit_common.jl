module ActivationHabitExperiments

using Agents
using EBM
using Random
using Statistics

const T = EBM.Traffic
const S = T.SequentialModel

Base.@kwdef struct ExperimentConfig
    seeds::UnitRange{Int} = 20260801:20260830
    steps::Int = 1_000
    burn_in::Int = 250
    population::Int = 120
    ring_y::Int = 300
    lookahead::Int = 60
    error_rate::Float64 = 0.01
    habit_weight::Float64 = 0.5
end

const RESULT_COLUMNS = (
    :seed,
    :timing,
    :habit,
    :steps,
    :burn_in,
    :population,
    :ring_y,
    :lookahead,
    :error_rate,
    :habit_weight,
    :encounters,
    :failures,
    :encounter_rate,
    :failure_rate,
    :compatibility_rate,
    :precoordination_rate,
    :disposition_alignment,
    :disposition_predictability,
    :death_rate,
    :persistence,
    :convention_strength,
    :mean_abs_habitus,
)

safe_ratio(numerator, denominator) = iszero(denominator) ? NaN : numerator / denominator

timing_name(::S.SequentialActivation) = "sequential"
timing_name(::S.SimultaneousActivation) = "simultaneous"

function summarize_run(model, config, seed, timing, habit)
    diagnostics = model.diagnostics
    first_tick = config.burn_in + 1
    sample_ticks = first_tick:config.steps
    encounters = sum(@view diagnostics.encounter_pairs[sample_ticks])
    failures = sum(@view diagnostics.failed_encounters[sample_ticks])
    compatible = sum(@view diagnostics.compatible_encounters[sample_ticks])
    precoordinated = sum(@view diagnostics.precoordinated_encounters[sample_ticks])
    aligned_dispositions = sum(@view diagnostics.aligned_disposition_encounters[sample_ticks])
    observed_dispositions = sum(@view diagnostics.observed_disposition_encounters[sample_ticks])
    disposition_consistent = sum(@view diagnostics.disposition_consistent_actions[sample_ticks])
    disposition_actions = sum(@view diagnostics.disposition_observed_actions[sample_ticks])
    persistent = sum(@view diagnostics.persistent_actions[sample_ticks])
    actions = sum(@view diagnostics.observed_actions[sample_ticks])
    deaths = sum(@view diagnostics.deaths[sample_ticks])
    cars = collect(allagents(model))
    convention = abs(mean(S.relative_lane_sign(car.pos[1], car.direction) for car in cars))

    return (
        seed = seed,
        timing = timing_name(timing),
        habit = habit,
        steps = config.steps,
        burn_in = config.burn_in,
        population = config.population,
        ring_y = config.ring_y,
        lookahead = config.lookahead,
        error_rate = config.error_rate,
        habit_weight = config.habit_weight,
        encounters = encounters,
        failures = failures,
        encounter_rate = encounters / (length(sample_ticks) * config.population),
        failure_rate = safe_ratio(failures, encounters),
        compatibility_rate = safe_ratio(compatible, encounters),
        precoordination_rate = safe_ratio(precoordinated, observed_dispositions),
        disposition_alignment = safe_ratio(aligned_dispositions, observed_dispositions),
        disposition_predictability = safe_ratio(disposition_consistent, disposition_actions),
        death_rate = deaths / (length(sample_ticks) * config.population),
        persistence = safe_ratio(persistent, actions),
        convention_strength = convention,
        mean_abs_habitus = mean(abs(car.habitus) for car in cars),
    )
end

function run_condition(config, seed, timing, habit)
    0 <= config.burn_in < config.steps ||
        throw(ArgumentError("burn_in must satisfy 0 <= burn_in < steps"))
    iseven(config.population) || throw(ArgumentError("population must be even"))
    config.population <= 2 * config.ring_y ||
        throw(ArgumentError("population exceeds the two-lane torus capacity"))

    params = T.ModelParams(
        δ = 0.2,
        ϵ = config.error_rate,
        init_agents = config.population,
        K = 10.0,
        lookahead = config.lookahead,
        ring_x = 2,
        ring_y = config.ring_y,
    )
    weights = T.Weights(
        wₛ = 0.5,
        wₒ = 0.5,
        wₐ = 0.5,
        wₕ = habit ? config.habit_weight : 0.0,
    )
    model = S.init_model(params, weights; seed = seed, timing = timing)
    Agents.step!(model, config.steps)
    return summarize_run(model, config, seed, timing, habit)
end

function run_experiment(config = ExperimentConfig(); progress = true)
    rows = NamedTuple[]
    conditions = (
        (S.SequentialActivation(), false),
        (S.SequentialActivation(), true),
        (S.SimultaneousActivation(), false),
        (S.SimultaneousActivation(), true),
    )
    total = length(config.seeds) * length(conditions)
    completed = 0
    for seed in config.seeds, (timing, habit) in conditions
        push!(rows, run_condition(config, seed, timing, habit))
        completed += 1
        progress && println("completed $completed/$total: seed=$seed, timing=$(timing_name(timing)), habit=$habit")
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
        timing = parts[2],
        habit = parse(Bool, parts[3]),
        steps = parse(Int, parts[4]),
        burn_in = parse(Int, parts[5]),
        population = parse(Int, parts[6]),
        ring_y = parse(Int, parts[7]),
        lookahead = parse(Int, parts[8]),
        error_rate = parse(Float64, parts[9]),
        habit_weight = parse(Float64, parts[10]),
        encounters = parse(Int, parts[11]),
        failures = parse(Int, parts[12]),
        encounter_rate = parse(Float64, parts[13]),
        failure_rate = parse(Float64, parts[14]),
        compatibility_rate = parse(Float64, parts[15]),
        precoordination_rate = parse(Float64, parts[16]),
        disposition_alignment = parse(Float64, parts[17]),
        disposition_predictability = parse(Float64, parts[18]),
        death_rate = parse(Float64, parts[19]),
        persistence = parse(Float64, parts[20]),
        convention_strength = parse(Float64, parts[21]),
        mean_abs_habitus = parse(Float64, parts[22]),
    )
end

function read_results(path)
    lines = readlines(path)
    isempty(lines) && error("empty result file: $path")
    Symbol.(split(first(lines), ',')) == collect(RESULT_COLUMNS) ||
        error("unexpected result schema in $path")
    return [parse_result(split(line, ',')) for line in Iterators.drop(lines, 1) if !isempty(line)]
end

function validate_results(rows)
    isempty(rows) && error("no experiment rows")
    seeds = sort!(unique(row.seed for row in rows))
    for seed in seeds, timing in ("sequential", "simultaneous"), habit in (false, true)
        matches = count(
            row -> row.seed == seed && row.timing == timing && row.habit == habit,
            rows,
        )
        matches == 1 || error("expected one row for seed=$seed, timing=$timing, habit=$habit")
    end
    all(row -> row.encounters >= row.failures >= 0, rows) ||
        error("failure counts must be bounded by encounter counts")
    all(row -> isapprox(row.failure_rate + row.compatibility_rate, 1.0; atol = 1e-12), rows) ||
        error("encounter compatibility and failure rates must be complementary")
    all(row -> 0 <= row.death_rate <= 1, rows) || error("invalid death rate")
    return seeds
end

function select_value(rows, seed, timing, habit, metric)
    index = findfirst(
        row -> row.seed == seed && row.timing == timing && row.habit == habit,
        rows,
    )
    isnothing(index) && error("missing condition for seed=$seed")
    return getproperty(rows[index], metric)
end

function paired_differences(rows, metric, condition_a, condition_b)
    seeds = validate_results(rows)
    return [
        select_value(rows, seed, condition_a..., metric) -
        select_value(rows, seed, condition_b..., metric)
            for seed in seeds
    ]
end

function interaction_differences(rows, metric)
    seeds = validate_results(rows)
    return [
        (select_value(rows, seed, "simultaneous", true, metric) -
         select_value(rows, seed, "simultaneous", false, metric)) -
        (select_value(rows, seed, "sequential", true, metric) -
         select_value(rows, seed, "sequential", false, metric))
            for seed in seeds
    ]
end

function randomization_pvalue(differences; alternative, draws = 20_000, seed = 710_2026)
    values = filter(isfinite, differences)
    isempty(values) && return NaN
    observed = mean(values)
    rng = Random.Xoshiro(seed)
    more_extreme = 0
    for _ in 1:draws
        under_null = mean((rand(rng, Bool) ? value : -value) for value in values)
        more_extreme += alternative == :greater ? under_null >= observed : under_null <= observed
    end
    return (more_extreme + 1) / (draws + 1)
end

function bootstrap_interval(differences; draws = 20_000, seed = 711_2026)
    values = filter(isfinite, differences)
    isempty(values) && return (NaN, NaN)
    rng = Random.Xoshiro(seed)
    estimates = [mean(rand(rng, values, length(values))) for _ in 1:draws]
    return quantile(estimates, (0.025, 0.975))
end

function effect_summary(differences; alternative)
    values = filter(isfinite, differences)
    interval = bootstrap_interval(values)
    return (
        estimate = mean(values),
        lower = interval[1],
        upper = interval[2],
        pvalue = randomization_pvalue(values; alternative = alternative),
        replicates = length(values),
    )
end

function hypothesis_results(rows)
    return [
        (
            id = "H1",
            claim = "simultaneous timing reduces compatible joint choices without habit",
            expected = :less,
            effect = effect_summary(
                paired_differences(
                    rows,
                    :compatibility_rate,
                    ("simultaneous", false),
                    ("sequential", false),
                );
                alternative = :less,
            ),
        ),
        (
            id = "H2",
            claim = "habit increases compatible joint choices under simultaneous timing",
            expected = :greater,
            effect = effect_summary(
                paired_differences(
                    rows,
                    :compatibility_rate,
                    ("simultaneous", true),
                    ("simultaneous", false),
                );
                alternative = :greater,
            ),
        ),
        (
            id = "H3",
            claim = "habit increases pre-coordinated encounters",
            expected = :greater,
            effect = effect_summary(
                paired_differences(
                    rows,
                    :precoordination_rate,
                    ("simultaneous", true),
                    ("simultaneous", false),
                );
                alternative = :greater,
            ),
        ),
        (
            id = "H4",
            claim = "habit improves compatibility more under simultaneous than sequential timing",
            expected = :greater,
            effect = effect_summary(
                interaction_differences(rows, :compatibility_rate);
                alternative = :greater,
            ),
        ),
    ]
end

function is_supported(result; alpha = 0.05)
    effect = result.effect
    correct_sign = result.expected == :greater ? effect.estimate > 0 : effect.estimate < 0
    excludes_zero = result.expected == :greater ? effect.lower > 0 : effect.upper < 0
    return correct_sign && excludes_zero && effect.pvalue < alpha
end

export ExperimentConfig,
    RESULT_COLUMNS,
    effect_summary,
    hypothesis_results,
    is_supported,
    paired_differences,
    read_results,
    run_experiment,
    validate_results,
    write_results

end

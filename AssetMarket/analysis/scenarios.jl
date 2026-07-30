struct ScenarioSpec
    name::String
    slug::String
    description::String
    params::ModelParams
end

struct ScenarioMetrics
    mean_mispricing_pct::Float64
    mean_abs_mispricing_pct::Float64
    return_volatility_pct::Float64
    excess_kurtosis::Float64
    mean_volume::Float64
    return_acf1::Float64
    absolute_return_acf1::Float64
    technical_usage_pct::Float64
    control_usage_pct::Float64
    maximum_drawdown_pct::Float64
end

struct ScenarioRun
    scenario::ScenarioSpec
    seed::Int64
    logger::Logger
    metrics::ScenarioMetrics
end

struct ScenarioSummary
    scenario::ScenarioSpec
    means::ScenarioMetrics
    ci95::ScenarioMetrics
    minima::ScenarioMetrics
    maxima::ScenarioMetrics
    replicates::Int64
end

struct ScenarioAnalysis
    scenarios::Vector{ScenarioSpec}
    runs::Vector{ScenarioRun}
    summaries::Vector{ScenarioSummary}
    steps::Int64
    burn_in::Int64
    replicate_seeds::Vector{Int64}
end

const SCENARIO_METRIC_NAMES = fieldnames(ScenarioMetrics)

function default_scenarios()
    return [
        ScenarioSpec(
            "Selection only",
            "selection_only",
            "Existing predictors compete, but genetic rule replacement is effectively disabled.",
            complex_market_params(evolution_interval = 1.0e12),
        ),
        ScenarioSpec(
            "Slow exploration",
            "slow_exploration",
            "The paper's slow-learning regime: infrequent evolution and slow error updating.",
            slow_market_params(),
        ),
        ScenarioSpec(
            "Medium exploration",
            "medium_exploration",
            "The paper's complex regime with intermediate evolutionary and scoring rates.",
            complex_market_params(),
        ),
        ScenarioSpec(
            "Rapid exploration",
            "rapid_exploration",
            "A stress scenario with faster scoring, more frequent evolution, and stronger mutation.",
            complex_market_params(
                accuracy_rate = 1 / 30,
                evolution_interval = 100.0,
                bit_mutation_probability = 0.05,
                a_mutation_std = 0.075,
                b_mutation_std = 1.5,
            ),
        ),
    ]
end

sample_mean(values) = sum(values) / length(values)

function sample_std(values)
    length(values) == 1 && return 0.0
    center = sample_mean(values)
    return sqrt(sum((value - center)^2 for value in values) / (length(values) - 1))
end

function lag_one_autocorrelation(values)
    length(values) < 2 && return 0.0
    center = sample_mean(values)
    centered = values .- center
    denominator = sum(abs2, centered)
    denominator == 0.0 && return 0.0
    return sum(@view(centered[1:(end - 1)]) .* @view(centered[2:end])) / denominator
end

function excess_kurtosis(values)
    center = sample_mean(values)
    second_moment = sample_mean((value - center)^2 for value in values)
    second_moment == 0.0 && return 0.0
    fourth_moment = sample_mean((value - center)^4 for value in values)
    return fourth_moment / second_moment^2 - 3.0
end

function maximum_drawdown(prices)
    peak = first(prices)
    maximum_loss = 0.0
    for price in prices
        peak = max(peak, price)
        maximum_loss = max(maximum_loss, (peak - price) / peak)
    end
    return maximum_loss
end

function calculate_scenario_metrics(logger::Logger; burn_in::Integer = 250)
    isempty(logger.prices) && throw(ArgumentError("cannot analyze an empty simulation"))
    burn_in >= 0 || throw(ArgumentError("burn_in cannot be negative"))
    first_step = min(burn_in + 1, length(logger.prices))
    sample_range = first_step:length(logger.prices)
    prices = logger.prices[sample_range]
    fundamentals = logger.fundamental_prices[sample_range]
    returns = logger.log_returns[sample_range]
    mispricing = 100 .* (prices .- fundamentals) ./ fundamentals

    return ScenarioMetrics(
        sample_mean(mispricing),
        sample_mean(abs.(mispricing)),
        100 * sample_std(returns),
        excess_kurtosis(returns),
        sample_mean(logger.volumes[sample_range]),
        lag_one_autocorrelation(returns),
        lag_one_autocorrelation(abs.(returns)),
        100 * sample_mean(logger.technical_usage[sample_range]),
        100 * sample_mean(logger.control_usage[sample_range]),
        100 * maximum_drawdown(prices),
    )
end

function metric_values(runs, metric)
    return [getproperty(run.metrics, metric) for run in runs]
end

function metrics_from_function(fn, runs)
    return ScenarioMetrics((fn(metric_values(runs, metric)) for metric in SCENARIO_METRIC_NAMES)...)
end

function summarize_scenario(scenario, runs)
    return ScenarioSummary(
        scenario,
        metrics_from_function(sample_mean, runs),
        metrics_from_function(values -> 1.96 * sample_std(values) / sqrt(length(values)), runs),
        metrics_from_function(minimum, runs),
        metrics_from_function(maximum, runs),
        length(runs),
    )
end

function run_scenarios(
    scenarios::AbstractVector{<:ScenarioSpec} = default_scenarios();
    steps::Integer = 1_500,
    burn_in::Integer = 300,
    replicates::Integer = 5,
    seed::Integer = 2026,
)
    steps > 1 || throw(ArgumentError("steps must be at least 2"))
    0 <= burn_in < steps || throw(ArgumentError("burn_in must be in [0, steps)"))
    replicates > 0 || throw(ArgumentError("replicates must be positive"))
    isempty(scenarios) && throw(ArgumentError("at least one scenario is required"))

    replicate_seeds = rand(Random.Xoshiro(seed), Int64, replicates)
    runs = ScenarioRun[]
    for scenario in scenarios, replicate_seed in replicate_seeds
        logger = main(
            ModelArgs(seed = replicate_seed, params = scenario.params, steps = steps),
        )
        metrics = calculate_scenario_metrics(logger; burn_in = burn_in)
        push!(runs, ScenarioRun(scenario, replicate_seed, logger, metrics))
    end
    summaries = [
        summarize_scenario(
            scenario,
            filter(run -> run.scenario.slug == scenario.slug, runs),
        ) for scenario in scenarios
    ]
    return ScenarioAnalysis(
        collect(scenarios),
        runs,
        summaries,
        steps,
        burn_in,
        replicate_seeds,
    )
end

function write_scenario_tables(analysis::ScenarioAnalysis, output_dir)
    mkpath(output_dir)
    summary_path = joinpath(output_dir, "scenario_summary.csv")
    open(summary_path, "w") do io
        println(io, "scenario,slug,metric,mean,ci95,minimum,maximum,replicates")
        for summary in analysis.summaries, metric in SCENARIO_METRIC_NAMES
            println(
                io,
                join(
                    (
                        summary.scenario.name,
                        summary.scenario.slug,
                        metric,
                        getproperty(summary.means, metric),
                        getproperty(summary.ci95, metric),
                        getproperty(summary.minima, metric),
                        getproperty(summary.maxima, metric),
                        summary.replicates,
                    ),
                    ',',
                ),
            )
        end
    end

    run_path = joinpath(output_dir, "scenario_runs.csv")
    open(run_path, "w") do io
        println(io, "scenario,slug,seed,$(join(SCENARIO_METRIC_NAMES, ','))")
        for run in analysis.runs
            values = (getproperty(run.metrics, metric) for metric in SCENARIO_METRIC_NAMES)
            println(
                io,
                join((run.scenario.name, run.scenario.slug, run.seed, values...), ','),
            )
        end
    end
    return (summary = summary_path, runs = run_path)
end

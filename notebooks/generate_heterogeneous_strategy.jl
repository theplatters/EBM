using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using EBM
using Statistics

const T = EBM.Traffic
const STEPS = 5_000
const BURN_IN = 1_000
const POPULATION = 48
const SEEDS = 20260730:20260759
const OUTPUT_DIR = normpath(joinpath(@__DIR__, "..", "plots", "heterogeneous_strategy"))

const SCENARIOS = (
    (
        name = "Heterogeneous",
        strategy = T.HeterogeneousStrategy(
            T.DecisionAwareStrategy() => 0.50,
            T.TwoFrameNaiveStrategy() => 0.25,
            T.NaiveStrategy() => 0.25,
        ),
    ),
    (name = "Decision-aware", strategy = T.DecisionAwareStrategy()),
    (name = "Two-frame naive", strategy = T.TwoFrameNaiveStrategy()),
    (name = "Naive", strategy = T.NaiveStrategy()),
)

function run_scenario(scenario, seed)
    params = T.ModelParams(
        init_agents = POPULATION,
        ring_y = 300,
        lookahead = 20,
    )
    args = T.ModelArgs(
        seed = seed,
        params = params,
        prediction_strategy = scenario.strategy,
        steps = STEPS,
    )
    return T.main(args)
end

jobs = [(scenario_index, seed) for scenario_index in eachindex(SCENARIOS) for seed in SEEDS]
logs = Vector{T.Logger}(undef, length(jobs))
Threads.@threads for index in eachindex(jobs)
    scenario_index, seed = jobs[index]
    logs[index] = run_scenario(SCENARIOS[scenario_index], seed)
end

function replicate_row(scenario, seed, logger)
    sample = (BURN_IN + 1):STEPS
    return (
        scenario = scenario.name,
        seed = seed,
        steps = STEPS,
        burn_in = BURN_IN,
        lookahead = 20,
        final_age = last(logger.mean_age),
        mean_postburn_age = mean(@view logger.mean_age[sample]),
        replacement_rate = sum(@view logger.deaths[sample]) /
                           ((STEPS - BURN_IN) * POPULATION),
        switch_rate = mean(1 .- @view logger.stay_ratio[sample]),
        final_abs_habitus = last(logger.mean_abs_habitus),
    )
end

rows = NamedTuple[]
for (index, (scenario_index, seed)) in enumerate(jobs)
    push!(rows, replicate_row(SCENARIOS[scenario_index], seed, logs[index]))
end

mkpath(OUTPUT_DIR)
replicate_path = joinpath(OUTPUT_DIR, "replicates.csv")
open(replicate_path, "w") do io
    columns = propertynames(first(rows))
    println(io, join(columns, ','))
    for row in rows
        println(io, join((getproperty(row, column) for column in columns), ','))
    end
end

summary_path = joinpath(OUTPUT_DIR, "summary.csv")
open(summary_path, "w") do io
    println(io, "scenario,replicates,steps,burn_in,lookahead,final_age_mean,final_age_sd,mean_postburn_age_mean,mean_postburn_age_sd,replacement_rate_mean,replacement_rate_sd,switch_rate_mean,switch_rate_sd,final_abs_habitus_mean,final_abs_habitus_sd")
    for scenario in SCENARIOS
        selected = filter(row -> row.scenario == scenario.name, rows)
        values(metric) = getproperty.(selected, metric)
        println(
            io,
            join(
                (
                    scenario.name,
                    length(selected),
                    STEPS,
                    BURN_IN,
                    20,
                    mean(values(:final_age)),
                    std(values(:final_age)),
                    mean(values(:mean_postburn_age)),
                    std(values(:mean_postburn_age)),
                    mean(values(:replacement_rate)),
                    std(values(:replacement_rate)),
                    mean(values(:switch_rate)),
                    std(values(:switch_rate)),
                    mean(values(:final_abs_habitus)),
                    std(values(:final_abs_habitus)),
                ),
                ',',
            ),
        )
    end
end

colors = Makie.wong_colors()[1:length(SCENARIOS)]
metrics = (
    (:final_age, "Final mean car age"),
    (:mean_postburn_age, "Mean post-burn-in car age"),
    (:replacement_rate, "Post-burn-in replacements per car-step"),
    (:switch_rate, "Post-burn-in lane-switch rate"),
)
comparison = Figure(size = (1350, 800), fontsize = 15)
for (panel, (metric, ylabel)) in enumerate(metrics)
    axis = Axis(
        comparison[(panel - 1) ÷ 2 + 1, (panel - 1) % 2 + 1];
        ylabel = ylabel,
        xticks = (1:length(SCENARIOS), collect(getproperty.(SCENARIOS, :name))),
    )
    for (scenario_index, scenario) in enumerate(SCENARIOS)
        values = [getproperty(row, metric) for row in rows if row.scenario == scenario.name]
        scatter!(axis, fill(scenario_index, length(values)), values; color = (colors[scenario_index], 0.28))
        scatter!(axis, [scenario_index], [mean(values)]; color = colors[scenario_index], markersize = 14)
        errorbars!(axis, [scenario_index], [mean(values)], [std(values)], [std(values)]; color = :black, whiskerwidth = 8)
    end
end
Label(
    comparison[0, :],
    "Heterogeneous forecast policies — mean ± 1 SD across $(length(SEEDS)) paired runs";
    fontsize = 21,
    font = :bold,
)
save(joinpath(OUTPUT_DIR, "heterogeneous_results.png"), comparison; px_per_unit = 2)

function series_summary(scenario_index, field; cumulative = false, invert = false)
    selected = [
        logs[(scenario_index - 1) * length(SEEDS) + seed_index]
            for seed_index in eachindex(SEEDS)
    ]
    matrix = reduce(hcat, [
        begin
            values = Float64.(getproperty(logger, field))
            cumulative && (values = cumsum(values) ./ POPULATION)
            invert && (values = 1 .- values)
            values
        end
            for logger in selected
    ])
    return vec(mean(matrix; dims = 2)), vec(std(matrix; dims = 2))
end

dynamics = Figure(size = (1350, 800), fontsize = 15)
dynamic_metrics = (
    (:mean_age, false, false, "Mean car age"),
    (:deaths, true, false, "Cumulative replacements per initial car"),
    (:stay_ratio, false, true, "Lane-switch rate"),
    (:mean_abs_habitus, false, false, "Mean |habitus|"),
)
for (panel, (field, cumulative, invert, ylabel)) in enumerate(dynamic_metrics)
    axis = Axis(
        dynamics[(panel - 1) ÷ 2 + 1, (panel - 1) % 2 + 1];
        xlabel = "tick",
        ylabel = ylabel,
    )
    for (scenario_index, scenario) in enumerate(SCENARIOS)
        center, deviation = series_summary(
            scenario_index,
            field;
            cumulative = cumulative,
            invert = invert,
        )
        band!(axis, 1:STEPS, center .- deviation, center .+ deviation; color = (colors[scenario_index], 0.12))
        lines!(axis, 1:STEPS, center; color = colors[scenario_index], linewidth = 2, label = scenario.name)
    end
    vlines!(axis, [BURN_IN]; color = (:black, 0.3), linestyle = :dash)
    panel == 1 && axislegend(axis; position = :lt, framevisible = false)
end
Label(
    dynamics[0, :],
    "Heterogeneous forecast-policy dynamics — mean ± 1 SD";
    fontsize = 21,
    font = :bold,
)
save(joinpath(OUTPUT_DIR, "heterogeneous_dynamics.png"), dynamics; px_per_unit = 2)

println("wrote $replicate_path")
println("wrote $summary_path")
println("wrote heterogeneous_results.png and heterogeneous_dynamics.png")

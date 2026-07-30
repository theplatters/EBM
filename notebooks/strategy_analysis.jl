using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using EBM
using Random
using Statistics

const MASTER_SEED = 42
const REPLICATES = 100
const STEPS = 100
const DENSITIES = [10, 20, 40, 60, 80]
const BASELINE_CARS = 40
const OUTPUT_DIR = normpath(joinpath(@__DIR__, "..", "plots", "strategy_analysis"))

const STRATEGIES = [
    (name = "Per-entity habitus", short = "Per-entity", strategy = Traffic.PerEntityHabitusStrategy()),
    (name = "Mean habitus", short = "Mean", strategy = Traffic.MeanHabitusStrategy()),
    (name = "Random", short = "Random", strategy = Traffic.RandomStrategy()),
    (name = "Naive", short = "Naive", strategy = Traffic.NaiveStrategy()),
    (name = "Two-frame Naive", short = "Two-frame", strategy = Traffic.TwoFrameNaiveStrategy()),
    (name = "Decision-aware", short = "Decision", strategy = Traffic.DecisionAwareStrategy()),
    (name = "Unsure", short = "Unsure", strategy = Traffic.UnsureStrategy()),
    (name = "Switch", short = "Switch", strategy = Traffic.SwitchStrategy()),
]

function mean_ci(values)
    center = mean(values)
    half_width = 1.96 * std(values) / sqrt(length(values))
    return center, half_width
end

function summarize_series(matrix)
    center = vec(mean(matrix; dims = 2))
    half_width = 1.96 .* vec(std(matrix; dims = 2)) ./ sqrt(size(matrix, 2))
    return center, half_width
end

function run_group(spec, initial_cars, seeds)
    params = Traffic.ModelParams(init_agents = initial_cars)
    logs = Vector{Traffic.Logger}(undef, length(seeds))

    Threads.@threads for replicate in eachindex(seeds)
        args = Traffic.ModelArgs(
            seed = seeds[replicate],
            params = params,
            prediction_strategy = spec.strategy,
            steps = STEPS,
        )
        logs[replicate] = Traffic.main(args)
    end
    return logs
end

function scalar_row(spec, initial_cars, replicate, logger)
    first_replacement = something(findfirst(>(0), logger.deaths), STEPS + 1)
    return (
        strategy = spec.name,
        short = spec.short,
        initial_cars = initial_cars,
        replicate = replicate,
        final_age = last(logger.mean_age),
        trajectory_age = mean(logger.mean_age),
        replacements = sum(logger.deaths),
        replacement_rate = sum(logger.deaths) / (initial_cars * STEPS),
        switch_rate = mean(1 .- logger.stay_ratio),
        final_abs_habitus = last(logger.mean_abs_habitus),
        first_replacement = first_replacement,
    )
end

function baseline_series(logs)
    ages = reduce(hcat, [logger.mean_age for logger in logs])
    replacements = reduce(hcat, [cumsum(logger.deaths) for logger in logs])
    switches = reduce(hcat, [1 .- logger.stay_ratio for logger in logs])
    habitus = reduce(hcat, [logger.mean_abs_habitus for logger in logs])
    return (
        age = summarize_series(ages),
        replacements = summarize_series(replacements),
        switches = summarize_series(switches),
        habitus = summarize_series(habitus),
    )
end

function density_summary(rows, strategy, initial_cars)
    selected = filter(row -> row.strategy == strategy && row.initial_cars == initial_cars, rows)
    return (
        final_age = mean_ci(getproperty.(selected, :final_age)),
        trajectory_age = mean_ci(getproperty.(selected, :trajectory_age)),
        replacements = mean_ci(getproperty.(selected, :replacements)),
        replacement_rate = mean_ci(getproperty.(selected, :replacement_rate)),
        switch_rate = mean_ci(getproperty.(selected, :switch_rate)),
        final_abs_habitus = mean_ci(getproperty.(selected, :final_abs_habitus)),
        first_replacement = mean_ci(Float64.(getproperty.(selected, :first_replacement))),
    )
end

function write_replicates(rows)
    path = joinpath(OUTPUT_DIR, "paired_replicates.csv")
    open(path, "w") do io
        println(io, "strategy,initial_cars,replicate,final_age,trajectory_age,replacements,replacement_rate,switch_rate,final_abs_habitus,first_replacement")
        for row in rows
            println(
                io,
                join(
                    (
                        row.strategy,
                        row.initial_cars,
                        row.replicate,
                        row.final_age,
                        row.trajectory_age,
                        row.replacements,
                        row.replacement_rate,
                        row.switch_rate,
                        row.final_abs_habitus,
                        row.first_replacement,
                    ),
                    ',',
                ),
            )
        end
    end
    return path
end

function write_density_summary(rows)
    path = joinpath(OUTPUT_DIR, "density_summary.csv")
    open(path, "w") do io
        println(io, "strategy,initial_cars,road_occupancy,final_age_mean,final_age_ci,replacement_rate_mean,replacement_rate_ci,switch_rate_mean,switch_rate_ci,final_abs_habitus_mean,first_replacement_mean")
        for spec in STRATEGIES, initial_cars in DENSITIES
            summary = density_summary(rows, spec.name, initial_cars)
            println(
                io,
                join(
                    (
                        spec.name,
                        initial_cars,
                        initial_cars / 200,
                        summary.final_age[1],
                        summary.final_age[2],
                        summary.replacement_rate[1],
                        summary.replacement_rate[2],
                        summary.switch_rate[1],
                        summary.switch_rate[2],
                        summary.final_abs_habitus[1],
                        summary.first_replacement[1],
                    ),
                    ',',
                ),
            )
        end
    end
    return path
end

function paired_difference(rows, strategy_a, strategy_b, initial_cars, metric)
    rows_a = sort(
        filter(row -> row.strategy == strategy_a && row.initial_cars == initial_cars, rows);
        by = row -> row.replicate,
    )
    rows_b = sort(
        filter(row -> row.strategy == strategy_b && row.initial_cars == initial_cars, rows);
        by = row -> row.replicate,
    )
    @assert getproperty.(rows_a, :replicate) == getproperty.(rows_b, :replicate)
    differences = getproperty.(rows_a, metric) .- getproperty.(rows_b, metric)
    return mean_ci(differences)
end

function write_paired_comparisons(rows)
    path = joinpath(OUTPUT_DIR, "paired_comparisons.csv")
    comparisons = [
        ("Decision-aware", "Naive"),
        ("Two-frame Naive", "Naive"),
        ("Decision-aware", "Mean habitus"),
        ("Naive", "Mean habitus"),
        ("Naive", "Per-entity habitus"),
    ]
    metrics = (:final_age, :trajectory_age, :replacements, :replacement_rate)

    open(path, "w") do io
        println(io, "strategy_a,strategy_b,initial_cars,metric,mean_difference,ci_low,ci_high")
        for (strategy_a, strategy_b) in comparisons, initial_cars in DENSITIES, metric in metrics
            center, half_width = paired_difference(rows, strategy_a, strategy_b, initial_cars, metric)
            println(
                io,
                join(
                    (
                        strategy_a,
                        strategy_b,
                        initial_cars,
                        metric,
                        center,
                        center - half_width,
                        center + half_width,
                    ),
                    ',',
                ),
            )
        end
    end
    return path
end

function plot_overview(series)
    figure = Figure(size = (1500, 1000))
    axes = [
        Axis(figure[1, 1], xlabel = "Step", ylabel = "Mean car age", title = "Survival"),
        Axis(figure[1, 2], xlabel = "Step", ylabel = "Cumulative replacements", title = "Collision cost"),
        Axis(figure[2, 1], xlabel = "Step", ylabel = "Lane-switch rate", title = "Lane stability"),
        Axis(figure[2, 2], xlabel = "Step", ylabel = "Mean |habitus|", title = "Habit formation"),
    ]
    colors = Makie.resample_cmap(:tab10, length(STRATEGIES))
    steps = 1:STEPS

    for (index, spec) in enumerate(STRATEGIES)
        values = series[spec.name]
        for (axis, field) in zip(axes, (:age, :replacements, :switches, :habitus))
            center, half_width = getproperty(values, field)
            band!(axis, steps, center .- half_width, center .+ half_width; color = (colors[index], 0.12))
            lines!(axis, steps, center; color = colors[index], linewidth = 2, label = spec.name)
        end
    end
    axislegend(axes[1]; position = :lt, framevisible = false)
    save(joinpath(OUTPUT_DIR, "strategy_overview.png"), figure)
    return figure
end

function plot_density(rows)
    figure = Figure(size = (1450, 730))
    age_axis = Axis(
        figure[1, 1],
        xlabel = "Initial cars (ring capacity: 200)",
        ylabel = "Final mean age",
        title = "Survival as traffic density increases",
        xticks = DENSITIES,
    )
    replacement_axis = Axis(
        figure[1, 2],
        xlabel = "Initial cars (ring capacity: 200)",
        ylabel = "Replacements per car-step",
        title = "Normalized collision cost",
        xticks = DENSITIES,
    )
    colors = Makie.resample_cmap(:tab10, length(STRATEGIES))

    for (index, spec) in enumerate(STRATEGIES)
        summaries = [density_summary(rows, spec.name, initial_cars) for initial_cars in DENSITIES]
        age = first.(getproperty.(summaries, :final_age))
        age_ci = last.(getproperty.(summaries, :final_age))
        replacement = first.(getproperty.(summaries, :replacement_rate))
        replacement_ci = last.(getproperty.(summaries, :replacement_rate))

        band!(age_axis, DENSITIES, age .- age_ci, age .+ age_ci; color = (colors[index], 0.12))
        lines!(age_axis, DENSITIES, age; color = colors[index], linewidth = 2, label = spec.name)
        scatter!(age_axis, DENSITIES, age; color = colors[index], markersize = 9)

        band!(replacement_axis, DENSITIES, replacement .- replacement_ci, replacement .+ replacement_ci; color = (colors[index], 0.12))
        lines!(replacement_axis, DENSITIES, replacement; color = colors[index], linewidth = 2)
        scatter!(replacement_axis, DENSITIES, replacement; color = colors[index], markersize = 9)
    end
    Legend(
        figure[2, 1:2],
        age_axis;
        orientation = :horizontal,
        nbanks = 2,
        framevisible = false,
    )
    save(joinpath(OUTPUT_DIR, "density_sensitivity.png"), figure)
    return figure
end

function plot_tradeoff(rows)
    figure = Figure(size = (1100, 650))
    axis = Axis(
        figure[1, 1],
        xlabel = "Mean lane-switch rate",
        ylabel = "Replacements per car-step",
        title = "Default-weight behavioral trade-off (40 cars)",
    )
    colors = Makie.resample_cmap(:tab10, length(STRATEGIES))

    for (index, spec) in enumerate(STRATEGIES)
        summary = density_summary(rows, spec.name, BASELINE_CARS)
        x = summary.switch_rate[1]
        y = summary.replacement_rate[1]
        scatter!(axis, [x], [y]; color = colors[index], markersize = 18, label = spec.short)
    end
    Legend(figure[1, 2], axis; framevisible = false)
    colsize!(figure.layout, 1, Relative(0.80))
    save(joinpath(OUTPUT_DIR, "strategy_tradeoff.png"), figure)
    return figure
end

function analyze_weights()
    sweep = Traffic.run_all(resolution = 5, depth = 50, seed = MASTER_SEED)
    rows = NamedTuple[]

    for spec in STRATEGIES
        results = sweep[spec.strategy]
        scores = [mean(result.logger.mean_age) for result in results]
        best_index = argmax(scores)
        best = results[best_index]
        push!(
            rows,
            (
                strategy = spec.name,
                scores = scores,
                median_score = median(scores),
                minimum_score = minimum(scores),
                maximum_score = maximum(scores),
                best_score = scores[best_index],
                best_weights = best.weights,
            ),
        )
    end

    path = joinpath(OUTPUT_DIR, "weight_sensitivity.csv")
    open(path, "w") do io
        println(io, "strategy,minimum_trajectory_age,median_trajectory_age,maximum_trajectory_age,best_ws,best_wo,best_wa,best_wh")
        for row in rows
            weights = row.best_weights
            println(
                io,
                join(
                    (
                        row.strategy,
                        row.minimum_score,
                        row.median_score,
                        row.maximum_score,
                        weights.wₛ,
                        weights.wₒ,
                        weights.wₐ,
                        weights.wₕ,
                    ),
                    ',',
                ),
            )
        end
    end

    figure = Figure(size = (1200, 650))
    axis = Axis(
        figure[1, 1],
        xlabel = "Strategy",
        ylabel = "Trajectory mean age",
        title = "Sensitivity across 35 simplex weight combinations",
        xticks = (1:length(STRATEGIES), getproperty.(STRATEGIES, :short)),
    )
    colors = Makie.resample_cmap(:tab10, length(STRATEGIES))
    for (index, row) in enumerate(rows)
        offsets = range(-0.16, 0.16; length = length(row.scores))
        scatter!(axis, index .+ offsets, row.scores; color = (:gray45, 0.35), markersize = 7)
        lines!(axis, [index - 0.19, index + 0.19], fill(row.median_score, 2); color = :black, linewidth = 4)
        scatter!(axis, [index], [row.best_score]; color = colors[index], marker = :star5, markersize = 20)
    end
    save(joinpath(OUTPUT_DIR, "weight_sensitivity.png"), figure)
    return rows
end

function print_summary(rows, weight_rows)
    println("\nBASELINE SUMMARY (40 cars, default weights)")
    for spec in STRATEGIES
        summary = density_summary(rows, spec.name, BASELINE_CARS)
        println(
            "SUMMARY|$(spec.name)|final_age=$(round(summary.final_age[1]; digits = 3))",
            "|trajectory_age=$(round(summary.trajectory_age[1]; digits = 3))",
            "|replacements=$(round(summary.replacements[1]; digits = 2))",
            "|replacement_rate=$(round(summary.replacement_rate[1]; digits = 4))",
            "|switch_rate=$(round(summary.switch_rate[1]; digits = 4))",
            "|abs_habitus=$(round(summary.final_abs_habitus[1]; digits = 4))",
            "|first_replacement=$(round(summary.first_replacement[1]; digits = 2))",
        )
    end

    println("\nDENSITY ENDPOINTS")
    for spec in STRATEGIES
        low = density_summary(rows, spec.name, first(DENSITIES))
        high = density_summary(rows, spec.name, last(DENSITIES))
        println(
            "DENSITY|$(spec.name)|age10=$(round(low.final_age[1]; digits = 3))",
            "|age80=$(round(high.final_age[1]; digits = 3))",
            "|replacement_rate10=$(round(low.replacement_rate[1]; digits = 4))",
            "|replacement_rate80=$(round(high.replacement_rate[1]; digits = 4))",
        )
    end

    println("\nWEIGHT SENSITIVITY")
    for row in weight_rows
        weights = row.best_weights
        println(
            "WEIGHTS|$(row.strategy)|min=$(round(row.minimum_score; digits = 3))",
            "|median=$(round(row.median_score; digits = 3))",
            "|max=$(round(row.maximum_score; digits = 3))",
            "|best=($(weights.wₛ),$(weights.wₒ),$(weights.wₐ),$(weights.wₕ))",
        )
    end

    println("\nPAIRED BASELINE DIFFERENCES (first strategy minus second)")
    comparisons = [
        ("Decision-aware", "Naive"),
        ("Two-frame Naive", "Naive"),
        ("Decision-aware", "Mean habitus"),
        ("Naive", "Mean habitus"),
        ("Naive", "Per-entity habitus"),
    ]
    for (strategy_a, strategy_b) in comparisons, metric in (:final_age, :trajectory_age, :replacements)
        center, half_width = paired_difference(rows, strategy_a, strategy_b, BASELINE_CARS, metric)
        println(
            "PAIRED|$(strategy_a)|$(strategy_b)|$(metric)",
            "|difference=$(round(center; digits = 3))",
            "|ci=($(round(center - half_width; digits = 3)),$(round(center + half_width; digits = 3)))",
        )
    end
end

function main()
    mkpath(OUTPUT_DIR)
    seeds = rand(Random.Xoshiro(MASTER_SEED), Int64, REPLICATES)
    rows = NamedTuple[]
    series = Dict{String, Any}()

    for initial_cars in DENSITIES
        for spec in STRATEGIES
            @info "Running paired density group" strategy = spec.name initial_cars
            logs = run_group(spec, initial_cars, seeds)
            append!(rows, [scalar_row(spec, initial_cars, replicate, logger) for (replicate, logger) in enumerate(logs)])
            initial_cars == BASELINE_CARS && (series[spec.name] = baseline_series(logs))
        end
    end

    write_replicates(rows)
    write_density_summary(rows)
    write_paired_comparisons(rows)
    plot_overview(series)
    plot_density(rows)
    plot_tradeoff(rows)
    weight_rows = analyze_weights()
    print_summary(rows, weight_rows)
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()

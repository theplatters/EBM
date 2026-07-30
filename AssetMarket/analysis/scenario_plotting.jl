const SCENARIO_COLORS = Makie.resample_cmap(:tableau_10, 4)

function scenario_runs(analysis, scenario)
    return filter(run -> run.scenario.slug == scenario.slug, analysis.runs)
end

function ensemble_mean(runs, field)
    observations = getproperty.(getproperty.(runs, :logger), field)
    return [sample_mean(values[step] for values in observations) for step in eachindex(first(observations))]
end

"""Plot replicate and ensemble-mean mispricing paths for every scenario."""
function plot_scenario_price_paths(analysis::ScenarioAnalysis; size = (1500, 950))
    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:2],
        "Price deviations under alternative learning regimes";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )
    for (index, scenario) in enumerate(analysis.scenarios)
        row = 2 + (index - 1) ÷ 2
        column = 1 + (index - 1) % 2
        axis = style_axis!(Axis(
            figure[row, column];
            xlabel = "Period",
            ylabel = "Mispricing (%)",
            title = scenario.name,
        ))
        runs = scenario_runs(analysis, scenario)
        paths = [
            100 .* (run.logger.prices .- run.logger.fundamental_prices) ./
            run.logger.fundamental_prices for run in runs
        ]
        for path in paths
            lines!(axis, path; color = (:gray45, 0.20), linewidth = 1)
        end
        mean_path = [sample_mean(path[step] for path in paths) for step in 1:analysis.steps]
        lines!(
            axis,
            mean_path;
            color = SCENARIO_COLORS[index],
            linewidth = 2.7,
            label = "Ensemble mean",
        )
        hlines!(axis, [0.0]; color = :gray25, linestyle = :dot)
        axislegend(axis; position = :rt, framevisible = false)
    end
    rowgap!(figure.layout, 18)
    colgap!(figure.layout, 22)
    return figure
end

const PLOTTED_SCENARIO_METRICS = (
    (:mean_abs_mispricing_pct, "Mean absolute mispricing", "%"),
    (:return_volatility_pct, "Return volatility", "%"),
    (:mean_volume, "Mean trading volume", "shares"),
    (:excess_kurtosis, "Return excess kurtosis", ""),
    (:absolute_return_acf1, "Absolute-return persistence", "lag-1 ACF"),
    (:technical_usage_pct, "Technical information use", "% of bits set"),
)

"""Compare ensemble metric means and 95% normal confidence intervals."""
function plot_scenario_metric_comparison(analysis::ScenarioAnalysis; size = (1500, 1050))
    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:3],
        "Post-burn-in market outcomes by learning scenario";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )
    labels = [scenario.name for scenario in analysis.scenarios]
    positions = 1:length(labels)

    for (index, (metric, title, unit)) in enumerate(PLOTTED_SCENARIO_METRICS)
        row = 2 + (index - 1) ÷ 3
        column = 1 + (index - 1) % 3
        axis = style_axis!(Axis(
            figure[row, column];
            ylabel = unit,
            title = title,
            xticks = (positions, labels),
            xticklabelrotation = π / 5,
        ))
        centers = [getproperty(summary.means, metric) for summary in analysis.summaries]
        intervals = [getproperty(summary.ci95, metric) for summary in analysis.summaries]
        barplot!(axis, positions, centers; color = SCENARIO_COLORS[1:length(positions)])
        errorbars!(axis, positions, centers, intervals; color = :gray20, whiskerwidth = 12)
        metric in (:excess_kurtosis, :absolute_return_acf1) &&
            hlines!(axis, [0.0]; color = :gray35, linestyle = :dot)
    end
    rowgap!(figure.layout, 30)
    colgap!(figure.layout, 22)
    return figure
end

"""Plot replicate-level adaptation, volatility, and mispricing outcomes."""
function plot_scenario_tradeoff(analysis::ScenarioAnalysis; size = (1200, 800))
    figure = Figure(size = size, backgroundcolor = :white)
    axis = style_axis!(Axis(
        figure[1, 1];
        xlabel = "Technical information use (% of bits set)",
        ylabel = "Return volatility (%)",
        title = "Adaptation–volatility trade-off across matched-seed runs",
    ))
    for (index, scenario) in enumerate(analysis.scenarios)
        runs = scenario_runs(analysis, scenario)
        technical = [run.metrics.technical_usage_pct for run in runs]
        volatility = [run.metrics.return_volatility_pct for run in runs]
        mispricing = [run.metrics.mean_abs_mispricing_pct for run in runs]
        markersizes = 8 .+ 0.7 .* mispricing
        scatter!(
            axis,
            technical,
            volatility;
            color = (SCENARIO_COLORS[index], 0.55),
            markersize = markersizes,
            label = scenario.name,
        )
        scatter!(
            axis,
            [sample_mean(technical)],
            [sample_mean(volatility)];
            color = SCENARIO_COLORS[index],
            marker = :star5,
            markersize = 22,
            strokecolor = :white,
            strokewidth = 1,
        )
    end
    axislegend(axis; position = :rt, framevisible = false)
    Label(
        figure[2, 1],
        "Marker size represents mean absolute mispricing; stars are scenario means.";
        fontsize = 14,
        color = :gray35,
        tellwidth = false,
    )
    return figure
end

function generate_scenario_analysis(
    scenarios::AbstractVector{<:ScenarioSpec} = default_scenarios();
    output_dir::AbstractString = normpath(joinpath(@__DIR__, "..", "scenarios")),
    steps::Integer = 1_500,
    burn_in::Integer = 300,
    replicates::Integer = 5,
    seed::Integer = 2026,
)
    analysis = run_scenarios(
        scenarios;
        steps = steps,
        burn_in = burn_in,
        replicates = replicates,
        seed = seed,
    )
    mkpath(output_dir)
    tables = write_scenario_tables(analysis, output_dir)
    paths = (
        price_paths = joinpath(output_dir, "scenario_price_paths.png"),
        metrics = joinpath(output_dir, "scenario_metrics.png"),
        tradeoff = joinpath(output_dir, "scenario_tradeoff.png"),
    )
    save(paths.price_paths, plot_scenario_price_paths(analysis))
    save(paths.metrics, plot_scenario_metric_comparison(analysis))
    save(paths.tradeoff, plot_scenario_tradeoff(analysis))
    return (analysis = analysis, tables = tables, paths = paths)
end

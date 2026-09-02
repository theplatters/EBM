const SUGARSCAPE_COLORS = Makie.wong_colors()

function require_observations(logger::Logger)
    isempty(logger.step) && throw(ArgumentError("cannot plot an empty simulation"))
    return nothing
end

function style_axis!(axis)
    axis.xgridcolor = (:gray70, 0.25)
    axis.ygridcolor = (:gray70, 0.25)
    axis.xgridstyle = :dash
    axis.ygridstyle = :dash
    return axis
end

"""
    plot_sugarscape(world)

Show the current resource landscape, citizens colored by wealth, and the current wealth
distribution.
"""
function plot_sugarscape(world; size = (1400, 650))
    landscape = Ark.get_resource(world, SugarLandscape)
    citizens = citizen_snapshot(world)
    isempty(citizens) && throw(ArgumentError("cannot plot a world without citizens"))
    params = Ark.get_resource(world, ModelParams)

    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:2],
        "Sugarscape state at period $(Ark.get_resource(world, SimulationClock).step)";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )
    landscape_axis = Axis(
        figure[2, 1];
        xlabel = "x",
        ylabel = "y",
        title = "Resource landscape and citizen wealth",
        aspect = DataAspect(),
    )
    heatmap!(
        landscape_axis,
        1:params.width,
        1:params.height,
        landscape.current;
        colormap = :YlOrBr,
        colorrange = (0, max(1, maximum(landscape.capacity))),
    )
    wealth = [citizen.sugar for citizen in citizens]
    citizen_plot = scatter!(
        landscape_axis,
        [citizen.position.x for citizen in citizens],
        [citizen.position.y for citizen in citizens];
        color = wealth,
        colormap = :viridis,
        markersize = 8,
        strokecolor = (:black, 0.45),
        strokewidth = 0.5,
    )
    infected = [citizen for citizen in citizens if citizen.infected]
    if !isempty(infected)
        scatter!(
            landscape_axis,
            [citizen.position.x for citizen in infected],
            [citizen.position.y for citizen in infected];
            marker = :xcross,
            color = :red,
            markersize = 12,
            strokewidth = 2,
            label = "Infected",
        )
        axislegend(landscape_axis; position = :rt, framevisible = false)
    end
    Colorbar(figure[2, 1, Right()], citizen_plot; label = "Citizen sugar")
    limits!(landscape_axis, 0.5, params.width + 0.5, 0.5, params.height + 0.5)

    wealth_axis = style_axis!(Axis(
        figure[2, 2];
        xlabel = "Sugar wealth",
        ylabel = "Citizens",
        title = "Wealth distribution (Gini = $(round(gini_coefficient(wealth), digits = 3)))",
    ))
    hist!(
        wealth_axis,
        wealth;
        bins = min(30, max(5, length(unique(wealth)))),
        color = (SUGARSCAPE_COLORS[2], 0.75),
        strokecolor = :white,
    )
    colgap!(figure.layout, 28)
    return figure
end

function _trait_wealth_matrix(citizens, visions, metabolisms)
    totals = zeros(Float64, length(metabolisms), length(visions))
    counts = zeros(Int64, size(totals))
    vision_index = Dict(value => index for (index, value) in enumerate(visions))
    metabolism_index = Dict(value => index for (index, value) in enumerate(metabolisms))
    for citizen in citizens
        column = vision_index[citizen.vision]
        row = metabolism_index[citizen.metabolism]
        totals[row, column] += citizen.sugar
        counts[row, column] += 1
    end
    return [counts[row, column] == 0 ? NaN : totals[row, column] / counts[row, column]
            for row in eachindex(metabolisms), column in eachindex(visions)]
end

"""
    plot_wealth_distribution(world)

Show the final wealth histogram, Lorenz curve, age–wealth relationship, and mean wealth
for each observed vision/metabolism trait combination.
"""
function plot_wealth_distribution(world; size = (1400, 950))
    citizens = citizen_snapshot(world)
    isempty(citizens) && throw(ArgumentError("cannot plot a world without citizens"))
    wealth = [citizen.sugar for citizen in citizens]
    ages = [citizen.age for citizen in citizens]
    metabolisms = sort!(unique([citizen.metabolism for citizen in citizens]))
    visions = sort!(unique([citizen.vision for citizen in citizens]))
    distribution = wealth_statistics(wealth)
    lorenz = lorenz_curve(wealth)

    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:2],
        "Sugarscape wealth distribution at period $(Ark.get_resource(world, SimulationClock).step)";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )

    histogram_axis = style_axis!(Axis(
        figure[2, 1];
        xlabel = "Sugar wealth",
        ylabel = "Citizens",
        title = "Distribution (mean $(round(distribution.mean_wealth; digits = 1)), " *
                "median $(round(distribution.median_wealth; digits = 1)))",
    ))
    hist!(
        histogram_axis,
        wealth;
        bins = min(35, max(5, length(unique(wealth)))),
        color = (SUGARSCAPE_COLORS[2], 0.78),
        strokecolor = :white,
        strokewidth = 0.5,
    )
    vlines!(
        histogram_axis,
        [distribution.mean_wealth];
        color = SUGARSCAPE_COLORS[1],
        linewidth = 2.2,
        label = "Mean",
    )
    vlines!(
        histogram_axis,
        [distribution.median_wealth];
        color = SUGARSCAPE_COLORS[6],
        linestyle = :dash,
        linewidth = 2.2,
        label = "Median",
    )
    axislegend(histogram_axis; position = :rt, framevisible = false)

    lorenz_axis = style_axis!(Axis(
        figure[2, 2];
        xlabel = "Cumulative share of citizens",
        ylabel = "Cumulative share of wealth",
        title = "Lorenz curve (Gini = $(round(distribution.gini; digits = 3)))",
        aspect = 1,
    ))
    lines!(
        lorenz_axis,
        [0.0, 1.0],
        [0.0, 1.0];
        color = :gray45,
        linestyle = :dash,
        linewidth = 1.8,
        label = "Equality",
    )
    lines!(
        lorenz_axis,
        lorenz.population_share,
        lorenz.wealth_share;
        color = SUGARSCAPE_COLORS[3],
        linewidth = 3,
        label = "Observed",
    )
    limits!(lorenz_axis, 0, 1, 0, 1)
    axislegend(lorenz_axis; position = :lt, framevisible = false)

    age_axis = style_axis!(Axis(
        figure[3, 1];
        xlabel = "Age",
        ylabel = "Sugar wealth",
        title = "Wealth over the life course (color: metabolism)",
    ))
    age_plot = scatter!(
        age_axis,
        ages,
        wealth;
        color = [citizen.metabolism for citizen in citizens],
        colormap = :viridis,
        markersize = 7,
        alpha = 0.62,
    )
    infected = [citizen for citizen in citizens if citizen.infected]
    if !isempty(infected)
        scatter!(
            age_axis,
            [citizen.age for citizen in infected],
            [citizen.sugar for citizen in infected];
            marker = :xcross,
            color = :red,
            markersize = 10,
            label = "Infected",
        )
        axislegend(age_axis; position = :rt, framevisible = false)
    end
    Colorbar(figure[3, 1, Right()], age_plot; label = "Metabolism")

    trait_axis = Axis(
        figure[3, 2];
        xlabel = "Metabolism",
        ylabel = "Vision",
        title = "Mean wealth by fixed traits",
    )
    trait_plot = heatmap!(
        trait_axis,
        metabolisms,
        visions,
        _trait_wealth_matrix(citizens, visions, metabolisms);
        colormap = :viridis,
    )
    trait_axis.xticks = metabolisms
    trait_axis.yticks = visions
    Colorbar(figure[3, 2, Right()], trait_plot; label = "Mean sugar")

    rowgap!(figure.layout, 18)
    colgap!(figure.layout, 28)
    return figure
end

"""
    plot_model_diagnostics(logger)

Plot population, wealth inequality, resource stocks, movement, conflicts, and deaths.
"""
function plot_model_diagnostics(logger::Logger; size = (1400, 950))
    require_observations(logger)
    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:2],
        "Sugarscape wealth-distribution diagnostics";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )

    wealth_axis = style_axis!(Axis(
        figure[2, 1];
        xlabel = "Period",
        ylabel = "Sugar",
        title = "Mean and median citizen wealth",
    ))
    lines!(wealth_axis, logger.step, logger.mean_wealth; linewidth = 2.5, label = "Mean")
    lines!(
        wealth_axis,
        logger.step,
        logger.median_wealth;
        linewidth = 2.5,
        label = "Median",
        color = SUGARSCAPE_COLORS[2],
    )
    axislegend(wealth_axis; framevisible = false)

    inequality_axis = style_axis!(Axis(
        figure[2, 2];
        xlabel = "Period",
        ylabel = "Gini coefficient",
        title = "Endogenous wealth inequality",
    ))
    lines!(
        inequality_axis,
        logger.step,
        logger.gini;
        linewidth = 2.5,
        color = SUGARSCAPE_COLORS[3],
    )
    ylims!(inequality_axis, 0, 1)

    stock_axis = style_axis!(Axis(
        figure[3, 1];
        xlabel = "Period",
        ylabel = "Total sugar",
        title = "Sugar held by citizens and remaining on patches",
    ))
    lines!(
        stock_axis,
        logger.step,
        logger.total_agent_sugar;
        linewidth = 2.3,
        label = "Citizens",
    )
    lines!(
        stock_axis,
        logger.step,
        logger.total_landscape_sugar;
        linewidth = 2.3,
        color = SUGARSCAPE_COLORS[4],
        label = "Landscape",
    )
    axislegend(stock_axis; framevisible = false)

    activity_axis = style_axis!(Axis(
        figure[3, 2];
        xlabel = "Period",
        ylabel = "Citizens / events",
        title = "Movement, contested destinations, and deaths",
    ))
    lines!(activity_axis, logger.step, logger.moved; linewidth = 2, label = "Moved")
    lines!(
        activity_axis,
        logger.step,
        logger.conflicts;
        linewidth = 2,
        label = "Conflicts",
        color = SUGARSCAPE_COLORS[5],
    )
    lines!(
        activity_axis,
        logger.step,
        logger.deaths;
        linewidth = 2,
        label = "Deaths",
        color = SUGARSCAPE_COLORS[6],
    )
    lines!(
        activity_axis,
        logger.step,
        logger.births;
        linewidth = 2,
        label = "Births",
        color = SUGARSCAPE_COLORS[3],
    )
    lines!(
        activity_axis,
        logger.step,
        logger.infected;
        linewidth = 2,
        label = "Infected",
        color = SUGARSCAPE_COLORS[7],
    )
    axislegend(activity_axis; framevisible = false)
    rowgap!(figure.layout, 14)
    colgap!(figure.layout, 18)
    return figure
end

"""
    plot_population_dynamics(logger)

Plot population and mean age, demographic turnover, mortality causes, and disease
prevalence with infection and recovery flows.
"""
function plot_population_dynamics(logger::Logger; size = (1400, 950))
    require_observations(logger)
    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:2],
        "Sugarscape population and health dynamics";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )

    population_axis = style_axis!(Axis(
        figure[2, 1];
        xlabel = "Period",
        ylabel = "Citizens / years",
        title = "Population and mean citizen age",
    ))
    lines!(population_axis, logger.step, logger.population; linewidth = 2.5, label = "Population")
    lines!(
        population_axis,
        logger.step,
        logger.mean_age;
        linewidth = 2.3,
        color = SUGARSCAPE_COLORS[2],
        label = "Mean age",
    )
    axislegend(population_axis; framevisible = false)

    turnover_axis = style_axis!(Axis(
        figure[2, 2];
        xlabel = "Period",
        ylabel = "Events",
        title = "Population turnover",
    ))
    lines!(turnover_axis, logger.step, logger.deaths; linewidth = 2, label = "Deaths")
    lines!(
        turnover_axis,
        logger.step,
        logger.births;
        linewidth = 2,
        color = SUGARSCAPE_COLORS[3],
        label = "Births",
    )
    lines!(
        turnover_axis,
        logger.step,
        logger.replacements;
        linewidth = 2,
        color = SUGARSCAPE_COLORS[4],
        label = "Replacements",
    )
    axislegend(turnover_axis; framevisible = false)

    mortality_axis = style_axis!(Axis(
        figure[3, 1];
        xlabel = "Period",
        ylabel = "Deaths",
        title = "Mortality by cause",
    ))
    lines!(
        mortality_axis,
        logger.step,
        logger.starvation_deaths;
        linewidth = 2.2,
        color = SUGARSCAPE_COLORS[6],
        label = "Starvation",
    )
    lines!(
        mortality_axis,
        logger.step,
        logger.old_age_deaths;
        linewidth = 2.2,
        color = SUGARSCAPE_COLORS[5],
        label = "Old age",
    )
    axislegend(mortality_axis; framevisible = false)

    disease_axis = style_axis!(Axis(
        figure[3, 2];
        xlabel = "Period",
        ylabel = "Citizens / disease events",
        title = "Disease burden and flows",
    ))
    lines!(
        disease_axis,
        logger.step,
        logger.infected;
        linewidth = 2.5,
        color = SUGARSCAPE_COLORS[7],
        label = "Infected citizens",
    )
    lines!(
        disease_axis,
        logger.step,
        logger.infections;
        linewidth = 1.8,
        color = SUGARSCAPE_COLORS[1],
        label = "Acquisitions",
    )
    lines!(
        disease_axis,
        logger.step,
        logger.recoveries;
        linewidth = 1.8,
        color = SUGARSCAPE_COLORS[3],
        label = "Clearances",
    )
    axislegend(disease_axis; framevisible = false)
    linkxaxes!(population_axis, turnover_axis, mortality_axis, disease_axis)
    rowgap!(figure.layout, 18)
    colgap!(figure.layout, 22)
    return figure
end

"""
    generate_plot_suite(args; output_dir, burn_in)

Run Sugarscape once and save the spatial state, wealth-distribution, model-diagnostic,
and population/health figures plus a TSV summary. Returns the logger, the computed
statistics, and a named tuple of output paths.
"""
function generate_plot_suite(
    args::ModelArgs = ModelArgs(steps = 250);
    output_dir::AbstractString = normpath(joinpath(@__DIR__, "..", "plots")),
    burn_in::Integer = min(50, args.steps ÷ 5),
)
    args.steps > 0 || throw(ArgumentError("plot generation requires at least one step"))
    burn_in >= 0 || throw(ArgumentError("burn_in cannot be negative"))
    mkpath(output_dir)
    world = run_model(args)
    logger = Ark.get_resource(world, Logger)
    statistics = summary_statistics(world; burn_in = burn_in)
    paths = (
        state = joinpath(output_dir, "sugarscape_state.png"),
        wealth = joinpath(output_dir, "wealth_distribution.png"),
        diagnostics = joinpath(output_dir, "model_diagnostics.png"),
        population = joinpath(output_dir, "population_dynamics.png"),
        statistics = joinpath(output_dir, "summary_statistics.tsv"),
    )
    save(paths.state, plot_sugarscape(world))
    save(paths.wealth, plot_wealth_distribution(world))
    save(paths.diagnostics, plot_model_diagnostics(logger))
    save(paths.population, plot_population_dynamics(logger))
    write_summary_statistics(paths.statistics, statistics)
    return (logger = logger, statistics = statistics, paths = paths)
end

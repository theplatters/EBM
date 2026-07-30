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

function generate_plot_suite(
    args::ModelArgs = ModelArgs(steps = 250);
    output_dir::AbstractString = normpath(joinpath(@__DIR__, "..", "plots")),
)
    args.steps > 0 || throw(ArgumentError("plot generation requires at least one step"))
    mkpath(output_dir)
    world = run_model(args)
    logger = Ark.get_resource(world, Logger)
    paths = (
        state = joinpath(output_dir, "sugarscape_state.png"),
        diagnostics = joinpath(output_dir, "model_diagnostics.png"),
    )
    save(paths.state, plot_sugarscape(world))
    save(paths.diagnostics, plot_model_diagnostics(logger))
    return (logger = logger, paths = paths)
end

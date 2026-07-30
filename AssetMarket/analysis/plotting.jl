const ASSET_MARKET_COLORS = Makie.wong_colors()

function require_observations(logger::Logger)
    isempty(logger.prices) && throw(ArgumentError("cannot plot an empty simulation"))
    return nothing
end

function rolling_mean(values::AbstractVector{<:Real}, window::Integer)
    window > 0 || throw(ArgumentError("rolling window must be positive"))
    result = Vector{Float64}(undef, length(values))
    running_total = 0.0
    for i in eachindex(values)
        running_total += values[i]
        i > window && (running_total -= values[i - window])
        result[i] = running_total / min(i, window)
    end
    return result
end

function rolling_std(values::AbstractVector{<:Real}, window::Integer)
    window > 1 || throw(ArgumentError("volatility window must be at least 2"))
    result = Vector{Float64}(undef, length(values))
    for i in eachindex(values)
        first_index = max(1, i - window + 1)
        observations = @view values[first_index:i]
        observation_mean = sum(observations) / length(observations)
        result[i] = sqrt(
            sum((value - observation_mean)^2 for value in observations) /
            length(observations),
        )
    end
    return result
end

function autocorrelations(values::AbstractVector{<:Real}, maximum_lag::Integer)
    maximum_lag >= 1 || throw(ArgumentError("maximum_lag must be positive"))
    sample = Float64.(values)
    sample_mean = sum(sample) / length(sample)
    centered = sample .- sample_mean
    denominator = sum(abs2, centered)
    lags = 1:min(maximum_lag, length(sample) - 1)
    isempty(lags) && return Int64[], Float64[]
    denominator == 0.0 && return collect(Int64, lags), zeros(length(lags))
    correlations = [
        sum(@view(centered[1:(end - lag)]) .* @view(centered[(lag + 1):end])) /
        denominator for lag in lags
    ]
    return collect(Int64, lags), correlations
end

function style_axis!(axis)
    axis.xgridcolor = (:gray70, 0.25)
    axis.ygridcolor = (:gray70, 0.25)
    axis.xgridstyle = :dash
    axis.ygridstyle = :dash
    return axis
end

"""
    plot_market_dynamics(logger; rolling_window = 50)

Show the market price against the homogeneous rational-expectations benchmark,
relative mispricing, and the exogenous dividend process.
"""
function plot_market_dynamics(
    logger::Logger;
    rolling_window::Integer = 50,
    size = (1500, 950),
)
    require_observations(logger)
    steps = eachindex(logger.prices)
    mispricing = 100 .* (logger.prices .- logger.fundamental_prices) ./
                 logger.fundamental_prices
    smoothed_mispricing = rolling_mean(mispricing, rolling_window)

    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1],
        "Santa Fe artificial stock market dynamics";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )

    price_axis = style_axis!(Axis(
        figure[2, 1];
        xlabel = "Period",
        ylabel = "Price",
        title = "Endogenous market price and homogeneous-expectations benchmark",
    ))
    lines!(
        price_axis,
        steps,
        logger.prices;
        color = ASSET_MARKET_COLORS[1],
        linewidth = 1.8,
        label = "Market price",
    )
    lines!(
        price_axis,
        steps,
        logger.fundamental_prices;
        color = ASSET_MARKET_COLORS[6],
        linewidth = 2,
        linestyle = :dash,
        label = "HREE price",
    )
    axislegend(price_axis; position = :rt, framevisible = false)

    deviation_axis = style_axis!(Axis(
        figure[3, 1];
        xlabel = "Period",
        ylabel = "Price deviation (%)",
        title = "Bubbles and corrections relative to fundamental value",
    ))
    lines!(
        deviation_axis,
        steps,
        mispricing;
        color = (:gray35, 0.38),
        linewidth = 1,
        label = "Period deviation",
    )
    lines!(
        deviation_axis,
        steps,
        smoothed_mispricing;
        color = ASSET_MARKET_COLORS[2],
        linewidth = 2.6,
        label = "$rolling_window-period mean",
    )
    hlines!(deviation_axis, [0.0]; color = :gray30, linestyle = :dot)
    axislegend(deviation_axis; position = :rt, framevisible = false)

    dividend_axis = style_axis!(Axis(
        figure[4, 1];
        xlabel = "Period",
        ylabel = "Dividend",
        title = "Exogenous AR(1) dividend process",
    ))
    lines!(
        dividend_axis,
        steps,
        logger.dividends;
        color = ASSET_MARKET_COLORS[3],
        linewidth = 1.6,
    )
    linkxaxes!(price_axis, deviation_axis, dividend_axis)
    rowgap!(figure.layout, 12)
    return figure
end

"""
    plot_volatility_and_volume(logger; burn_in = 250, volatility_window = 50)

Diagnose returns, clustered volatility, trading volume, return dependence, and the
relationship between market activity and absolute price changes.
"""
function plot_volatility_and_volume(
    logger::Logger;
    burn_in::Integer = 250,
    volatility_window::Integer = 50,
    maximum_lag::Integer = 30,
    size = (1500, 950),
)
    require_observations(logger)
    burn_in >= 0 || throw(ArgumentError("burn_in cannot be negative"))
    first_step = min(burn_in + 1, length(logger.prices))
    sample_range = first_step:length(logger.prices)
    returns = 100 .* logger.log_returns
    volatility = 100 .* rolling_std(logger.log_returns, volatility_window)
    lags, return_acf = autocorrelations(logger.log_returns[sample_range], maximum_lag)
    _, absolute_acf = autocorrelations(abs.(logger.log_returns[sample_range]), maximum_lag)

    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:2],
        "Returns, volatility, and trading activity";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )

    return_axis = style_axis!(Axis(
        figure[2, 1];
        xlabel = "Period",
        ylabel = "Log return (%)",
        title = "Market returns",
    ))
    lines!(return_axis, eachindex(returns), returns; color = ASSET_MARKET_COLORS[1], linewidth = 1)
    hlines!(return_axis, [0.0]; color = :gray40, linewidth = 1)

    volatility_axis = style_axis!(Axis(
        figure[2, 2];
        xlabel = "Period",
        ylabel = "Rolling volatility (%)",
        title = "$volatility_window-period return volatility",
    ))
    lines!(
        volatility_axis,
        eachindex(volatility),
        volatility;
        color = ASSET_MARKET_COLORS[6],
        linewidth = 2,
    )

    volume_axis = style_axis!(Axis(
        figure[3, 1];
        xlabel = "Period",
        ylabel = "Shares traded",
        title = "Trading volume",
    ))
    lines!(
        volume_axis,
        eachindex(logger.volumes),
        logger.volumes;
        color = ASSET_MARKET_COLORS[3],
        linewidth = 1.4,
    )

    dependence_axis = style_axis!(Axis(
        figure[3, 2];
        xlabel = "Lag",
        ylabel = "Autocorrelation",
        title = "Raw versus absolute-return dependence after burn-in",
    ))
    if !isempty(lags)
        lines!(
            dependence_axis,
            lags,
            return_acf;
            color = ASSET_MARKET_COLORS[1],
            linewidth = 2,
            label = "Returns",
        )
        scatter!(dependence_axis, lags, return_acf; color = ASSET_MARKET_COLORS[1], markersize = 6)
        lines!(
            dependence_axis,
            lags,
            absolute_acf;
            color = ASSET_MARKET_COLORS[6],
            linewidth = 2,
            label = "Absolute returns",
        )
        scatter!(dependence_axis, lags, absolute_acf; color = ASSET_MARKET_COLORS[6], markersize = 6)
    end
    hlines!(dependence_axis, [0.0]; color = :gray40, linewidth = 1)
    axislegend(dependence_axis; position = :rt, framevisible = false)
    rowgap!(figure.layout, 18)
    colgap!(figure.layout, 22)
    return figure
end

"""
    plot_belief_ecology(logger; rolling_window = 50)

Show which information categories selected rules condition on, dispersion in active
expectations and holdings, and the asynchronous genetic-replacement schedule.
"""
function plot_belief_ecology(
    logger::Logger;
    rolling_window::Integer = 50,
    size = (1500, 950),
)
    require_observations(logger)
    steps = eachindex(logger.prices)

    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1],
        "Ecology of endogenous market expectations";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )

    usage_axis = style_axis!(Axis(
        figure[2, 1];
        xlabel = "Period",
        ylabel = "Mean fraction of bits set",
        title = "Information used by selected forecasting rules",
    ))
    usage_series = (
        (logger.fundamental_usage, "Fundamental descriptors", ASSET_MARKET_COLORS[3]),
        (logger.technical_usage, "Technical descriptors", ASSET_MARKET_COLORS[1]),
        (logger.control_usage, "Control descriptors", ASSET_MARKET_COLORS[6]),
    )
    for (values, label, color) in usage_series
        lines!(
            usage_axis,
            steps,
            rolling_mean(values, rolling_window);
            color = color,
            linewidth = 2.4,
            label = label,
        )
    end
    ylims!(usage_axis, 0, 1)
    axislegend(usage_axis; position = :rt, framevisible = false)

    forecast_axis = style_axis!(Axis(
        figure[3, 1];
        xlabel = "Period",
        ylabel = "Cross-agent SD",
        title = "Heterogeneity of active forecasts and asset holdings",
    ))
    lines!(
        forecast_axis,
        steps,
        logger.forecast_std;
        color = ASSET_MARKET_COLORS[2],
        linewidth = 2,
        label = "Forecast dispersion",
    )
    lines!(
        forecast_axis,
        steps,
        logger.holding_std;
        color = ASSET_MARKET_COLORS[4],
        linewidth = 2,
        label = "Holding dispersion",
    )
    axislegend(forecast_axis; position = :rt, framevisible = false)

    evolution_axis = style_axis!(Axis(
        figure[4, 1];
        xlabel = "Period",
        ylabel = "Traders evolving rules",
        title = "Asynchronous genetic-algorithm events",
    ))
    barplot!(
        evolution_axis,
        steps,
        logger.evolution_events;
        color = (ASSET_MARKET_COLORS[5], 0.75),
        gap = 0,
    )
    linkxaxes!(usage_axis, forecast_axis, evolution_axis)
    rowgap!(figure.layout, 12)
    return figure
end

"""
    generate_plot_suite(args; output_dir, burn_in = 250)

Run the asset market and save the dynamics, volatility/volume, and belief-ecology
figures. Returns the logger and a named tuple of generated paths.
"""
function generate_plot_suite(
    args::ModelArgs = ModelArgs(steps = 2_500);
    output_dir::AbstractString = normpath(joinpath(@__DIR__, "..", "plots")),
    burn_in::Integer = min(250, args.steps ÷ 4),
    rolling_window::Integer = 50,
)
    args.steps > 1 || throw(ArgumentError("plot generation requires at least two steps"))
    mkpath(output_dir)
    logger = main(args)
    paths = (
        dynamics = joinpath(output_dir, "market_dynamics.png"),
        volatility = joinpath(output_dir, "volatility_and_volume.png"),
        beliefs = joinpath(output_dir, "belief_ecology.png"),
    )

    save(
        paths.dynamics,
        plot_market_dynamics(logger; rolling_window = rolling_window),
    )
    save(
        paths.volatility,
        plot_volatility_and_volume(
            logger;
            burn_in = burn_in,
            volatility_window = rolling_window,
        ),
    )
    save(
        paths.beliefs,
        plot_belief_ecology(logger; rolling_window = rolling_window),
    )
    return (logger = logger, paths = paths)
end

const ELFAROL_COLORS = Makie.wong_colors()

function require_observations(logger::Logger)
    isempty(logger.attendance) && throw(ArgumentError("cannot plot an empty simulation"))
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

function diagnostic_range(logger::Logger, burn_in::Integer)
    burn_in >= 0 || throw(ArgumentError("burn_in cannot be negative"))
    first_step = min(burn_in + 1, length(logger.attendance))
    return first_step:length(logger.attendance)
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

function tuple_matrix(values::AbstractVector{<:NTuple{N, Float64}}) where {N}
    matrix = Matrix{Float64}(undef, length(values), N)
    for row in eachindex(values), column in 1:N
        matrix[row, column] = values[row][column]
    end
    return matrix
end

function style_axis!(axis)
    axis.xgridcolor = (:gray70, 0.25)
    axis.ygridcolor = (:gray70, 0.25)
    axis.xgridstyle = :dash
    axis.ygridstyle = :dash
    return axis
end

"""
    plot_attendance_dynamics(logger, params; rolling_window = 25)

Plot realized attendance, its rolling mean, mean forecasts, and cross-agent forecast
dispersion. The capacity reference makes convergence and persistent fluctuations visible.
"""
function plot_attendance_dynamics(
    logger::Logger,
    params::ModelParams;
    rolling_window::Integer = 25,
    size = (1400, 850),
)
    require_observations(logger)
    steps = eachindex(logger.attendance)
    smoothed_attendance = rolling_mean(logger.attendance, rolling_window)
    lower_forecast = max.(0.0, logger.mean_forecast .- logger.forecast_std)
    upper_forecast = min.(params.population, logger.mean_forecast .+ logger.forecast_std)

    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1],
        "El Farol attendance and expectations";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )

    attendance_axis = style_axis!(Axis(
        figure[2, 1];
        xlabel = "Period",
        ylabel = "Attendance",
        title = "Realized attendance fluctuates around the congestion threshold",
    ))
    lines!(
        attendance_axis,
        steps,
        logger.attendance;
        color = (:gray35, 0.42),
        linewidth = 1.2,
        label = "Weekly attendance",
    )
    lines!(
        attendance_axis,
        steps,
        smoothed_attendance;
        color = ELFAROL_COLORS[1],
        linewidth = 3,
        label = "$rolling_window-period mean",
    )
    hlines!(
        attendance_axis,
        [params.capacity];
        color = ELFAROL_COLORS[6],
        linestyle = :dash,
        linewidth = 2.5,
        label = "Capacity ($(params.capacity))",
    )
    ylims!(attendance_axis, 0, params.population)
    axislegend(attendance_axis; position = :rt, framevisible = false)

    forecast_axis = style_axis!(Axis(
        figure[3, 1];
        xlabel = "Period",
        ylabel = "Expected attendance",
        title = "Mean active forecast and one cross-agent standard deviation",
    ))
    band!(
        forecast_axis,
        steps,
        lower_forecast,
        upper_forecast;
        color = (ELFAROL_COLORS[2], 0.20),
        label = "±1 SD across agents",
    )
    lines!(
        forecast_axis,
        steps,
        logger.mean_forecast;
        color = ELFAROL_COLORS[2],
        linewidth = 2.5,
        label = "Mean active forecast",
    )
    hlines!(
        forecast_axis,
        [params.capacity];
        color = ELFAROL_COLORS[6],
        linestyle = :dash,
        linewidth = 2.5,
        label = "Capacity",
    )
    ylims!(forecast_axis, 0, params.population)
    axislegend(forecast_axis; position = :rt, framevisible = false)
    linkxaxes!(attendance_axis, forecast_axis)
    rowgap!(figure.layout, 12)
    return figure
end

"""
    plot_coordination_diagnostics(logger, params; burn_in = 100, maximum_lag = 30)

Diagnose aggregate coordination after a burn-in period using forecast calibration,
the attendance distribution, serial dependence, and congestion-regime frequencies.
"""
function plot_coordination_diagnostics(
    logger::Logger,
    params::ModelParams;
    burn_in::Integer = 100,
    maximum_lag::Integer = 30,
    size = (1400, 950),
)
    require_observations(logger)
    sample_range = diagnostic_range(logger, burn_in)
    attendance = logger.attendance[sample_range]
    forecasts = logger.mean_forecast[sample_range]
    lags, correlations = autocorrelations(attendance, maximum_lag)

    below_capacity = count(<(params.capacity), attendance)
    at_capacity = count(==(params.capacity), attendance)
    above_capacity = count(>(params.capacity), attendance)
    regimes = 100 .* [below_capacity, at_capacity, above_capacity] ./ length(attendance)

    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:2],
        "Coordination diagnostics after $burn_in-period burn-in";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )

    calibration_axis = style_axis!(Axis(
        figure[2, 1];
        xlabel = "Mean active forecast",
        ylabel = "Realized attendance",
        title = "Aggregate forecast calibration",
        aspect = 1,
    ))
    scatter!(
        calibration_axis,
        forecasts,
        attendance;
        color = (ELFAROL_COLORS[1], 0.30),
        markersize = 7,
    )
    lines!(
        calibration_axis,
        [0, params.population],
        [0, params.population];
        color = :gray35,
        linestyle = :dot,
        linewidth = 2,
    )
    vlines!(calibration_axis, [params.capacity]; color = ELFAROL_COLORS[6], linestyle = :dash)
    hlines!(calibration_axis, [params.capacity]; color = ELFAROL_COLORS[6], linestyle = :dash)
    limits!(calibration_axis, 0, params.population, 0, params.population)

    distribution_axis = style_axis!(Axis(
        figure[2, 2];
        xlabel = "Attendance",
        ylabel = "Periods",
        title = "Post-burn-in attendance distribution",
    ))
    hist!(
        distribution_axis,
        attendance;
        bins = min(params.population + 1, 35),
        color = (ELFAROL_COLORS[3], 0.72),
        strokecolor = :white,
        strokewidth = 0.5,
    )
    vlines!(
        distribution_axis,
        [params.capacity];
        color = ELFAROL_COLORS[6],
        linestyle = :dash,
        linewidth = 2.5,
        label = "Capacity",
    )
    axislegend(distribution_axis; position = :rt, framevisible = false)

    autocorrelation_axis = style_axis!(Axis(
        figure[3, 1];
        xlabel = "Lag",
        ylabel = "Autocorrelation",
        title = "Serial structure in attendance",
    ))
    if !isempty(lags)
        barplot!(
            autocorrelation_axis,
            lags,
            correlations;
            color = ifelse.(correlations .>= 0, ELFAROL_COLORS[1], ELFAROL_COLORS[6]),
        )
    end
    hlines!(autocorrelation_axis, [0.0]; color = :gray35, linewidth = 1)

    regime_axis = style_axis!(Axis(
        figure[3, 2];
        xlabel = "Attendance regime",
        ylabel = "Share of periods (%)",
        title = "How often the bar is crowded",
        xticks = (1:3, ["Below capacity", "At capacity", "Above capacity"]),
    ))
    barplot!(
        regime_axis,
        1:3,
        regimes;
        color = [ELFAROL_COLORS[3], ELFAROL_COLORS[4], ELFAROL_COLORS[6]],
    )
    ylims!(regime_axis, 0, max(100.0, 1.12 * maximum(regimes)))
    for i in eachindex(regimes)
        text!(
            regime_axis,
            i,
            regimes[i] + 2;
            text = "$(round(regimes[i]; digits = 1))%",
            align = (:center, :bottom),
            fontsize = 14,
        )
    end
    rowgap!(figure.layout, 18)
    colgap!(figure.layout, 22)
    return figure
end

"""
    plot_predictor_ecology(logger; averaging_window = 50)

Visualize which heterogeneous predictor components are selected and how their virtual
forecast errors evolve. Family RMSE includes every predictor, whether selected or not.
"""
function plot_predictor_ecology(
    logger::Logger;
    averaging_window::Integer = 50,
    size = (1400, 900),
)
    require_observations(logger)
    averaging_window > 0 || throw(ArgumentError("averaging_window must be positive"))
    steps = collect(eachindex(logger.attendance))
    shares = tuple_matrix(logger.predictor_shares)
    rmse = tuple_matrix(logger.predictor_rmse)
    colors = Makie.resample_cmap(:tableau_10, 5)

    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:2],
        "Ecology of heterogeneous forecasting rules";
        fontsize = 24,
        font = :bold,
        tellwidth = false,
    )

    share_axis = style_axis!(Axis(
        figure[2, 1:2];
        xlabel = "Period",
        ylabel = "Share of active predictors",
        title = "Composition of the predictors actually used by agents",
    ))
    lower = zeros(length(steps))
    for family in 1:5
        upper = lower .+ shares[:, family]
        band!(
            share_axis,
            steps,
            lower,
            upper;
            color = (colors[family], 0.88),
            label = PREDICTOR_FAMILY_LABELS[family],
        )
        lower = upper
    end
    ylims!(share_axis, 0, 1)
    axislegend(
        share_axis;
        position = :rt,
        orientation = :horizontal,
        nbanks = 2,
        framevisible = false,
    )

    error_axis = style_axis!(Axis(
        figure[3, 1];
        xlabel = "Period",
        ylabel = "Virtual RMSE",
        title = "Forecast accuracy of each complete predictor family",
    ))
    for family in 1:5
        lines!(
            error_axis,
            steps,
            rmse[:, family];
            color = colors[family],
            linewidth = 2.2,
            label = PREDICTOR_FAMILY_LABELS[family],
        )
    end
    axislegend(error_axis; position = :rt, framevisible = false)

    final_axis = style_axis!(Axis(
        figure[3, 2];
        xlabel = "Predictor family",
        ylabel = "Mean active share",
        title = "Composition over the final $averaging_window periods",
        xticks = (1:5, collect(PREDICTOR_FAMILY_LABELS)),
    ))
    first_period = max(1, length(steps) - averaging_window + 1)
    final_shares = [
        sum(@view shares[first_period:end, family]) / (length(steps) - first_period + 1)
        for family in 1:5
    ]
    barplot!(final_axis, 1:5, final_shares; color = colors)
    ylims!(final_axis, 0, max(0.5, 1.15 * maximum(final_shares)))
    for family in 1:5
        text!(
            final_axis,
            family,
            final_shares[family] + 0.015;
            text = "$(round(100 * final_shares[family]; digits = 1))%",
            align = (:center, :bottom),
            fontsize = 13,
        )
    end
    rowgap!(figure.layout, 18)
    colgap!(figure.layout, 22)
    return figure
end

"""
    generate_plot_suite(args; output_dir, burn_in = 100, rolling_window = 25)

Run the model and save the attendance, coordination, and predictor-ecology figures as
PNG files. Returns the logger and a named tuple containing the generated paths.
"""
function generate_plot_suite(
    args::ModelArgs = ModelArgs(steps = 500);
    output_dir::AbstractString = normpath(joinpath(@__DIR__, "..", "plots")),
    burn_in::Integer = min(100, args.steps ÷ 4),
    rolling_window::Integer = 25,
)
    args.steps > 0 || throw(ArgumentError("plot generation requires at least one step"))
    mkpath(output_dir)
    logger = main(args)

    paths = (
        dynamics = joinpath(output_dir, "attendance_dynamics.png"),
        coordination = joinpath(output_dir, "coordination_diagnostics.png"),
        predictors = joinpath(output_dir, "predictor_ecology.png"),
    )
    save(
        paths.dynamics,
        plot_attendance_dynamics(logger, args.params; rolling_window = rolling_window),
    )
    save(
        paths.coordination,
        plot_coordination_diagnostics(logger, args.params; burn_in = burn_in),
    )
    save(paths.predictors, plot_predictor_ecology(logger))
    return (logger = logger, paths = paths)
end

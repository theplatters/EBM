mutable struct Logger
    attendance::Vector{Int64}
    mean_forecast::Vector{Float64}
    forecast_std::Vector{Float64}
    attendance_rate::Vector{Float64}
    mean_payoff::Vector{Float64}
    active_above_capacity::Vector{Float64}
    successful_decision_rate::Vector{Float64}
    predictor_shares::Vector{NTuple{5, Float64}}
    predictor_rmse::Vector{NTuple{5, Float64}}
end

Logger() = Logger(
    Int64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    NTuple{5, Float64}[],
    NTuple{5, Float64}[],
)

const PREDICTOR_FAMILY_TYPES = (
    LagPredictor,
    MeanPredictor,
    MirrorPredictor,
    TrendPredictor,
    ConstantPredictor,
)

const PREDICTOR_FAMILY_LABELS = ("Lag", "Mean", "Mirror", "Trend", "Constant")

function selected_predictor_ids(world)
    selected = BitSet()
    for (entities, predictor_ids) in Query(world, (SelectedPredictor,))
        @inbounds for i in eachindex(entities)
            push!(selected, predictor_ids[i].val)
        end
    end
    return selected
end

function predictor_family_diagnostics(world, family_type, selected, step)
    selected_count = 0
    squared_error = 0.0
    predictor_count = 0

    for (entities, predictor_ids, errors, _) in
        Query(world, (PredictorId, SquaredError, family_type))
        @inbounds for i in eachindex(entities)
            selected_count += predictor_ids[i].val in selected
            squared_error += errors[i].val
            predictor_count += 1
        end
    end

    mean_rmse = predictor_count == 0 ? NaN : sqrt(squared_error / (predictor_count * step))
    return selected_count, mean_rmse
end

function predictor_diagnostics(world, participants)
    selected = selected_predictor_ids(world)
    step = Ark.get_resource(world, SimulationClock).step
    diagnostics = map(PREDICTOR_FAMILY_TYPES) do family_type
        predictor_family_diagnostics(world, family_type, selected, step)
    end
    shares = ntuple(i -> diagnostics[i][1] / participants, 5)
    rmse = ntuple(i -> diagnostics[i][2], 5)
    return shares, rmse
end

function logger!(world)
    logger = Ark.get_resource(world, Logger)
    params = Ark.get_resource(world, ModelParams)
    attendance = Ark.get_resource(world, CurrentAttendance).val

    forecast_total = 0.0
    forecast_squared_total = 0.0
    payoff_total = 0.0
    above_capacity = 0
    successful_decisions = 0
    participants = 0
    uncrowded = attendance < params.capacity

    for (entities, forecasts, decisions, payoffs) in Query(
        world,
        (ExpectedAttendance, AttendanceDecision, CumulativePayoff),
    )
        @inbounds for i in eachindex(entities)
            forecast_total += forecasts[i].val
            forecast_squared_total += forecasts[i].val^2
            payoff_total += payoffs[i].val
            above_capacity += forecasts[i].val >= params.capacity
            successful_decisions += decisions[i].attend == uncrowded
            participants += 1
        end
    end

    mean_forecast = forecast_total / participants
    forecast_variance = forecast_squared_total / participants - mean_forecast^2
    predictor_shares, predictor_rmse = predictor_diagnostics(world, participants)

    push!(logger.attendance, attendance)
    push!(logger.mean_forecast, mean_forecast)
    push!(logger.forecast_std, sqrt(max(0.0, forecast_variance)))
    push!(logger.attendance_rate, attendance / params.population)
    push!(logger.mean_payoff, payoff_total / participants)
    push!(logger.active_above_capacity, above_capacity / participants)
    push!(logger.successful_decision_rate, successful_decisions / participants)
    push!(logger.predictor_shares, predictor_shares)
    push!(logger.predictor_rmse, predictor_rmse)
    return nothing
end

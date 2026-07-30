clamp_forecast(value::Real, population::Integer) =
    clamp(Float64(value), 0.0, Float64(population))

function evaluate_predictor(
    predictor::LagPredictor,
    history::AbstractVector{<:Real},
    population::Integer,
)
    return clamp_forecast(history[end - predictor.lag + 1], population)
end

function evaluate_predictor(
    predictor::MeanPredictor,
    history::AbstractVector{<:Real},
    population::Integer,
)
    window = min(predictor.window, length(history))
    observations = @view history[(end - window + 1):end]
    return clamp_forecast(sum(observations) / window, population)
end

function evaluate_predictor(
    predictor::MirrorPredictor,
    history::AbstractVector{<:Real},
    population::Integer,
)
    lagged_attendance = history[end - predictor.lag + 1]
    return clamp_forecast(2.0 * predictor.center - lagged_attendance, population)
end

function evaluate_predictor(
    predictor::TrendPredictor,
    history::AbstractVector{<:Real},
    population::Integer,
)
    window = min(predictor.window, length(history))
    observations = @view history[(end - window + 1):end]
    midpoint = (window + 1.0) / 2.0
    mean_attendance = sum(observations) / window
    denominator = sum((i - midpoint)^2 for i in 1:window)
    slope = sum(
        (i - midpoint) * (observations[i] - mean_attendance) for i in 1:window
    ) / denominator
    next_forecast = mean_attendance + slope * (window + 1.0 - midpoint)
    return clamp_forecast(next_forecast, population)
end

function evaluate_predictor(
    predictor::ConstantPredictor,
    history::AbstractVector{<:Real},
    population::Integer,
)
    return clamp_forecast(predictor.value, population)
end

function evaluate_predictor_family!(world, ::Type{T}) where {T}
    history = Ark.get_resource(world, AttendanceHistory).values
    population = Ark.get_resource(world, ModelParams).population

    for (entities, forecasts, predictors) in Query(world, (Forecast, T))
        @inbounds for i in eachindex(entities)
            value = evaluate_predictor(predictors[i], history, population)
            forecasts[i] = Forecast(value)
        end
    end
    return nothing
end

function evaluate_predictors!(world)
    evaluate_predictor_family!(world, LagPredictor)
    evaluate_predictor_family!(world, MeanPredictor)
    evaluate_predictor_family!(world, MirrorPredictor)
    evaluate_predictor_family!(world, TrendPredictor)
    evaluate_predictor_family!(world, ConstantPredictor)
    return nothing
end

mutable struct Logger
    prices::Vector{Float64}
    dividends::Vector{Float64}
    fundamental_prices::Vector{Float64}
    log_returns::Vector{Float64}
    volumes::Vector{Float64}
    mean_forecasts::Vector{Float64}
    forecast_std::Vector{Float64}
    holding_std::Vector{Float64}
    fundamental_usage::Vector{Float64}
    technical_usage::Vector{Float64}
    control_usage::Vector{Float64}
    evolution_events::Vector{Int64}
end

Logger() = Logger(
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Int64[],
)

function selected_information_usage(world, population)
    selected = BitSet()
    for (entities, predictor_ids) in Query(world, (SelectedPredictor,))
        @inbounds for i in eachindex(entities)
            push!(selected, predictor_ids[i].val)
        end
    end

    fundamental_bits = 0
    technical_bits = 0
    control_bits = 0
    for (entities, predictor_ids, conditions) in Query(world, (PredictorId, ConditionBits))
        @inbounds for i in eachindex(entities)
            predictor_ids[i].val in selected || continue
            bits = conditions[i].values
            fundamental_bits += count(!=(WILDCARD), bits[1:6])
            technical_bits += count(!=(WILDCARD), bits[7:10])
            control_bits += count(!=(WILDCARD), bits[11:12])
        end
    end

    return (
        fundamental_bits / (6 * population),
        technical_bits / (4 * population),
        control_bits / (2 * population),
    )
end

function logger!(world)
    logger = Ark.get_resource(world, Logger)
    state = Ark.get_resource(world, MarketState)
    params = Ark.get_resource(world, ModelParams)

    forecast_total = 0.0
    forecast_squared_total = 0.0
    holding_total = 0.0
    holding_squared_total = 0.0
    traders = 0
    for (entities, forecasts, holdings) in Query(world, (ExpectedPayoff, AssetHolding))
        @inbounds for i in eachindex(entities)
            forecast_total += forecasts[i].val
            forecast_squared_total += forecasts[i].val^2
            holding_total += holdings[i].val
            holding_squared_total += holdings[i].val^2
            traders += 1
        end
    end

    mean_forecast = forecast_total / traders
    forecast_variance = forecast_squared_total / traders - mean_forecast^2
    mean_holding = holding_total / traders
    holding_variance = holding_squared_total / traders - mean_holding^2
    fundamental_usage, technical_usage, control_usage =
        selected_information_usage(world, params.population)

    push!(logger.prices, state.price)
    push!(logger.dividends, state.dividend)
    push!(logger.fundamental_prices, state.fundamental_price)
    push!(logger.log_returns, log(state.price / state.previous_price))
    push!(logger.volumes, state.volume)
    push!(logger.mean_forecasts, mean_forecast)
    push!(logger.forecast_std, sqrt(max(0.0, forecast_variance)))
    push!(logger.holding_std, sqrt(max(0.0, holding_variance)))
    push!(logger.fundamental_usage, fundamental_usage)
    push!(logger.technical_usage, technical_usage)
    push!(logger.control_usage, control_usage)
    push!(logger.evolution_events, Ark.get_resource(world, EvolutionEvents).count)
    return nothing
end

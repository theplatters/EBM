function update_predictor_accuracy!(world)
    params = Ark.get_resource(world, ModelParams)
    state = Ark.get_resource(world, MarketState)
    realized_payoff = state.price + state.dividend

    for (entities, forecasts, errors, matched) in Query(
        world,
        (PredictorForecast, ForecastErrorVariance, Matched),
    )
        @inbounds for i in eachindex(entities)
            matched[i].val || continue
            forecast_error = realized_payoff - forecasts[i].val
            updated_error = (1.0 - params.accuracy_rate) * errors[i].val +
                            params.accuracy_rate * forecast_error^2
            errors[i] = ForecastErrorVariance(max(params.minimum_forecast_variance, updated_error))
        end
    end
    return nothing
end

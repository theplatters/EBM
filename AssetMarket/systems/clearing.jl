function clear_market!(world)
    params = Ark.get_resource(world, ModelParams)
    state = Ark.get_resource(world, MarketState)
    weighted_forecasts = 0.0
    total_precision = 0.0

    for (entities, forecasts, variances) in
        Query(world, (ExpectedPayoff, PerceivedVariance))
        @inbounds for i in eachindex(entities)
            precision = 1.0 / (params.risk_aversion * variances[i].val)
            weighted_forecasts += forecasts[i].val * precision
            total_precision += precision
        end
    end

    supply = total_share_supply(params)
    price = (weighted_forecasts - supply) /
            ((1.0 + params.interest_rate) * total_precision)
    state.previous_price = state.price
    state.price = max(params.price_floor, price)
    state.fundamental_price = fundamental_price(params, state.dividend)

    for (entities, forecasts, variances, desired) in Query(
        world,
        (ExpectedPayoff, PerceivedVariance, DesiredHolding),
    )
        @inbounds for i in eachindex(entities)
            numerator = forecasts[i].val - state.price * (1.0 + params.interest_rate)
            denominator = params.risk_aversion * variances[i].val
            desired[i] = DesiredHolding(numerator / denominator)
        end
    end
    return nothing
end

function settle_trades!(world)
    state = Ark.get_resource(world, MarketState)
    turnover = 0.0
    for (entities, desired, holdings) in Query(world, (DesiredHolding, AssetHolding))
        @inbounds for i in eachindex(entities)
            turnover += abs(desired[i].val - holdings[i].val)
            holdings[i] = AssetHolding(desired[i].val)
        end
    end
    state.volume = turnover / 2.0

    history = Ark.get_resource(world, MarketHistory)
    push!(history.prices, state.price)
    push!(history.dividends, state.dividend)
    push!(history.fundamental_prices, state.fundamental_price)
    push!(history.volumes, state.volume)
    return nothing
end

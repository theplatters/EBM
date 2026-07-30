function advance_dividend!(world)
    state = Ark.get_resource(world, MarketState)
    params = Ark.get_resource(world, ModelParams)
    innovation = sqrt(params.dividend_variance) * randn(dividend_rng(world))
    state.dividend = params.dividend_mean +
                     params.dividend_persistence * (state.dividend - params.dividend_mean) +
                     innovation
    return nothing
end

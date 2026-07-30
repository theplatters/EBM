function trailing_mean(values::AbstractVector{<:Real}, window::Integer)
    observations = @view values[(end - window + 1):end]
    return sum(observations) / window
end

function market_descriptor_values(state::MarketState, history::MarketHistory, params::ModelParams)
    fundamental_ratio = state.price * params.interest_rate / max(state.dividend, eps())
    thresholds = (0.25, 0.50, 0.75, 0.875, 1.0, 1.125)
    fundamental = ntuple(i -> fundamental_ratio > thresholds[i], 6)
    windows = (5, 10, 100, 500)
    technical = ntuple(i -> state.price > trailing_mean(history.prices, windows[i]), 4)
    return (fundamental..., technical..., true, false)
end

function update_market_descriptors!(world)
    state = Ark.get_resource(world, MarketState)
    history = Ark.get_resource(world, MarketHistory)
    params = Ark.get_resource(world, ModelParams)
    Ark.get_resource(world, MarketDescriptors).values =
        market_descriptor_values(state, history, params)
    return nothing
end

function condition_matches(condition::ConditionBits, descriptors::MarketDescriptors)
    return all(
        condition.values[i] == WILDCARD ||
        condition.values[i] == Int8(descriptors.values[i]) for i in 1:DESCRIPTOR_COUNT
    )
end

specificity(condition::ConditionBits) = count(!=(WILDCARD), condition.values)

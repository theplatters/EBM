"""
Advance the market by one period using the paper's staged timing.

Existing expectations clear the current market. The realized price-plus-dividend then
updates every previously matching predictor before market descriptors, genetic
replacement, and expectations for the next period are computed.
"""
function step!(world)
    advance_dividend!(world)
    clear_market!(world)
    settle_trades!(world)
    update_predictor_accuracy!(world)

    Ark.get_resource(world, SimulationClock).step += 1
    update_market_descriptors!(world)
    evolve_predictors!(world)
    form_expectations!(world)
    logger!(world)
    return nothing
end

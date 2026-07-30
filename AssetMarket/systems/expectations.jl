function better_predictor(error, tie_breaker, predictor_id, best, owner)
    error < best.errors[owner] && return true
    error > best.errors[owner] && return false
    tie_breaker < best.tie_breakers[owner] && return true
    tie_breaker > best.tie_breakers[owner] && return false
    return predictor_id < best.predictor_ids[owner]
end

function evaluate_predictors!(world)
    state = Ark.get_resource(world, MarketState)
    descriptors = Ark.get_resource(world, MarketDescriptors)
    best = Ark.get_resource(world, BestForecasts)
    reset!(best)

    for (
        entities,
        owners,
        predictor_ids,
        conditions,
        parameters,
        forecasts,
        errors,
        demand_variances,
        matched,
        tie_breakers,
    ) in Query(
        world,
        (
            PredictorOwner,
            PredictorId,
            ConditionBits,
            ForecastParameters,
            PredictorForecast,
            ForecastErrorVariance,
            DemandVariance,
            Matched,
            TieBreaker,
        ),
    )
        @inbounds for i in eachindex(entities)
            is_match = condition_matches(conditions[i], descriptors)
            matched[i] = Matched(is_match)
            is_match || continue

            forecast = parameters[i].a * (state.price + state.dividend) + parameters[i].b
            forecasts[i] = PredictorForecast(forecast)
            owner = owners[i].val
            predictor_id = predictor_ids[i].val
            if better_predictor(
                errors[i].val,
                tie_breakers[i].val,
                predictor_id,
                best,
                owner,
            )
                best.errors[owner] = errors[i].val
                best.tie_breakers[owner] = tie_breakers[i].val
                best.predictor_ids[owner] = predictor_id
                best.forecasts[owner] = forecast
                best.variances[owner] = demand_variances[i].val
            end
        end
    end
    return nothing
end

function select_expectations!(world)
    best = Ark.get_resource(world, BestForecasts)
    for (entities, trader_ids, selected, forecasts, variances) in Query(
        world,
        (TraderId, SelectedPredictor, ExpectedPayoff, PerceivedVariance),
    )
        @inbounds for i in eachindex(entities)
            trader = trader_ids[i].val
            best.predictor_ids[trader] != 0 ||
                error("trader $trader has no matching predictor")
            selected[i] = SelectedPredictor(best.predictor_ids[trader])
            forecasts[i] = ExpectedPayoff(best.forecasts[trader])
            variances[i] = PerceivedVariance(best.variances[trader])
        end
    end
    return nothing
end

function form_expectations!(world)
    evaluate_predictors!(world)
    select_expectations!(world)
    return nothing
end

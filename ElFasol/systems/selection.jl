function is_better_predictor(error, tie_breaker, predictor_id, best, owner)
    error < best.errors[owner] && return true
    error > best.errors[owner] && return false
    tie_breaker < best.tie_breakers[owner] && return true
    tie_breaker > best.tie_breakers[owner] && return false
    return predictor_id < best.predictor_ids[owner]
end

function select_predictors!(world)
    best = Ark.get_resource(world, BestPredictions)
    reset!(best)

    for (entities, owners, ids, forecasts, errors, tie_breakers) in Query(
        world,
        (PredictorOwner, PredictorId, Forecast, SquaredError, TieBreaker),
    )
        @inbounds for i in eachindex(entities)
            owner = owners[i].val
            predictor_id = ids[i].val
            error = errors[i].val
            tie_breaker = tie_breakers[i].val
            if is_better_predictor(error, tie_breaker, predictor_id, best, owner)
                best.errors[owner] = error
                best.tie_breakers[owner] = tie_breaker
                best.predictor_ids[owner] = predictor_id
                best.forecasts[owner] = forecasts[i].val
            end
        end
    end

    for (entities, participants, selected, expected) in Query(
        world,
        (ParticipantId, SelectedPredictor, ExpectedAttendance),
    )
        @inbounds for i in eachindex(entities)
            participant = participants[i].val
            predictor_id = best.predictor_ids[participant]
            predictor_id != 0 || error("participant $participant has no predictors")
            selected[i] = SelectedPredictor(predictor_id)
            expected[i] = ExpectedAttendance(best.forecasts[participant])
        end
    end
    return nothing
end

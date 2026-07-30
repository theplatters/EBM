"""
Advance one El Farol period.

All forecasts and decisions are completed before attendance is aggregated. Predictor
scores are updated only after the realized attendance is known, including scores for
predictors that their owners did not select.
"""
function step!(world)
    evaluate_predictors!(world)
    select_predictors!(world)
    decide_attendance!(world)
    aggregate_attendance!(world)
    update_payoffs!(world)
    update_predictor_scores!(world)
    finalize_period!(world)
    logger!(world)
    return nothing
end

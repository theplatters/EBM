function geometric_wait(rng, mean_interval::Real)
    probability = min(1.0, 1.0 / mean_interval)
    probability == 1.0 && return Int64(1)
    return max(Int64(1), ceil(Int64, log1p(-rand(rng)) / log1p(-probability)))
end

function predictor_quality(world, entity, params::ModelParams)
    condition, error = Ark.get_components(
        world,
        entity,
        (ConditionBits, ForecastErrorVariance),
    )
    return error.val + params.specificity_cost * specificity(condition)
end

function tournament_parent(world, candidates, rng, params)
    best = rand(rng, candidates)
    best_quality = predictor_quality(world, best, params)
    for _ in 2:3
        candidate = rand(rng, candidates)
        quality = predictor_quality(world, candidate, params)
        if quality < best_quality
            best = candidate
            best_quality = quality
        end
    end
    return best
end

function crossover_conditions(first, second, rng, crossover_probability)
    if rand(rng) >= crossover_probability
        return first.values
    end
    return ntuple(
        i -> rand(rng, Bool) ? first.values[i] : second.values[i],
        DESCRIPTOR_COUNT,
    )
end

function mutate_conditions(values, rng, mutation_probability)
    mutated = false
    result = ntuple(DESCRIPTOR_COUNT) do i
        if rand(rng) < mutation_probability
            mutated = true
            alternatives = values[i] == WILDCARD ? (Int8(0), Int8(1)) :
                           values[i] == 0 ? (WILDCARD, Int8(1)) : (WILDCARD, Int8(0))
            return rand(rng, alternatives)
        end
        return values[i]
    end
    return result, mutated
end

function crossover_parameters(first, second, rng, crossover_probability)
    rand(rng) >= crossover_probability && return first.a, first.b
    method = rand(rng, 1:3)
    method == 1 && return (
        rand(rng, Bool) ? first.a : second.a,
        rand(rng, Bool) ? first.b : second.b,
    )
    if method == 2
        weight = rand(rng)
        return (
            weight * first.a + (1.0 - weight) * second.a,
            weight * first.b + (1.0 - weight) * second.b,
        )
    end
    selected = rand(rng, Bool) ? first : second
    return selected.a, selected.b
end

function mean_predictor_error(world, entities)
    total = sum(
        Ark.get_components(world, entity, (ForecastErrorVariance,))[1].val
            for entity in entities
    )
    return total / length(entities)
end

function replace_predictor!(world, target, parent_a, parent_b, owner_entities, rng, params)
    condition_a, parameters_a, error_a = Ark.get_components(
        world,
        parent_a,
        (ConditionBits, ForecastParameters, ForecastErrorVariance),
    )
    condition_b, parameters_b, error_b = Ark.get_components(
        world,
        parent_b,
        (ConditionBits, ForecastParameters, ForecastErrorVariance),
    )

    condition_values = crossover_conditions(
        condition_a,
        condition_b,
        rng,
        params.crossover_probability,
    )
    condition_values, condition_mutated = mutate_conditions(
        condition_values,
        rng,
        params.bit_mutation_probability,
    )
    a, b = crossover_parameters(
        parameters_a,
        parameters_b,
        rng,
        params.crossover_probability,
    )
    a += params.a_mutation_std * randn(rng)
    b += params.b_mutation_std * randn(rng)

    error_variance = (error_a.val + error_b.val) / 2.0
    condition_mutated && (error_variance = mean_predictor_error(world, owner_entities))
    error_variance = max(params.minimum_forecast_variance, error_variance)

    Ark.set_components!(
        world,
        target,
        (
            ConditionBits(condition_values),
            ForecastParameters(a, b),
            PredictorForecast(0.0),
            ForecastErrorVariance(error_variance),
            DemandVariance(error_variance),
            Matched(false),
            DefaultPredictor(false),
            TieBreaker(rand(rng)),
        ),
    )
    return nothing
end

function refresh_demand_variances!(world, entities, params)
    for entity in entities
        error = Ark.get_components(world, entity, (ForecastErrorVariance,))[1].val
        Ark.set_components!(
            world,
            entity,
            (DemandVariance(max(params.minimum_forecast_variance, error)),),
        )
    end
    return nothing
end

function evolve_trader_predictors!(world, trader, rng, params)
    registry = Ark.get_resource(world, PredictorRegistry)
    entities = registry.entities[trader]
    candidates = filter(entities) do entity
        !Ark.get_components(world, entity, (DefaultPredictor,))[1].val
    end
    replacement_count = clamp(
        round(Int, params.replacement_fraction * length(entities)),
        1,
        length(candidates),
    )
    ranked = sort(candidates; by = entity -> predictor_quality(world, entity, params))
    survivors = ranked[1:(end - replacement_count)]
    default_entity = only(filter(entities) do entity
        Ark.get_components(world, entity, (DefaultPredictor,))[1].val
    end)
    parent_pool = [survivors; default_entity]
    targets = ranked[(end - replacement_count + 1):end]

    for target in targets
        parent_a = tournament_parent(world, parent_pool, rng, params)
        parent_b = tournament_parent(world, parent_pool, rng, params)
        replace_predictor!(world, target, parent_a, parent_b, entities, rng, params)
    end
    refresh_demand_variances!(world, entities, params)
    return nothing
end

function evolve_predictors!(world)
    params = Ark.get_resource(world, ModelParams)
    schedule = Ark.get_resource(world, EvolutionSchedule)
    events = Ark.get_resource(world, EvolutionEvents)
    step = Ark.get_resource(world, SimulationClock).step
    rng = learning_rng(world)
    events.count = 0

    for trader in 1:params.population
        step < schedule.next_steps[trader] && continue
        evolve_trader_predictors!(world, trader, rng, params)
        events.count += 1
        schedule.next_steps[trader] = step + geometric_wait(rng, params.evolution_interval)
    end
    return nothing
end

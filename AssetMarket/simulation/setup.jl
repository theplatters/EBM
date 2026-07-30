function setup_history(params::ModelParams, rng)
    prices = Vector{Float64}(undef, params.history_length)
    dividends = Vector{Float64}(undef, params.history_length)
    fundamental_prices = Vector{Float64}(undef, params.history_length)
    dividend = params.dividend_mean
    for i in eachindex(prices)
        innovation = sqrt(params.dividend_variance) * randn(rng)
        dividend = params.dividend_mean +
                   params.dividend_persistence * (dividend - params.dividend_mean) +
                   innovation
        price = fundamental_price(params, dividend)
        dividends[i] = dividend
        prices[i] = price
        fundamental_prices[i] = price
    end
    return MarketHistory(prices, dividends, fundamental_prices, zeros(params.history_length))
end

function setup_resources!(world, args::ModelArgs)
    params = args.params
    Ark.add_resource!(world, params)
    master_rng = Random.Xoshiro(args.seed)
    Ark.add_resource!(world, DividendRNG(Random.Xoshiro(rand(master_rng, UInt64))))
    Ark.add_resource!(world, LearningRNG(Random.Xoshiro(rand(master_rng, UInt64))))
    history = setup_history(params, dividend_rng(world))
    Ark.add_resource!(world, history)
    Ark.add_resource!(
        world,
        MarketState(
            last(history.prices),
            last(history.prices),
            last(history.dividends),
            last(history.fundamental_prices),
            0.0,
        ),
    )
    Ark.add_resource!(world, MarketDescriptors(ntuple(_ -> false, DESCRIPTOR_COUNT)))
    Ark.add_resource!(world, BestForecasts(params.population))
    Ark.add_resource!(world, SimulationClock(0))
    Ark.add_resource!(world, EvolutionEvents(0))
    Ark.add_resource!(world, Logger())
    return nothing
end

function spawn_traders!(world, params::ModelParams)
    for trader in 1:params.population
        Ark.new_entity!(
            world,
            (
                TraderId(trader),
                SelectedPredictor(0),
                ExpectedPayoff(0.0),
                PerceivedVariance(params.initial_forecast_variance),
                DesiredHolding(params.shares_per_trader),
                AssetHolding(params.shares_per_trader),
            ),
        )
    end
    return nothing
end

function random_condition(rng, condition_probability)
    return ntuple(DESCRIPTOR_COUNT) do _
        rand(rng) < condition_probability ? rand(rng, (Int8(0), Int8(1))) : WILDCARD
    end
end

function spawn_predictors!(world, params::ModelParams)
    rng = learning_rng(world)
    registry = [Ark.Entity[] for _ in 1:params.population]
    predictor_id = 0

    for owner in 1:params.population
        for slot in 1:params.predictors_per_trader
            predictor_id += 1
            is_default = slot == 1
            condition = is_default ? ntuple(_ -> WILDCARD, DESCRIPTOR_COUNT) :
                        random_condition(rng, params.condition_probability)
            a = rand(rng) * 0.5 + 0.7
            b = rand(rng) * 29.002 - 10.0
            entity = Ark.new_entity!(
                world,
                (
                    PredictorId(predictor_id),
                    PredictorOwner(owner),
                    ConditionBits(condition),
                    ForecastParameters(a, b),
                    PredictorForecast(0.0),
                    ForecastErrorVariance(params.initial_forecast_variance),
                    DemandVariance(params.initial_forecast_variance),
                    Matched(false),
                    DefaultPredictor(is_default),
                    TieBreaker(rand(rng)),
                ),
            )
            push!(registry[owner], entity)
        end
    end
    Ark.add_resource!(world, PredictorRegistry(registry))
    next_steps = [
        geometric_wait(rng, params.evolution_interval) for _ in 1:params.population
    ]
    Ark.add_resource!(world, EvolutionSchedule(next_steps))
    return nothing
end

function setup_world(args::ModelArgs = ModelArgs())
    validate(args)
    world = Ark.World(
        TraderId,
        SelectedPredictor,
        ExpectedPayoff,
        PerceivedVariance,
        DesiredHolding,
        AssetHolding,
        PredictorId,
        PredictorOwner,
        ConditionBits,
        ForecastParameters,
        PredictorForecast,
        ForecastErrorVariance,
        DemandVariance,
        Matched,
        DefaultPredictor,
        TieBreaker,
    )
    setup_resources!(world, args)
    spawn_traders!(world, args.params)
    spawn_predictors!(world, args.params)
    update_market_descriptors!(world)
    form_expectations!(world)
    return world
end

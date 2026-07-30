function initial_history(args::ModelArgs, rng)
    if isnothing(args.initial_history)
        return rand(rng, Int64(0):args.params.population, args.params.history_length)
    end
    return copy(args.initial_history)
end

function setup_resources!(world, args::ModelArgs)
    Ark.add_resource!(world, args.params)
    Ark.add_resource!(world, SimulationRNG(args.seed))
    history = initial_history(args, simulation_rng(world))
    Ark.add_resource!(world, AttendanceHistory(history))
    Ark.add_resource!(world, CurrentAttendance(last(history)))
    Ark.add_resource!(world, SimulationClock(0))
    Ark.add_resource!(world, BestPredictions(args.params.population))
    Ark.add_resource!(world, Logger())
    return nothing
end

function spawn_participants!(world, population::Integer)
    for participant_id in 1:population
        Ark.new_entity!(
            world,
            (
                ParticipantId(participant_id),
                ExpectedAttendance(0.0),
                AttendanceDecision(false),
                SelectedPredictor(0),
                CumulativePayoff(0.0),
            ),
        )
    end
    return nothing
end

function random_predictor(rng, params::ModelParams)
    max_lag = min(params.history_length, 10)
    max_window = min(params.history_length, 12)
    family = rand(rng, 1:5)
    family == 1 && return LagPredictor(rand(rng, Int64(1):max_lag))
    family == 2 && return MeanPredictor(rand(rng, Int64(2):max_window))
    family == 3 && return MirrorPredictor(
        params.population / 2.0,
        rand(rng, Int64(1):max_lag),
    )
    family == 4 && return TrendPredictor(rand(rng, Int64(2):max_window))
    return ConstantPredictor(rand(rng, Int64(0):params.population))
end

function spawn_predictors!(world, params::ModelParams)
    rng = simulation_rng(world)
    predictor_id = 0
    for owner in 1:params.population
        for _ in 1:params.predictors_per_agent
            predictor_id += 1
            predictor = random_predictor(rng, params)
            Ark.new_entity!(
                world,
                (
                    PredictorId(predictor_id),
                    PredictorOwner(owner),
                    Forecast(0.0),
                    SquaredError(0.0),
                    TieBreaker(rand(rng)),
                    predictor,
                ),
            )
        end
    end
    return nothing
end

function setup_world(args::ModelArgs = ModelArgs())
    validate(args)
    world = Ark.World(
        ParticipantId,
        ExpectedAttendance,
        AttendanceDecision,
        SelectedPredictor,
        CumulativePayoff,
        PredictorId,
        PredictorOwner,
        Forecast,
        SquaredError,
        TieBreaker,
        LagPredictor,
        MeanPredictor,
        MirrorPredictor,
        TrendPredictor,
        ConstantPredictor,
    )
    setup_resources!(world, args)
    spawn_participants!(world, args.params.population)
    spawn_predictors!(world, args.params)
    return world
end

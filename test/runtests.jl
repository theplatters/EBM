using EBM
using Agents
using Random
using StatsBase
using Test

snapshot_signature(snapshot) = [
    (
        car.lane, car.cell, Int(car.direction), car.age, car.habitus, car.decision,
        Int(car.strategy.kind), car.strategy.iterations, car.strategy.damping,
    )
        for car in snapshot.cars
]

function sequential_fixture(cars; params = Traffic.ModelParams(init_agents = length(cars)))
    model = StandardABM(
        Traffic.SequentialModel.Car,
        GridSpace((params.ring_x, params.ring_y));
        model_step! = Traffic.SequentialModel.sequential_step!,
        properties = (params = params,),
        rng = Random.Xoshiro(1),
    )
    sensitivities = Traffic.SequentialModel.Sensitivities(1.0, 1.0, 1.0, 1.0)
    weights = Traffic.Weights(wₛ = 0.0, wₒ = 0.0, wₐ = 0.0, wₕ = 0.0)
    for (position, direction) in cars
        add_agent!(
            position,
            model;
            lr = 0.0,
            sensitivities = sensitivities,
            habitus = 0.0,
            weights = weights,
            direction = direction,
            age = 1,
        )
    end
    return model
end

@testset "Predicted occupancy grid" begin
    ring = Traffic.Ring(2, 5)
    occupancy = Traffic.PredictedOccupancy(ring)
    cells = vec(occupancy.grid)

    @test length(unique(objectid, cells)) == length(cells)
    @test all(isempty, cells)

    params = Traffic.ModelParams(init_agents = 4, ring_y = 20)
    args = Traffic.ModelArgs(
        seed = 42,
        params = params,
        prediction_strategy = Traffic.UnsureStrategy(),
        steps = 0,
    )
    world = Traffic.setup_world(args)
    predicted = Traffic.Ark.get_resource(world, Traffic.PredictedOccupancy)

    @test sum(length, predicted.grid) == 2 * params.init_agents
end

@testset "Decision-aware occupancy forecast" begin
    strategy = Traffic.DecisionAwareStrategy(iterations = 3, damping = 0.5)
    params = Traffic.ModelParams(init_agents = 12, ring_y = 30)
    args = Traffic.ModelArgs(
        seed = 77,
        params = params,
        prediction_strategy = strategy,
        steps = 0,
    )
    world = Traffic.setup_world(args)
    predicted = Traffic.Ark.get_resource(world, Traffic.PredictedOccupancy)

    @test sum(
        sum((entry[3] for entry in cell); init = 0.0) for cell in predicted.grid
    ) ≈ params.init_agents
    @test all(
        0.0 <= entry[3] <= 1.0
            for cell in predicted.grid for entry in cell
    )
    @test_throws ArgumentError Traffic.DecisionAwareStrategy(iterations = 0)
    @test_throws ArgumentError Traffic.DecisionAwareStrategy(damping = 0.0)

    first_run = Traffic.traffic_history(
        Traffic.ModelArgs(seed = 77, params = params, prediction_strategy = strategy, steps = 10),
    )
    second_run = Traffic.traffic_history(
        Traffic.ModelArgs(seed = 77, params = params, prediction_strategy = strategy, steps = 10),
    )
    @test snapshot_signature.(first_run) == snapshot_signature.(second_run)
end

@testset "Two-frame Naive decision" begin
    strategy = Traffic.TwoFrameNaiveStrategy()
    params = Traffic.ModelParams(init_agents = 12, ring_y = 30)
    world = Traffic.setup_world(
        Traffic.ModelArgs(
            seed = 91,
            params = params,
            prediction_strategy = strategy,
            steps = 0,
        ),
    )
    Traffic.calculate_lr!(world, strategy)

    weights = Traffic.Ark.get_resource(world, Traffic.Weights)
    ring = Traffic.Ark.get_resource(world, Traffic.Ring)
    occupancy = Traffic.Ark.get_resource(world, Traffic.PredictedOccupancy).grid
    checked = 0
    for (entities, positions, directions, same, opposite, avoidance, habitgene, habitus, lr) in
        Traffic.Query(
            world,
            (
                Traffic.Position,
                Traffic.Direction,
                Traffic.SSensitvity,
                Traffic.OSensitvity,
                Traffic.Avoidance,
                Traffic.Habitgene,
                Traffic.Habitus,
                Traffic.LR,
            ),
        )
        for i in eachindex(entities)
            current = Traffic.decision_score(
                entities[i], occupancy, positions[i], directions[i], same[i].val,
                opposite[i].val, avoidance[i].val, habitgene[i].val, habitus[i].val,
                ring, params, weights,
            )
            advanced_position = Traffic.predict_position(positions[i], directions[i], ring)
            advanced = Traffic.decision_score(
                entities[i], occupancy, advanced_position, directions[i], same[i].val,
                opposite[i].val, avoidance[i].val, habitgene[i].val, habitus[i].val,
                ring, params, weights,
            )
            expected = sign(current) == sign(advanced) ? (current + advanced) / 2 : 0.0
            @test lr[i].val == expected
            checked += 1
        end
    end
    @test checked == params.init_agents

    first_run = Traffic.traffic_history(
        Traffic.ModelArgs(seed = 91, params = params, prediction_strategy = strategy, steps = 10),
    )
    second_run = Traffic.traffic_history(
        Traffic.ModelArgs(seed = 91, params = params, prediction_strategy = strategy, steps = 10),
    )
    @test snapshot_signature.(first_run) == snapshot_signature.(second_run)
end

@testset "Seeded simulation randomness" begin
    params = Traffic.ModelParams(init_agents = 20, ring_y = 40)
    args = Traffic.ModelArgs(
        seed = 2026,
        params = params,
        prediction_strategy = Traffic.RandomStrategy(),
        steps = 25,
    )

    first_run = Traffic.traffic_history(args; every = 1)
    second_run = Traffic.traffic_history(args; every = 1)
    @test snapshot_signature.(first_run) == snapshot_signature.(second_run)

    different_seed = Traffic.ModelArgs(
        seed = 2027,
        params = params,
        prediction_strategy = Traffic.RandomStrategy(),
        steps = 0,
    )
    @test snapshot_signature(first(first_run)) !=
          snapshot_signature(first(Traffic.traffic_history(different_seed)))
end

@testset "Heterogeneous per-car strategies" begin
    mixture = Traffic.HeterogeneousStrategy(
        Traffic.DecisionAwareStrategy() => 0.5,
        Traffic.TwoFrameNaiveStrategy() => 0.3,
        Traffic.NaiveStrategy() => 0.2,
    )
    params = Traffic.ModelParams(init_agents = 17, ring_y = 20)
    args = Traffic.ModelArgs(
        seed = 20260730,
        params = params,
        prediction_strategy = mixture,
        steps = 30,
    )
    world = Traffic.setup_world(args)

    initial_snapshot = Traffic.traffic_snapshot(world)
    initial_counts = countmap(car.strategy.kind for car in initial_snapshot.cars)
    initial_composition = countmap(
        (car.direction, car.strategy.kind) for car in initial_snapshot.cars
    )
    @test initial_counts == Dict(
        Traffic.DecisionAwareKind => 9,
        Traffic.TwoFrameNaiveKind => 5,
        Traffic.NaiveKind => 3,
    )

    predicted = Traffic.Ark.get_resource(world, Traffic.PredictedOccupancy)
    @test sum(
        sum((entry[3] for entry in cell); init = 0.0) for cell in predicted.grid
    ) ≈ params.init_agents

    for _ in 1:args.steps
        Traffic.step!(world, mixture)
    end
    final_snapshot = Traffic.traffic_snapshot(world)
    final_counts = countmap(car.strategy.kind for car in final_snapshot.cars)
    final_composition = countmap(
        (car.direction, car.strategy.kind) for car in final_snapshot.cars
    )
    logger = Traffic.Ark.get_resource(world, Traffic.Logger)
    @test final_counts == initial_counts
    @test final_composition == initial_composition
    @test sum(logger.deaths) > 0

    first_run = Traffic.traffic_history(args)
    second_run = Traffic.traffic_history(args)
    @test snapshot_signature.(first_run) == snapshot_signature.(second_run)

    @test_throws ArgumentError Traffic.HeterogeneousStrategy(
        Traffic.NaiveStrategy() => 0.4,
        Traffic.TwoFrameNaiveStrategy() => 0.4,
    )
    @test_throws ArgumentError Traffic.HeterogeneousStrategy(
        Traffic.NaiveStrategy() => 0.5,
        Traffic.NaiveStrategy() => 0.5,
    )
end

@testset "Newborn lifecycle accounting" begin
    args = Traffic.ModelArgs(
        seed = 42,
        params = Traffic.ModelParams(init_agents = 8, ring_y = 20),
        prediction_strategy = Traffic.NaiveStrategy(),
        steps = 0,
    )
    world = Traffic.setup_world(args)

    Traffic.update_habitus!(world)
    @test all(iszero(car.habitus) for car in Traffic.traffic_snapshot(world).cars)

    logger = Traffic.Ark.get_resource(world, Traffic.Logger)
    Traffic.log_stays!(world, logger)
    @test logger.stay_ratio == [0.0]
end

@testset "Sequential model tick semantics" begin
    sequential = Traffic.SequentialModel
    params = Traffic.ModelParams(ϵ = 0.0, init_agents = 2, ring_y = 10)

    convoy = sequential_fixture(
        [((1, 1), sequential.Clockwise), ((1, 2), sequential.Clockwise)];
        params = params,
    )
    Agents.step!(convoy, 1)
    @test sort([(agent.id, agent.pos) for agent in allagents(convoy)]) ==
          [(1, (1, 2)), (2, (1, 3))]
    @test all(agent.age == 2 for agent in allagents(convoy))
    @test all(agent.habitus ≈ 1 / (params.K + 2) for agent in allagents(convoy))

    head_on = sequential_fixture(
        [((1, 1), sequential.Clockwise), ((1, 3), sequential.Counterclockwise)];
        params = params,
    )
    Agents.step!(head_on, 1)
    @test nagents(head_on) == 2
    @test all(agent.id > 2 for agent in allagents(head_on))
    @test all(agent.age == 1 for agent in allagents(head_on))

    previous = Dict(1 => (1, 4), 2 => (2, 4))
    crossed = Dict(1 => (2, 5), 2 => (1, 5))
    @test sequential.collision_ids(previous, crossed) == Set((1, 2))
end

@testset "Matched sequential initialization" begin
    params = Traffic.ModelParams(init_agents = 12, ring_y = 30)
    seed = 2026
    ecs = Traffic.setup_world(
        Traffic.ModelArgs(
            seed = seed,
            params = params,
            prediction_strategy = Traffic.NaiveStrategy(),
            steps = 0,
        ),
    )
    sequential = Traffic.SequentialModel.init_model(params, Traffic.Weights(); seed = seed)

    ecs_state = sort([
        (car.lane, car.cell, Int(car.direction), car.age)
            for car in Traffic.traffic_snapshot(ecs).cars
    ])
    sequential_state = sort([
        (agent.pos[1], agent.pos[2], Int(agent.direction), agent.age)
            for agent in allagents(sequential)
    ])
    @test sequential_state == ecs_state

    repeated = Traffic.SequentialModel.init_model(params, Traffic.Weights(); seed = seed)
    Agents.step!(sequential, 10)
    Agents.step!(repeated, 10)
    sequential_signature(model) = sort([
        (agent.id, agent.pos, Int(agent.direction), agent.age, agent.habitus)
            for agent in allagents(model)
    ])
    @test sequential_signature(sequential) == sequential_signature(repeated)
end

include(joinpath(@__DIR__, "..", "ElFasol", "test", "runtests.jl"))
include(joinpath(@__DIR__, "..", "AssetMarket", "test", "runtests.jl"))
include(joinpath(@__DIR__, "..", "Sugarscape", "test", "runtests.jl"))

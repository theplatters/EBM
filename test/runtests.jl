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

capability_signature(snapshot) = [
    (
        car.lane, car.cell, Int(car.direction), car.age, car.habitus,
        car.decision, car.speed, car.capabilities,
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

@testset "Capability-composed speed model" begin
    @test Traffic.ModelParams().ring_y == 300
    @test Traffic.ModelParams().lookahead == 60
    @test Traffic.ModelParams().init_agents == 120
    @test Traffic.ModelArgs(prediction_strategy = Traffic.CapabilityModel()).steps == 300
    @test_throws ArgumentError Traffic.Speed(0)
    @test_throws ArgumentError Traffic.Speed(4)
    default_capabilities = Traffic.CapabilityModel()
    @test (
        default_capabilities.same_direction_share,
        default_capabilities.opposite_direction_share,
        default_capabilities.avoidance_share,
    ) == (0.75, 0.75, 0.75)
    @test default_capabilities.replacement_policy isa Traffic.EntryDrawReplacement

    model = Traffic.CapabilityModel(
        habit_share = 0.0,
        convention_share = 0.0,
    )
    params = Traffic.ModelParams(
        ϵ = 0.0,
        init_agents = 1,
        ring_y = 30,
        lookahead = 10,
    )
    args = Traffic.ModelArgs(
        seed = 2026,
        params = params,
        prediction_strategy = model,
        steps = 2,
    )
    history = Traffic.traffic_history(args)
    repeated = Traffic.traffic_history(args)

    @test capability_signature.(history) == capability_signature.(repeated)
    @test all(isnothing(car.strategy) for snapshot in history for car in snapshot.cars)
    @test all(1 <= car.speed <= 3 for snapshot in history for car in snapshot.cars)
    @test only(history[2].cars).speed == 3
    @test only(history[3].cars).speed == 3
    @test all(snapshot.cumulative_replacements >= 0 for snapshot in history)
    @test all(snapshot.treatment == :entry_draw for snapshot in history)
    @test Traffic._resolve_color_by(last(history), :auto) == :speed
    @test Traffic.plot_traffic(last(history)) isa Traffic.Figure
    @test Traffic.plot_traffic_history(history) isa Traffic.Figure
    first_moved = only(history[2].cars)
    initial = only(history[1].cars)
    @test first_moved.cell == mod1(
        initial.cell + 3 * Int(initial.direction), params.ring_y,
    )

    world = Traffic.setup_world(args)
    @test !Traffic.Ark.has_resource(world, Traffic.PredictedOccupancy)
    @test Traffic.Ark.has_resource(world, Traffic.CapabilityModel)
    @test Traffic.Ark.has_components(
        world, only(Traffic.traffic_snapshot(world).cars).entity,
        (Traffic.SpeedAdjustment, Traffic.MovementPath),
    )

    start_a = Traffic.Position(1, 1)
    start_b = Traffic.Position(1, 4)
    path_a = (
        Traffic.Position(1, 2), Traffic.Position(1, 3), Traffic.Position(1, 4),
    )
    path_b = (
        Traffic.Position(1, 3), Traffic.Position(1, 2), Traffic.Position(1, 1),
    )
    @test Traffic.paths_conflict(start_a, path_a, start_b, path_b)

    invalid_model = Traffic.CapabilityModel(max_speed = 4)
    @test_throws ArgumentError Traffic.setup_world(
        Traffic.ModelArgs(
            params = params,
            prediction_strategy = invalid_model,
            steps = 0,
        ),
    )
end

@testset "Evolutionary capability replacement" begin
    parent = Traffic.CapabilityGenome(
        0.8,
        nothing,
        1.2,
        nothing,
        Traffic.ConventionPerception(0.2, 0.05),
    )
    model = Traffic.CapabilityModel(
        replacement_policy = Traffic.EvolutionaryReplacement(
            capability_mutation_rate = 0.0,
            trait_mutation_scale = 0.0,
        ),
    )
    inherited = Traffic.inherit_capability_genome(
        parent,
        model,
        ones(4),
        model.replacement_policy,
        Random.Xoshiro(1),
    )
    @test inherited == parent

    flip_policy = Traffic.EvolutionaryReplacement(
        capability_mutation_rate = 1.0,
        trait_mutation_scale = 0.0,
    )
    mutated = Traffic.inherit_capability_genome(
        parent, model, [1.0, 1.1, 1.2, 1.3], flip_policy, Random.Xoshiro(2),
    )
    @test isnothing(mutated.same_direction)
    @test mutated.opposite_direction == 1.1
    @test isnothing(mutated.avoidance)
    @test mutated.habit == 1.3
    @test isnothing(mutated.convention)

    disabled_model = Traffic.CapabilityModel(
        habit_share = 0.0,
        convention_share = 0.0,
        replacement_policy = flip_policy,
    )
    disabled_parent = Traffic.CapabilityGenome(
        nothing, nothing, nothing, nothing, nothing,
    )
    disabled_mutation = Traffic.inherit_capability_genome(
        disabled_parent,
        disabled_model,
        ones(4),
        flip_policy,
        Random.Xoshiro(3),
    )
    @test isnothing(disabled_mutation.habit)
    @test isnothing(disabled_mutation.convention)

    @test_throws ArgumentError Traffic.validate(
        Traffic.EvolutionaryReplacement(capability_mutation_rate = -0.1),
    )
    @test_throws ArgumentError Traffic.validate(
        Traffic.EvolutionaryReplacement(trait_mutation_scale = -0.1),
    )

    evolutionary_model = Traffic.CapabilityModel(
        replacement_policy = Traffic.EvolutionaryReplacement(
            capability_mutation_rate = 0.05,
            trait_mutation_scale = 0.05,
        ),
    )
    args = Traffic.ModelArgs(
        seed = 909,
        params = Traffic.ModelParams(init_agents = 24, ring_y = 60, lookahead = 15),
        prediction_strategy = evolutionary_model,
        steps = 15,
    )
    first_run = Traffic.traffic_history(args)
    second_run = Traffic.traffic_history(args)
    @test capability_signature.(first_run) == capability_signature.(second_run)
    @test all(snapshot.treatment == :evolutionary for snapshot in first_run)
    @test length(last(first_run).cars) == args.params.init_agents
    @test last(first_run).cumulative_replacements > 0
end

@testset "Capability composition and replacement" begin
    model = Traffic.CapabilityModel(
        same_direction_share = 0.5,
        opposite_direction_share = 0.5,
        avoidance_share = 0.5,
        habit_share = 0.5,
        convention_share = 0.5,
    )
    args = Traffic.ModelArgs(
        seed = 77,
        params = Traffic.ModelParams(init_agents = 30, ring_y = 75, lookahead = 15),
        prediction_strategy = model,
        steps = 20,
    )
    world = Traffic.setup_world(args)
    initial = Traffic.traffic_snapshot(world)
    @test length(unique(car.capabilities for car in initial.cars)) > 1
    initial_directions = countmap(car.direction for car in initial.cars)

    for _ in 1:args.steps
        Traffic.step!(world, model)
    end
    final = Traffic.traffic_snapshot(world)
    logger = Traffic.Ark.get_resource(world, Traffic.Logger)
    @test length(final.cars) == args.params.init_agents
    @test countmap(car.direction for car in final.cars) == initial_directions
    @test length(logger.deaths) == args.steps
    @test all(1 <= car.speed <= 3 for car in final.cars)
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

    diagnostic_model = sequential.init_model(
        params,
        Traffic.Weights();
        seed = 2026,
        timing = sequential.SimultaneousActivation(),
    )
    Agents.step!(diagnostic_model, 3)
    diagnostics = diagnostic_model.diagnostics
    @test length(diagnostics.encounter_pairs) == 3
    @test length(diagnostics.failed_encounters) == 3
    @test all(diagnostics.failed_encounters .<= diagnostics.encounter_pairs)
    @test diagnostics.compatible_encounters .+ diagnostics.failed_encounters ==
          diagnostics.encounter_pairs

    clockwise_previous = (1, 1)
    counterclockwise_previous = (1, 3)
    actions = (false, true)
    collision_matrix = [
        sequential.pair_collides(
            clockwise_previous,
            sequential.action_position(
                clockwise_previous,
                sequential.Clockwise,
                action_a,
                diagnostic_model,
            ),
            counterclockwise_previous,
            sequential.action_position(
                counterclockwise_previous,
                sequential.Counterclockwise,
                action_b,
                diagnostic_model,
            ),
        )
            for action_a in actions, action_b in actions
    ]
    @test collision_matrix == Bool[0 1; 1 0]
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

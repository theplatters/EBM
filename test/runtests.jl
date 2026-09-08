using EBM
using Agents
using Random
using StatsBase
using Test

if !isdefined(Main, :SocialHabitExperiments)
    include(joinpath(@__DIR__, "..", "notebooks", "social_habit_common.jl"))
end
using .SocialHabitExperiments

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
        car.social_habitus, car.decision, car.speed, car.risk_aversion,
        car.capabilities,
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

@testset "Opt-in reactive capability shutdown" begin
    params = Traffic.ModelParams(init_agents = 1, ring_y = 20, lookahead = 5)

    function reactive_score(age; avoidance_disable_age = nothing)
        model = Traffic.CapabilityModel(
            same_direction_share = 1.0,
            opposite_direction_share = 1.0,
            avoidance_share = 1.0,
            avoidance_disable_age = avoidance_disable_age,
        )
        world = Traffic.setup_world(Traffic.ModelArgs(
            seed = 7,
            params = params,
            prediction_strategy = model,
            steps = 0,
        ))
        for (_, observations, same, opposite, avoidance, scores, steps) in Traffic.Query(
                world,
                (Traffic.LocalObservation, Traffic.SameDirectionResponse,
                 Traffic.OppositeDirectionResponse, Traffic.NearFieldAvoidance,
                 Traffic.LaneScore, Traffic.Step),
            )
            observations[1] = Traffic.LocalObservation(1.0, 0.0, 0.0, 2.0, 0.0, 0)
            same[1] = Traffic.SameDirectionResponse(1.0)
            opposite[1] = Traffic.OppositeDirectionResponse(1.0)
            avoidance[1] = Traffic.NearFieldAvoidance(1.0)
            scores[1] = Traffic.LaneScore(0.0)
            steps[1] = Traffic.Step(age)
        end
        Traffic.add_same_direction_response!(world)
        Traffic.add_opposite_direction_response!(world)
        Traffic.add_near_field_avoidance!(world)
        return only(
            score.value
                for (_, scores) in Traffic.Query(world, (Traffic.LaneScore,))
                for score in scores
        )
    end

    # The default and an explicit `nothing` are both the historical behavior.
    @test reactive_score(50) == reactive_score(50; avoidance_disable_age = nothing)
    @test reactive_score(50) == 2.0
    @test reactive_score(49; avoidance_disable_age = 50) == 2.0
    @test reactive_score(50; avoidance_disable_age = 50) == 0.0

    habit_model = Traffic.CapabilityModel(
        same_direction_share = 1.0,
        opposite_direction_share = 1.0,
        avoidance_share = 1.0,
        habit_share = 1.0,
        convention_share = 1.0,
        social_habit_share = 1.0,
        avoidance_disable_age = 50,
    )
    habit_world = Traffic.setup_world(Traffic.ModelArgs(
        seed = 8,
        params = params,
        prediction_strategy = habit_model,
        steps = 0,
    ))
    for (_, values) in Traffic.Query(habit_world, (Traffic.HabitFormation,))
        values[1] = Traffic.HabitFormation(1.0)
    end
    for (_, values) in Traffic.Query(habit_world, (Traffic.Habitus,))
        values[1] = Traffic.Habitus(1.0)
    end
    for (_, values) in Traffic.Query(habit_world, (Traffic.PerceivedConvention,))
        values[1] = Traffic.PerceivedConvention(1.0, 1.0)
    end
    for (_, values) in Traffic.Query(habit_world, (Traffic.SocialHabitFormation,))
        values[1] = Traffic.SocialHabitFormation(1.0, 0.2, 0.0)
    end
    for (_, values) in Traffic.Query(habit_world, (Traffic.SocialHabitus,))
        values[1] = Traffic.SocialHabitus(1.0)
    end
    for (_, scores) in Traffic.Query(habit_world, (Traffic.LaneScore,))
        scores[1] = Traffic.LaneScore(0.0)
    end
    for (_, steps) in Traffic.Query(habit_world, (Traffic.Step,))
        steps[1] = Traffic.Step(50)
    end
    Traffic.add_same_direction_response!(habit_world)
    Traffic.add_opposite_direction_response!(habit_world)
    Traffic.add_near_field_avoidance!(habit_world)
    Traffic.add_habit_response!(habit_world)
    Traffic.add_convention_response!(habit_world)
    Traffic.add_social_habit_response!(habit_world)
    @test only(
        score.value
            for (_, scores) in Traffic.Query(habit_world, (Traffic.LaneScore,))
            for score in scores
    ) == 1.5

    @test_throws ArgumentError Traffic.validate(
        Traffic.CapabilityModel(avoidance_disable_age = 0),
    )
    @test_throws ArgumentError Traffic.validate(
        Traffic.CapabilityModel(avoidance_disable_age = -1),
    )
    @test_throws ArgumentError capability_model(
        ExperimentConfig(), :habit; avoidance_disable_age = 0,
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
    @test default_capabilities.social_habit_share == 0.0

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
    @test_throws ArgumentError Traffic.RiskAversion(-0.01)
    @test_throws ArgumentError Traffic.RiskAversion(1.01)
end

@testset "Capability risk and speed decisions" begin
    function speed_fixture(risk; dangerous = true, cars = 2)
        model = Traffic.CapabilityModel(max_speed = 3)
        world = Traffic.setup_world(Traffic.ModelArgs(
            seed = 31415,
            params = Traffic.ModelParams(init_agents = cars, ring_y = 20, lookahead = 5),
            prediction_strategy = model, steps = 0,
        ))
        subject = first(Traffic.traffic_snapshot(world).cars).entity
        for (es, positions, directions, speeds, controls, risks, lanes) in Traffic.Query(
                world, (Traffic.Position, Traffic.Direction, Traffic.Speed,
                        Traffic.SpeedAdjustment, Traffic.RiskAversion, Traffic.LaneProposal))
            for i in eachindex(es)
                if es[i] == subject
                    positions[i] = Traffic.Position(1, 1)
                    directions[i] = Traffic.Clockwise
                    speeds[i] = Traffic.Speed(3)
                    controls[i] = Traffic.SpeedAdjustment(3)
                    risks[i] = Traffic.RiskAversion(risk)
                    lanes[i] = Traffic.LaneProposal(1)
                elseif dangerous
                    positions[i] = Traffic.Position(1, 5)
                    directions[i] = Traffic.Counterclockwise
                    speeds[i] = Traffic.Speed(1)
                    controls[i] = Traffic.SpeedAdjustment(3)
                    lanes[i] = Traffic.LaneProposal(2)
                end
            end
        end
        return world, subject
    end

    world, subject = speed_fixture(1.0)
    ring = Traffic.Ark.get_resource(world, Traffic.Ring)
    subject_position, subject_direction = Traffic.Ark.get_components(
        world, subject, (Traffic.Position, Traffic.Direction),
    )
    motions = Traffic.observed_motions(world)
    # Speed 3 is dangerous on the binding lane 1 but would be clear on lane 2.
    @test !Traffic.observed_action_is_safe(
        subject, subject_position, subject_direction, 1, 3, motions, ring,
    )
    @test Traffic.observed_action_is_safe(
        subject, subject_position, subject_direction, 2, 3, motions, ring,
    )
    before_lane = Traffic.Ark.get_components(world, subject, (Traffic.LaneProposal,))[1].lane
    Traffic.propose_speeds!(world)
    lane, speed, path = Traffic.Ark.get_components(
        world, subject, (Traffic.LaneProposal, Traffic.SpeedProposal, Traffic.MovementPath),
    )
    @test before_lane == lane.lane == 1
    @test speed.value == 2  # fastest safe lower speed, not the unavoidable fallback
    @test all(position.x == 1 for position in path.positions)
    @test Traffic.Ark.get_resource(world, Traffic.CapabilityTickDiagnostics).dangerous_proposals == 1
    @test Traffic.Ark.get_resource(world, Traffic.CapabilityTickDiagnostics).accepted_dangerous_proposals == 0

    clear_world, clear_subject = speed_fixture(1.0; dangerous = false, cars = 1)
    Traffic.propose_speeds!(clear_world)
    @test Traffic.Ark.get_components(clear_world, clear_subject, (Traffic.SpeedProposal,))[1].value == 3
    @test Traffic.Ark.get_resource(clear_world, Traffic.CapabilityTickDiagnostics).dangerous_proposals == 0

    accepting_world, accepting_subject = speed_fixture(0.0)
    Traffic.propose_speeds!(accepting_world)
    @test Traffic.Ark.get_components(accepting_world, accepting_subject, (Traffic.SpeedProposal,))[1].value == 3
    accepting_diagnostics = Traffic.Ark.get_resource(accepting_world, Traffic.CapabilityTickDiagnostics)
    @test accepting_diagnostics.dangerous_proposals == 1
    @test accepting_diagnostics.accepted_dangerous_proposals == 1

    @test all(
        0.0 <= car.risk_aversion <= 1.0
            for car in Traffic.traffic_snapshot(
                Traffic.setup_world(Traffic.ModelArgs(
                    seed = 12,
                    params = Traffic.ModelParams(init_agents = 20, ring_y = 40),
                    prediction_strategy = Traffic.CapabilityModel(), steps = 0,
                )
            )).cars
    )
    initial_world = Traffic.setup_world(Traffic.ModelArgs(
        seed = 12,
        params = Traffic.ModelParams(init_agents = 20, ring_y = 40),
        prediction_strategy = Traffic.CapabilityModel(), steps = 0,
    ))
    initial_cars = Traffic.traffic_snapshot(initial_world).cars
    @test length(unique(car.risk_aversion for car in initial_cars)) > 1
    @test all(car.speed == 3 for car in initial_cars)
    @test all(Traffic.Ark.has_components(initial_world, car.entity, (Traffic.RiskAversion,))
              for car in initial_cars)
    repeat_initial = Traffic.traffic_snapshot(Traffic.setup_world(Traffic.ModelArgs(
        seed = 12,
        params = Traffic.ModelParams(init_agents = 20, ring_y = 40),
        prediction_strategy = Traffic.CapabilityModel(), steps = 0,
    )))
    @test [car.risk_aversion for car in initial_cars] ==
          [car.risk_aversion for car in repeat_initial.cars]

    # The speed decision has exactly one stochastic draw per decision entity.
    draw_world, _ = speed_fixture(1.0)
    expected = deepcopy(Traffic.Ark.get_resource(draw_world, Traffic.SimulationRNG).rng)
    n = length(Traffic.traffic_snapshot(draw_world).cars)
    rand(expected, n)
    Traffic.propose_speeds!(draw_world)
    @test rand(expected) == rand(Traffic.Ark.get_resource(
        draw_world, Traffic.SimulationRNG).rng)

    # Private proposals of an observed car cannot affect the subject.
    private_a, private_subject = speed_fixture(1.0)
    private_b = deepcopy(private_a)
    other = only(filter(car -> car.entity != private_subject,
                        Traffic.traffic_snapshot(private_b).cars)).entity
    for (es, lanes, speeds, paths) in Traffic.Query(
            private_b, (Traffic.LaneProposal, Traffic.SpeedProposal, Traffic.MovementPath))
        for i in eachindex(es)
            es[i] == other || continue
            lanes[i] = Traffic.LaneProposal(2)
            speeds[i] = Traffic.SpeedProposal(1)
            paths[i] = Traffic.MovementPath(Traffic.Position(2, 19))
        end
    end
    Traffic.propose_speeds!(private_a)
    Traffic.propose_speeds!(private_b)
    @test Traffic.Ark.get_components(private_a, private_subject, (Traffic.SpeedProposal,))[1] ==
          Traffic.Ark.get_components(private_b, private_subject, (Traffic.SpeedProposal,))[1]

    # Resolution is symmetric: conflicting submitted paths have no winner.
    collision_world, _ = speed_fixture(1.0)
    collision_entities = [car.entity for car in Traffic.traffic_snapshot(collision_world).cars]
    for (es, positions, paths) in Traffic.Query(collision_world, (Traffic.Position, Traffic.MovementPath))
        for i in eachindex(es)
            positions[i] = Traffic.Position(es[i] == collision_entities[1] ? 1 : 2, 1)
            paths[i] = Traffic.MovementPath(Traffic.Position(1, 2))
        end
    end
    replacements = Traffic.resolve_capability_movement!(collision_world)
    @test length(replacements) == 2
    @test isempty(Traffic.traffic_snapshot(collision_world).cars)

    # A position exchange is a collision at the resolver level, with no winner.
    exchange_world, _ = speed_fixture(1.0)
    exchange_entities = [car.entity for car in Traffic.traffic_snapshot(exchange_world).cars]
    for (es, positions, paths) in Traffic.Query(
            exchange_world, (Traffic.Position, Traffic.MovementPath))
        for i in eachindex(es)
            if es[i] == exchange_entities[1]
                positions[i] = Traffic.Position(1, 1)
                paths[i] = Traffic.MovementPath(Traffic.Position(1, 2))
            else
                positions[i] = Traffic.Position(1, 2)
                paths[i] = Traffic.MovementPath(Traffic.Position(1, 1))
            end
        end
    end
    @test Traffic.edges_cross(
        Traffic.Position(1, 1), Traffic.Position(1, 2),
        Traffic.Position(1, 2), Traffic.Position(1, 1),
    )
    exchange_replacements = Traffic.resolve_capability_movement!(exchange_world)
    @test length(exchange_replacements) == 2
    @test isempty(Traffic.traffic_snapshot(exchange_world).cars)

    @test Traffic.paths_conflict(Traffic.Position(1, 1),
        (Traffic.Position(1, 2), Traffic.Position(1, 2), Traffic.Position(1, 2)),
        Traffic.Position(2, 1),
        (Traffic.Position(1, 2), Traffic.Position(1, 3), Traffic.Position(1, 3)))
    @test Traffic.paths_conflict(Traffic.Position(1, 1),
        (Traffic.Position(1, 2), Traffic.Position(1, 3), Traffic.Position(1, 3)),
        Traffic.Position(1, 2),
        (Traffic.Position(1, 1), Traffic.Position(1, 1), Traffic.Position(1, 1)))
    @test Traffic.paths_conflict(Traffic.Position(1, 1),
        (Traffic.Position(2, 2), Traffic.Position(2, 2), Traffic.Position(2, 2)),
        Traffic.Position(2, 1),
        (Traffic.Position(1, 2), Traffic.Position(1, 2), Traffic.Position(1, 2)))
end

@testset "Habit, convention, and social-habit semantics" begin
    params = Traffic.ModelParams(
        ϵ = 0.0,
        init_agents = 1,
        ring_y = 20,
        lookahead = 5,
    )

    habit_model = Traffic.CapabilityModel(
        same_direction_share = 0.0,
        opposite_direction_share = 0.0,
        avoidance_share = 0.0,
        habit_share = 1.0,
        convention_share = 0.0,
        social_habit_share = 0.0,
        habit_weight = 0.75,
        max_speed = 1,
    )
    habit_world = Traffic.setup_world(
        Traffic.ModelArgs(
            seed = 19,
            params = params,
            prediction_strategy = habit_model,
            steps = 0,
        ),
    )
    initial_habit_car = only(Traffic.traffic_snapshot(habit_world).cars)
    initial_side = Traffic.relative_lane_sign(
        initial_habit_car.lane, initial_habit_car.direction,
    )
    Traffic.step!(habit_world, habit_model)
    first_habit_car = only(Traffic.traffic_snapshot(habit_world).cars)
    @test first_habit_car.habitus ≈ initial_side / (params.K + 2)

    habit_formation = Traffic.Ark.get_components(
        habit_world, first_habit_car.entity, (Traffic.HabitFormation,),
    )[1]
    expected_habit_lr = habit_model.habit_weight * habit_formation.disposition *
                        first_habit_car.habitus
    Traffic.step!(habit_world, habit_model)
    @test only(Traffic.traffic_snapshot(habit_world).cars).decision ≈ expected_habit_lr

    convention_model = Traffic.CapabilityModel(
        same_direction_share = 0.0,
        opposite_direction_share = 0.0,
        avoidance_share = 0.0,
        habit_share = 0.0,
        convention_share = 1.0,
        social_habit_share = 0.0,
        convention_weight = 0.75,
        convention_learning_rate = 0.25,
        convention_noise = 0.0,
        max_speed = 1,
    )
    convention_world = Traffic.setup_world(
        Traffic.ModelArgs(
            seed = 20,
            params = params,
            prediction_strategy = convention_model,
            steps = 0,
        ),
    )
    for (_, observations) in Traffic.Query(
            convention_world, (Traffic.LocalObservation,),
        )
        observations[1] = Traffic.LocalObservation(0.5, 0.5, 0.0, 0.0, 0.8, 4)
    end
    Traffic.learn_conventions!(convention_world)
    convention = only([
        value
            for (_, values) in Traffic.Query(
                convention_world, (Traffic.PerceivedConvention,),
            )
            for value in values
    ])
    @test convention.value ≈ 0.2
    @test convention.confidence ≈ 0.25

    Traffic.reset_lane_scores!(convention_world)
    for (_, scores) in Traffic.Query(convention_world, (Traffic.LaneScore,))
        scores[1] = Traffic.LaneScore(10.0)
    end
    Traffic.add_convention_response!(convention_world)
    Traffic.propose_lanes!(convention_world)
    expected_convention_lr = 10.0 + convention_model.convention_weight *
                                    convention.confidence * convention.value
    convention_entity = only(Traffic.traffic_snapshot(convention_world).cars).entity
    @test Traffic.Ark.get_components(
        convention_world, convention_entity, (Traffic.LR,),
    )[1].val ≈ expected_convention_lr

    for (_, observations) in Traffic.Query(
            convention_world, (Traffic.LocalObservation,),
        )
        observations[1] = Traffic.LocalObservation()
    end
    Traffic.learn_conventions!(convention_world)
    @test Traffic.Ark.get_components(
        convention_world, convention_entity, (Traffic.PerceivedConvention,),
    )[1] == convention

    social_model = Traffic.CapabilityModel(
        same_direction_share = 0.0,
        opposite_direction_share = 0.0,
        avoidance_share = 0.0,
        habit_share = 0.0,
        convention_share = 0.0,
        social_habit_share = 1.0,
        social_habit_weight = 0.75,
        social_habit_learning_rate = 0.25,
        social_habit_noise = 0.0,
        social_trace_retention = 0.5,
        social_trace_deposit = 0.4,
        max_speed = 2,
    )
    social_world = Traffic.setup_world(
        Traffic.ModelArgs(
            seed = 21,
            params = params,
            prediction_strategy = social_model,
            steps = 0,
        ),
    )

    social_car = only(Traffic.traffic_snapshot(social_world).cars)
    entity = social_car.entity
    @test Traffic.Ark.has_components(
        social_world,
        entity,
        (Traffic.SocialHabitFormation, Traffic.SocialHabitus),
    )
    @test Traffic.capability_mask(social_world, entity) == UInt8(1) << 5

    trace_lane = social_car.direction == Traffic.Clockwise ? 1 : 2
    trace_path = (
        Traffic.Position(trace_lane, 2),
        Traffic.Position(trace_lane, 3),
        Traffic.Position(trace_lane, 3),
    )
    for (_, speeds, paths) in Traffic.Query(
            social_world, (Traffic.Speed, Traffic.MovementPath),
        )
        speeds[1] = Traffic.Speed(2)
        paths[1] = Traffic.MovementPath(trace_path)
    end
    Traffic.deposit_success_traces!(social_world)
    traces = Traffic.Ark.get_resource(
        social_world, Traffic.SuccessfulDriverTrace,
    ).grid
    @test traces[trace_lane, 2] ≈ 0.4
    @test traces[trace_lane, 3] ≈ 0.4
    Traffic.decay_success_traces!(social_world)
    @test traces[trace_lane, 2] ≈ 0.2
    @test traces[trace_lane, 3] ≈ 0.2

    fill!(traces, 0.0)
    current_position = Traffic.Ark.get_components(
        social_world, entity, (Traffic.Position,),
    )[1]
    trace_y = Traffic.ahead_y(
        current_position.y, social_car.direction, 1, params.ring_y,
    )
    traces[trace_lane, trace_y] = 0.8
    Traffic.observe_capability_traffic!(social_world)
    observation = Traffic.Ark.get_components(
        social_world, entity, (Traffic.LocalObservation,),
    )[1]
    @test observation.success_trace ≈ 0.8
    @test observation.success_trace_samples == 1

    Traffic.learn_social_habits!(social_world)

    learned = only([
        habit.value
            for (_, habits) in Traffic.Query(social_world, (Traffic.SocialHabitus,))
            for habit in habits
    ])
    @test learned ≈ 0.2

    Traffic.reset_lane_scores!(social_world)
    Traffic.add_social_habit_response!(social_world)
    Traffic.propose_lanes!(social_world)
    formation = Traffic.Ark.get_components(
        social_world, entity, (Traffic.SocialHabitFormation,),
    )[1]
    expected_social_lr = social_model.social_habit_weight *
                         formation.disposition * learned
    @test Traffic.Ark.get_components(
        social_world, entity, (Traffic.LR,),
    )[1].val ≈ expected_social_lr

    for (_, observations) in Traffic.Query(
            social_world, (Traffic.LocalObservation,),
        )
        # A convention observation is deliberately not a success-trace observation.
        observations[1] = Traffic.LocalObservation(0.5, 0.5, 0.0, 0.0, -1.0, 4)
    end
    Traffic.learn_social_habits!(social_world)
    persisted = Traffic.Ark.get_components(
        social_world, entity, (Traffic.SocialHabitus,),
    )[1]
    @test persisted.value == learned

    @test_throws ArgumentError Traffic.validate(
        Traffic.CapabilityModel(social_habit_share = 1.1),
    )
    @test_throws ArgumentError Traffic.validate(
        Traffic.CapabilityModel(social_habit_learning_rate = -0.1),
    )
    @test_throws ArgumentError Traffic.validate(
        Traffic.CapabilityModel(social_trace_retention = 1.0),
    )
    @test_throws ArgumentError Traffic.validate(
        Traffic.CapabilityModel(social_trace_deposit = 0.0),
    )
end

@testset "Evolutionary capability replacement" begin
    parent = Traffic.CapabilityGenome(
        0.35,
        0.8,
        nothing,
        1.2,
        nothing,
        Traffic.ConventionPerception(0.2, 0.05),
        nothing,
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
        ones(5),
        model.replacement_policy,
        Random.Xoshiro(1),
    )
    @test inherited == parent

    social_parent = Traffic.CapabilityGenome(
        0.65,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        Traffic.SocialHabitFormation(1.1, 0.25, 0.05),
    )
    social_model = Traffic.CapabilityModel(
        social_habit_share = 1.0,
        replacement_policy = model.replacement_policy,
    )
    inherited_social = Traffic.inherit_capability_genome(
        social_parent,
        social_model,
        ones(5),
        social_model.replacement_policy,
        Random.Xoshiro(4),
    )
    @test inherited_social == social_parent

    flip_policy = Traffic.EvolutionaryReplacement(
        capability_mutation_rate = 1.0,
        trait_mutation_scale = 0.0,
    )
    mutated = Traffic.inherit_capability_genome(
        parent, model, [1.0, 1.1, 1.2, 1.3, 1.4], flip_policy, Random.Xoshiro(2),
    )
    @test isnothing(mutated.same_direction)
    @test mutated.opposite_direction == 1.1
    @test isnothing(mutated.avoidance)
    @test mutated.habit == 1.3
    @test isnothing(mutated.convention)

    removed_social = Traffic.inherit_capability_genome(
        social_parent,
        social_model,
        ones(5),
        flip_policy,
        Random.Xoshiro(5),
    )
    @test isnothing(removed_social.social_habit)

    disabled_model = Traffic.CapabilityModel(
        habit_share = 0.0,
        convention_share = 0.0,
        replacement_policy = flip_policy,
    )
    disabled_parent = Traffic.CapabilityGenome(
        0.5, nothing, nothing, nothing, nothing, nothing, nothing,
    )
    disabled_mutation = Traffic.inherit_capability_genome(
        disabled_parent,
        disabled_model,
        ones(5),
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

@testset "Capability replacement and diagnostics" begin
    entry_model = Traffic.CapabilityModel()
    entry_world = Traffic.setup_world(Traffic.ModelArgs(
        seed = 808,
        params = Traffic.ModelParams(init_agents = 1, ring_y = 20),
        prediction_strategy = entry_model, steps = 0,
    ))
    survivor = only(Traffic.traffic_snapshot(entry_world).cars).entity
    for (_, risks) in Traffic.Query(entry_world, (Traffic.RiskAversion,))
        risks[1] = Traffic.RiskAversion(0.123456)
    end
    Traffic.spawn_new_entities!(entry_world, [Traffic.CapabilityReplacementSpec(Traffic.Clockwise)])
    children = filter(car -> car.entity != survivor, Traffic.traffic_snapshot(entry_world).cars)
    @test length(children) == 1
    child = only(children)
    @test child.age == 1
    @test child.speed == entry_model.max_speed
    @test 0.0 <= child.risk_aversion <= 1.0
    @test child.risk_aversion != 0.123456

    parent = Traffic.CapabilityGenome(
        0.271828, 0.8, nothing, 1.2, nothing,
        Traffic.ConventionPerception(0.2, 0.05), nothing,
    )
    evolutionary = Traffic.CapabilityModel(
        replacement_policy = Traffic.EvolutionaryReplacement(
            capability_mutation_rate = 0.0, trait_mutation_scale = 0.0,
        ),
    )
    @test Traffic.inherit_capability_genome(
        parent, evolutionary, ones(5), evolutionary.replacement_policy, Random.Xoshiro(1),
    ) == parent

    # With one survivor the integrated replacement has an unambiguous parent.
    evo_world = Traffic.setup_world(Traffic.ModelArgs(
        seed = 909,
        params = Traffic.ModelParams(init_agents = 2, ring_y = 20),
        prediction_strategy = evolutionary, steps = 0,
    ))
    parent_entity = first(Traffic.traffic_snapshot(evo_world).cars).entity
    parent_genome = Traffic.capability_genome(evo_world, parent_entity)
    doomed = last(Traffic.traffic_snapshot(evo_world).cars).entity
    Traffic.Ark.remove_entity!(evo_world, doomed)
    Traffic.spawn_new_entities!(evo_world,
        [Traffic.CapabilityReplacementSpec(Traffic.Counterclockwise)])
    newborn = first(filter(car -> car.entity != parent_entity,
                           Traffic.traffic_snapshot(evo_world).cars))
    @test Traffic.capability_genome(evo_world, newborn.entity) == parent_genome
    @test newborn.age == 1
    @test newborn.habitus == 0.0

    for (risk, seed) in ((0.0, 111), (1.0, 112))
        extreme = Traffic.CapabilityGenome(
            risk, nothing, nothing, nothing, nothing, nothing, nothing,
        )
        mutated = Traffic.inherit_capability_genome(
            extreme, evolutionary,
            ones(5), Traffic.EvolutionaryReplacement(
                capability_mutation_rate = 0.0, trait_mutation_scale = 1.0e6,
            ), Random.Xoshiro(seed),
        )
        @test 0.0 <= mutated.risk_aversion <= 1.0
    end

    diagnostic_model = Traffic.CapabilityModel(
        replacement_policy = Traffic.EvolutionaryReplacement(
            capability_mutation_rate = 0.1, trait_mutation_scale = 0.1,
        ),
    )
    diagnostic_args = Traffic.ModelArgs(
        seed = 1234,
        params = Traffic.ModelParams(init_agents = 12, ring_y = 30),
        prediction_strategy = diagnostic_model, steps = 5,
    )
    diagnostic_world = Traffic.setup_world(diagnostic_args)
    for _ in 1:diagnostic_args.steps
        Traffic.step!(diagnostic_world, diagnostic_model)
    end
    logger = Traffic.Ark.get_resource(diagnostic_world, Traffic.Logger)
    fields = (logger.mean_risk_aversion, logger.std_risk_aversion,
              logger.distribution_risk_aversion, logger.survivor_risk,
              logger.selected_parent_risk, logger.dangerous_proposals,
              logger.accepted_dangerous_proposals, logger.mean_proposed_speed,
              logger.mean_realized_speed, logger.replacement_pressure)
    @test all(length(field) == diagnostic_args.steps for field in fields)
    @test all(isfinite, logger.mean_risk_aversion)
    @test all(0.0 .<= logger.mean_risk_aversion .<= 1.0)
    @test all(0.0 .<= logger.replacement_pressure .<= 1.0)
    @test all(0 .<= logger.accepted_dangerous_proposals .<= logger.dangerous_proposals)
    @test all(1.0 .<= logger.mean_proposed_speed .<= 3.0)
    @test all(1.0 .<= logger.mean_realized_speed .<= 3.0)
end

@testset "Completed movement experiment metric" begin
    config = ExperimentConfig(
        seeds = 20260901:20260901,
        steps = 10,
        burn_in = 0,
        population = 20,
        ring_y = 20,
        lookahead = 10,
    )
    capability = run_capability_condition(config, first(config.seeds), :no_habit)
    sequential = run_sequential_condition(config, first(config.seeds), :no_habit)

    @test capability.completed_cells_per_car_step ≈
          capability.mean_speed - capability.max_speed * capability.replacement_rate
    @test sequential.completed_cells_per_car_step ≈ 1.0 - sequential.replacement_rate
    @test 0.0 <= capability.completed_cells_per_car_step <= capability.max_speed
    @test 0.0 <= sequential.completed_cells_per_car_step <= sequential.max_speed

    mktempdir() do directory
        path = write_results(joinpath(directory, "completed_movement.csv"),
                             [capability, sequential])
        @test isequal(read_results(path), [capability, sequential])
    end
end

@testset "Non-heritable newborn risk hook" begin
    function risk_hook_fixture()
        model = Traffic.CapabilityModel(
            replacement_policy = Traffic.EvolutionaryReplacement(
                capability_mutation_rate = 0.0, trait_mutation_scale = 0.0,
            ),
        )
        world = Traffic.setup_world(Traffic.ModelArgs(
            seed = 4242,
            params = Traffic.ModelParams(init_agents = 1, ring_y = 20),
            prediction_strategy = model, steps = 0,
        ))
        survivor = only(Traffic.traffic_snapshot(world).cars).entity
        for (_, risks) in Traffic.Query(world, (Traffic.RiskAversion,))
            risks[1] = Traffic.RiskAversion(0.123456)
        end
        before = Set(survivor for (entities, _) in Traffic.Query(world, (Traffic.Position,))
                     for survivor in entities)
        Traffic.spawn_new_entities!(world, [
            Traffic.CapabilityReplacementSpec(Traffic.Clockwise),
            Traffic.CapabilityReplacementSpec(Traffic.Counterclockwise),
        ])
        return world, before, survivor
    end

    world, before, survivor = risk_hook_fixture()
    expected_rng = deepcopy(Traffic.simulation_rng(world))
    expected = rand(expected_rng, 2)
    count = uniform_newborn_risk!(world, before, 1)
    cars = sort!(collect(Traffic.traffic_snapshot(world).cars);
                 by = car -> (getfield(car.entity, :_id), getfield(car.entity, :_gen)))
    newborns = filter(car -> car.entity ∉ before, cars)
    @test count == 2
    @test length(newborns) == 2
    @test [car.risk_aversion for car in newborns] == expected
    @test only(filter(car -> car.entity == survivor, cars)).risk_aversion == 0.123456
    @test all(0.0 <= car.risk_aversion <= 1.0 for car in newborns)

    current_entities = Set(car.entity for car in cars)
    expected_rng = deepcopy(Traffic.simulation_rng(world))
    @test uniform_newborn_risk!(world, current_entities, 2) == 0
    @test rand(Traffic.simulation_rng(world)) == rand(expected_rng)

    repeated_world, repeated_before, _ = risk_hook_fixture()
    @test uniform_newborn_risk!(repeated_world, repeated_before, 1) == 2
    repeated_cars = sort!(collect(Traffic.traffic_snapshot(repeated_world).cars);
                          by = car -> (getfield(car.entity, :_id), getfield(car.entity, :_gen)))
    @test [car.risk_aversion for car in newborns] ==
          [car.risk_aversion for car in filter(car -> car.entity ∉ repeated_before, repeated_cars)]
end

@testset "Evolutionary replacement with no survivors" begin
    model = Traffic.CapabilityModel(
        replacement_policy = Traffic.EvolutionaryReplacement(
            capability_mutation_rate = 0.0,
            trait_mutation_scale = 0.0,
        ),
    )
    args = Traffic.ModelArgs(
        seed = 555,
        params = Traffic.ModelParams(init_agents = 6, ring_y = 20),
        prediction_strategy = model,
        steps = 0,
    )
    world = Traffic.setup_world(args)

    # Remove every car so no surviving parent genome exists.
    for entity in [car.entity for car in Traffic.traffic_snapshot(world).cars]
        Traffic.Ark.remove_entity!(world, entity)
    end
    @test isempty(Traffic.traffic_snapshot(world).cars)

    amount = 8
    directions = repeat(
        [Traffic.Clockwise, Traffic.Counterclockwise], cld(amount, 2),
    )[1:amount]
    specs = [Traffic.CapabilityReplacementSpec(dir) for dir in directions]

    Traffic.spawn_new_entities!(world, specs)

    snapshot = Traffic.traffic_snapshot(world)
    @test length(snapshot.cars) == amount  # population restored
    @test all(0.0 <= car.risk_aversion <= 1.0 for car in snapshot.cars)
    # Multiple independent entry draws should differ.
    @test length(unique(car.risk_aversion for car in snapshot.cars)) > 1
    @test all(car.speed == model.max_speed for car in snapshot.cars)
    @test isempty(
        Traffic.Ark.get_resource(world, Traffic.CapabilityTickDiagnostics).selected_parent_risks,
    )
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
include(
    joinpath(
        @__DIR__,
        "..",
        "Sugarscape",
        "agent_oriented",
        "test",
        "runtests.jl",
    ),
)

using Ark
using EBM
using Test

const SugarECS = EBM.Sugarscape

const LOGGER_FIELDS = (
    :step,
    :population,
    :mean_wealth,
    :median_wealth,
    :gini,
    :mean_age,
    :total_agent_sugar,
    :total_landscape_sugar,
    :moved,
    :conflicts,
    :harvested,
    :deaths,
    :starvation_deaths,
    :old_age_deaths,
    :replacements,
    :births,
    :infected,
    :infections,
    :recoveries,
)

ecs_logger(world) = Ark.get_resource(world, SugarECS.Logger)
ecs_landscape(world) = Ark.get_resource(world, SugarECS.SugarLandscape)

function position_signature(position)
    if hasproperty(position, :x) && hasproperty(position, :y)
        return (getproperty(position, :x), getproperty(position, :y))
    end
    return (position[1], position[2])
end

function citizen_signature(citizens)
    states = [
        (
            citizen.id,
            position_signature(citizen.position)...,
            citizen.vision,
            citizen.metabolism,
            citizen.sugar,
            citizen.age,
            citizen.maximum_age,
            citizen.sex,
            citizen.infected,
        ) for citizen in citizens
    ]
    sort!(states; by = first)
    return states
end

function landscape_signature(value)
    matrix = hasproperty(value, :current) ? getproperty(value, :current) : value
    return copy(matrix)
end

function logger_signature(value)
    return Tuple(copy(getproperty(value, field)) for field in LOGGER_FIELDS)
end

function assert_same_state(agent_module, world, model)
    @test citizen_signature(SugarECS.citizen_snapshot(world)) ==
          citizen_signature(agent_module.citizen_snapshot(model))
    @test landscape_signature(ecs_landscape(world)) ==
          landscape_signature(agent_module.landscape(model))
    @test isequal(logger_signature(ecs_logger(world)), logger_signature(agent_module.logger(model)))
    return nothing
end

function assert_stepwise_equivalence(agent_module, args)
    world = SugarECS.setup_world(args)
    model = agent_module.setup_model(args)
    assert_same_state(agent_module, world, model)
    for _ in 1:args.steps
        SugarECS.step!(world)
        agent_module.step!(model)
        assert_same_state(agent_module, world, model)
    end
    return world, model
end

function assert_run_equivalence(agent_module, args)
    world = SugarECS.run_model(args)
    model = agent_module.run_model(args)
    assert_same_state(agent_module, world, model)
    return nothing
end

@testset "Agents.jl Sugarscape public interface" begin
    for agent_module in (SugarECS.AgentSequential, SugarECS.AgentSynchronous)
        @test isdefined(agent_module, :setup_model)
        @test isdefined(agent_module, :step!)
        @test isdefined(agent_module, :run_model)
        @test isdefined(agent_module, :citizen_snapshot)
        @test isdefined(agent_module, :logger)
        @test isdefined(agent_module, :landscape)
    end
end

@testset "Agents.jl initialization and baseline tick equivalence" begin
    cases = (
        (
            SugarECS.AgentSequential,
            SugarECS.ShuffledSequentialMovement,
        ),
        (
            SugarECS.AgentSynchronous,
            SugarECS.SynchronousMovement,
        ),
    )
    for (agent_module, movement_mode) in cases
        @testset "$(nameof(agent_module))" begin
            params = SugarECS.ModelParams(
                width = 12,
                height = 10,
                population = 30,
                maximum_patch_sugar = 4,
                growback_rate = 1,
                minimum_vision = 1,
                maximum_vision = 5,
                minimum_metabolism = 1,
                maximum_metabolism = 4,
                minimum_initial_sugar = 8,
                maximum_initial_sugar = 24,
                minimum_lifespan = 30,
                maximum_lifespan = 50,
                movement_mode = movement_mode,
            )
            args = SugarECS.ModelArgs(seed = 20260813, params = params, steps = 8)
            world, model = assert_stepwise_equivalence(agent_module, args)

            ecs_log = ecs_logger(world)
            @test length(ecs_log.step) == args.steps
            @test all(==(params.population), ecs_log.population)
            movement_mode == SugarECS.ShuffledSequentialMovement &&
                @test all(iszero, ecs_log.conflicts)
        end
    end
end

@testset "Agents.jl run_model equivalence" begin
    for (agent_module, movement_mode) in (
        (SugarECS.AgentSequential, SugarECS.ShuffledSequentialMovement),
        (SugarECS.AgentSynchronous, SugarECS.SynchronousMovement),
    )
        params = SugarECS.ModelParams(
            width = 9,
            height = 8,
            population = 18,
            movement_mode = movement_mode,
        )
        args = SugarECS.ModelArgs(seed = 74, params = params, steps = 4)
        assert_run_equivalence(agent_module, args)
    end
end

@testset "Agents.jl synchronous contested-destination equivalence" begin
    capacity = zeros(Int64, 5, 5)
    capacity[3, 3] = 20
    params = SugarECS.ModelParams(
        width = 5,
        height = 5,
        population = 8,
        maximum_patch_sugar = 20,
        growback_rate = 0,
        minimum_vision = 4,
        maximum_vision = 4,
        minimum_metabolism = 0,
        maximum_metabolism = 0,
        minimum_initial_sugar = 20,
        maximum_initial_sugar = 20,
        minimum_lifespan = 100,
        maximum_lifespan = 100,
        movement_mode = SugarECS.SynchronousMovement,
    )
    args = SugarECS.ModelArgs(
        seed = 1,
        params = params,
        steps = 1,
        initial_capacity = capacity,
    )
    world, model = assert_stepwise_equivalence(SugarECS.AgentSynchronous, args)
    ecs_log = ecs_logger(world)

    # Four agents target the unique rich cell; precisely one moves and harvests it.
    @test ecs_log.conflicts == [3]
    @test ecs_log.moved == [1]
    @test ecs_log.harvested == [20]
    @test SugarECS.AgentSynchronous.logger(model).conflicts == [3]
    @test count(
        state -> (state[2], state[3]) == (3, 3),
        citizen_signature(SugarECS.AgentSynchronous.citizen_snapshot(model)),
    ) == 1
end

@testset "Agents.jl death and replacement equivalence" begin
    for (agent_module, movement_mode) in (
        (SugarECS.AgentSequential, SugarECS.ShuffledSequentialMovement),
        (SugarECS.AgentSynchronous, SugarECS.SynchronousMovement),
    )
        params = SugarECS.ModelParams(
            width = 5,
            height = 4,
            population = 10,
            maximum_patch_sugar = 0,
            growback_rate = 0,
            minimum_vision = 0,
            maximum_vision = 0,
            minimum_metabolism = 0,
            maximum_metabolism = 0,
            minimum_lifespan = 0,
            maximum_lifespan = 0,
            replace_dead = true,
            movement_mode = movement_mode,
        )
        args = SugarECS.ModelArgs(seed = 2, params = params, steps = 2)
        world, model = assert_stepwise_equivalence(agent_module, args)
        log = agent_module.logger(model)

        @test log.population == fill(params.population, args.steps)
        @test log.deaths == fill(params.population, args.steps)
        @test log.old_age_deaths == fill(params.population, args.steps)
        @test log.starvation_deaths == zeros(Int64, args.steps)
        @test log.replacements == fill(params.population, args.steps)
        @test minimum(first.(citizen_signature(SugarECS.citizen_snapshot(world)))) >
              params.population
    end
end

@testset "Agents.jl starvation without replacement equivalence" begin
    for (agent_module, movement_mode) in (
        (SugarECS.AgentSequential, SugarECS.ShuffledSequentialMovement),
        (SugarECS.AgentSynchronous, SugarECS.SynchronousMovement),
    )
        params = SugarECS.ModelParams(
            width = 3,
            height = 3,
            population = 4,
            maximum_patch_sugar = 0,
            growback_rate = 0,
            minimum_vision = 0,
            maximum_vision = 0,
            minimum_metabolism = 2,
            maximum_metabolism = 2,
            minimum_initial_sugar = 1,
            maximum_initial_sugar = 1,
            minimum_lifespan = 100,
            maximum_lifespan = 100,
            replace_dead = false,
            movement_mode = movement_mode,
        )
        args = SugarECS.ModelArgs(seed = 8, params = params, steps = 1)
        _, model = assert_stepwise_equivalence(agent_module, args)
        log = agent_module.logger(model)

        @test log.population == [0]
        @test log.deaths == [params.population]
        @test log.starvation_deaths == [params.population]
        @test log.old_age_deaths == [0]
        @test log.replacements == [0]
    end
end

@testset "Agents.jl disease transmission, cost, and recovery equivalence" begin
    for (agent_module, movement_mode) in (
        (SugarECS.AgentSequential, SugarECS.ShuffledSequentialMovement),
        (SugarECS.AgentSynchronous, SugarECS.SynchronousMovement),
    )
        params = SugarECS.ModelParams(
            width = 3,
            height = 3,
            population = 8,
            maximum_patch_sugar = 0,
            growback_rate = 0,
            minimum_vision = 0,
            maximum_vision = 0,
            minimum_metabolism = 0,
            maximum_metabolism = 0,
            minimum_initial_sugar = 20,
            maximum_initial_sugar = 20,
            minimum_lifespan = 100,
            maximum_lifespan = 100,
            replace_dead = false,
            movement_mode = movement_mode,
            initial_infection_probability = 0.5,
            disease_transmission_probability = 1.0,
            disease_duration = 2,
            disease_sugar_cost = 1,
        )
        args = SugarECS.ModelArgs(seed = 1, params = params, steps = 2)
        world, model = assert_stepwise_equivalence(agent_module, args)
        log = agent_module.logger(model)

        @test sum(log.infections) > 0
        @test sum(log.recoveries) > 0
        @test log.total_agent_sugar[end] < params.population * params.maximum_initial_sugar
        @test log.infected == ecs_logger(world).infected
    end
end

@testset "Agents.jl reproduction and inheritance-path equivalence" begin
    for (agent_module, movement_mode) in (
        (SugarECS.AgentSequential, SugarECS.ShuffledSequentialMovement),
        (SugarECS.AgentSynchronous, SugarECS.SynchronousMovement),
    )
        params = SugarECS.ModelParams(
            width = 3,
            height = 3,
            population = 6,
            maximum_patch_sugar = 0,
            growback_rate = 0,
            minimum_vision = 0,
            maximum_vision = 0,
            minimum_metabolism = 0,
            maximum_metabolism = 0,
            minimum_initial_sugar = 20,
            maximum_initial_sugar = 20,
            minimum_lifespan = 100,
            maximum_lifespan = 100,
            replace_dead = false,
            movement_mode = movement_mode,
            reproduction_enabled = true,
            minimum_fertility_age = 0,
            maximum_fertility_age = 100,
            reproduction_probability = 1.0,
        )
        args = SugarECS.ModelArgs(seed = 1, params = params, steps = 1)
        world, model = assert_stepwise_equivalence(agent_module, args)
        log = agent_module.logger(model)
        citizens = citizen_signature(agent_module.citizen_snapshot(model))

        @test log.births == [3]
        @test log.population == [9]
        @test length(citizens) == 9
        @test sum(state[6] for state in citizens) ==
              params.population * params.maximum_initial_sugar
        @test length(unique((state[2], state[3]) for state in citizens)) == length(citizens)
        @test citizen_signature(SugarECS.citizen_snapshot(world)) == citizens
    end
end

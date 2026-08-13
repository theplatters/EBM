using Ark
using EBM
using Test

const SugarModel = EBM.Sugarscape

function count_sugar_components(world, component_types)
    return sum(
        (length(first(result)) for result in Ark.Query(world, component_types));
        init = 0,
    )
end

function sugarscape_signature(world)
    return [
        (
            citizen.id,
            citizen.position.x,
            citizen.position.y,
            citizen.vision,
            citizen.metabolism,
            citizen.sugar,
            citizen.age,
            citizen.maximum_age,
            citizen.sex,
            citizen.infected,
        )
        for citizen in SugarModel.citizen_snapshot(world)
    ]
end

@testset "Sugarscape setup and landscape" begin
    params = SugarModel.ModelParams(width = 12, height = 10, population = 40)
    world = SugarModel.setup_world(SugarModel.ModelArgs(seed = 11, params = params, steps = 0))
    citizens = SugarModel.citizen_snapshot(world)
    landscape = Ark.get_resource(world, SugarModel.SugarLandscape)

    @test count_sugar_components(world, (SugarModel.CitizenId,)) == params.population
    @test count_sugar_components(world, (SugarModel.Female,)) == cld(params.population, 2)
    @test count_sugar_components(world, (SugarModel.Male,)) == params.population ÷ 2
    @test length(unique(citizen.position for citizen in citizens)) == params.population
    @test size(landscape.current) == (params.width, params.height)
    @test landscape.current == landscape.capacity
    @test all((0 .<= landscape.capacity) .& (landscape.capacity .<= params.maximum_patch_sugar))
    @test all(
        params.minimum_vision <= citizen.vision <= params.maximum_vision
        for citizen in citizens
    )
    @test all(
        params.minimum_metabolism <= citizen.metabolism <= params.maximum_metabolism
        for citizen in citizens
    )
end


@testset "Sugarscape type-based reproduction" begin
    params = SugarModel.ModelParams(
        width = 5,
        height = 5,
        population = 2,
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
        reproduction_enabled = true,
        minimum_fertility_age = 0,
        maximum_fertility_age = 100,
        reproduction_probability = 1.0,
    )
    world = SugarModel.setup_world(SugarModel.ModelArgs(seed = 14, params = params, steps = 0))
    for (entities, ids, positions, proposals) in Ark.Query(
        world,
        (SugarModel.CitizenId, SugarModel.Position, SugarModel.ProposedPosition),
    )
        @inbounds for i in eachindex(entities)
            position = ids[i].val == 1 ? SugarModel.Position(2, 2) : SugarModel.Position(3, 2)
            positions[i] = position
            proposals[i] = SugarModel.ProposedPosition(position)
        end
    end
    SugarModel.rebuild_occupancy!(world)
    SugarModel.reproduce!(world)
    citizens = SugarModel.citizen_snapshot(world)
    events = Ark.get_resource(world, SugarModel.StepEvents)

    @test length(citizens) == 3
    @test sum(citizen.sugar for citizen in citizens) == 40
    @test length(unique(citizen.position for citizen in citizens)) == 3
    @test events.births == 1
    @test count(citizen -> citizen.sex == :female, citizens) in (1, 2)
end

@testset "Sugarscape structural infection lifecycle" begin
    params = SugarModel.ModelParams(
        width = 4,
        height = 4,
        population = 2,
        disease_transmission_probability = 1.0,
        disease_duration = 2,
        disease_sugar_cost = 0,
    )
    world = SugarModel.setup_world(SugarModel.ModelArgs(seed = 19, params = params, steps = 0))
    infected_entity = nothing
    for (entities, ids, positions, proposals, immunity) in Ark.Query(
        world,
        (
            SugarModel.CitizenId,
            SugarModel.Position,
            SugarModel.ProposedPosition,
            SugarModel.ImmuneProfile,
        ),
    )
        @inbounds for i in eachindex(entities)
            position = ids[i].val == 1 ? SugarModel.Position(2, 2) : SugarModel.Position(3, 2)
            positions[i] = position
            proposals[i] = SugarModel.ProposedPosition(position)
            immunity[i] = SugarModel.ImmuneProfile(UInt64(ids[i].val))
            ids[i].val == 1 && (infected_entity = entities[i])
        end
    end
    Ark.add_components!(world, infected_entity, (SugarModel.Infection(0x00000000000000ff, 0),))
    SugarModel.rebuild_occupancy!(world)
    SugarModel.transmit_disease!(world)
    @test count_sugar_components(world, (SugarModel.Infection,)) == 2
    @test Ark.get_resource(world, SugarModel.StepEvents).infections == 1

    SugarModel.progress_infections!(world)
    @test count_sugar_components(world, (SugarModel.Infection,)) == 2
    SugarModel.progress_infections!(world)
    @test count_sugar_components(world, (SugarModel.Infection,)) == 0
    @test Ark.get_resource(world, SugarModel.StepEvents).recoveries == 2
end

@testset "Sugarscape seeded movement modes" begin
    for movement_mode in (
        SugarModel.ShuffledSequentialMovement,
        SugarModel.SynchronousMovement,
    )
        params = SugarModel.ModelParams(
            width = 16,
            height = 16,
            population = 80,
            movement_mode = movement_mode,
        )
        args = SugarModel.ModelArgs(seed = 2026, params = params, steps = 30)
        first_world = SugarModel.run_model(args)
        second_world = SugarModel.run_model(args)
        first_logger = Ark.get_resource(first_world, SugarModel.Logger)
        second_logger = Ark.get_resource(second_world, SugarModel.Logger)

        @test sugarscape_signature(first_world) == sugarscape_signature(second_world)
        @test first_logger.gini == second_logger.gini
        @test first_logger.deaths == second_logger.deaths
        @test length(first_logger.step) == args.steps
        @test all(==(params.population), first_logger.population)
        @test all(0.0 <= coefficient <= 1.0 for coefficient in first_logger.gini)
        final_positions = [citizen.position for citizen in SugarModel.citizen_snapshot(first_world)]
        @test length(unique(final_positions)) == length(final_positions)
    end
end

@testset "Synchronous movement conflict resolution" begin
    capacity = zeros(Int64, 3, 3)
    capacity[2, 2] = 10
    params = SugarModel.ModelParams(
        width = 3,
        height = 3,
        population = 2,
        growback_rate = 0,
        minimum_vision = 1,
        maximum_vision = 1,
        minimum_metabolism = 0,
        maximum_metabolism = 0,
        movement_mode = SugarModel.SynchronousMovement,
    )
    world = SugarModel.setup_world(
        SugarModel.ModelArgs(seed = 9, params = params, steps = 0, initial_capacity = capacity),
    )
    for (entities, ids, positions, proposals, sugars) in Ark.Query(
        world,
        (
            SugarModel.CitizenId,
            SugarModel.Position,
            SugarModel.ProposedPosition,
            SugarModel.Sugar,
        ),
    )
        @inbounds for i in eachindex(entities)
            position = ids[i].val == 1 ? SugarModel.Position(1, 2) : SugarModel.Position(3, 2)
            positions[i] = position
            proposals[i] = SugarModel.ProposedPosition(position)
            sugars[i] = SugarModel.Sugar(10)
        end
    end
    SugarModel.rebuild_occupancy!(world)
    SugarModel.plan_movements!(world)
    proposed = SugarModel.Position[]
    for (_, proposals) in Ark.Query(world, (SugarModel.ProposedPosition,))
        append!(proposed, SugarModel.Position(proposal) for proposal in proposals)
    end
    @test proposed == fill(SugarModel.Position(2, 2), 2)

    SugarModel.resolve_and_commit_movements!(world)
    citizens = SugarModel.citizen_snapshot(world)
    events = Ark.get_resource(world, SugarModel.StepEvents)
    @test count(citizen -> citizen.position == SugarModel.Position(2, 2), citizens) == 1
    @test sum(citizen.sugar for citizen in citizens) == 30
    @test events.conflicts == 1
    @test events.moved == 1
end

@testset "Sugarscape lifecycle and replacement" begin
    starvation_params = SugarModel.ModelParams(
        width = 2,
        height = 2,
        population = 1,
        maximum_patch_sugar = 0,
        growback_rate = 0,
        minimum_vision = 0,
        maximum_vision = 0,
        minimum_metabolism = 2,
        maximum_metabolism = 2,
        minimum_initial_sugar = 1,
        maximum_initial_sugar = 1,
        minimum_lifespan = 10,
        maximum_lifespan = 10,
        replace_dead = false,
    )
    world = SugarModel.run_model(
        SugarModel.ModelArgs(seed = 1, params = starvation_params, steps = 1),
    )
    logger = Ark.get_resource(world, SugarModel.Logger)
    @test isempty(SugarModel.citizen_snapshot(world))
    @test logger.deaths == [1]
    @test logger.starvation_deaths == [1]
    @test logger.replacements == [0]

    replacement_params = SugarModel.ModelParams(
        width = 5,
        height = 5,
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
    )
    world = SugarModel.run_model(
        SugarModel.ModelArgs(seed = 2, params = replacement_params, steps = 1),
    )
    logger = Ark.get_resource(world, SugarModel.Logger)
    @test logger.population == [replacement_params.population]
    @test logger.deaths == [replacement_params.population]
    @test logger.old_age_deaths == [replacement_params.population]
    @test logger.replacements == [replacement_params.population]
    @test all(
        citizen.id > replacement_params.population
        for citizen in SugarModel.citizen_snapshot(world)
    )
end

@testset "Sugarscape diagnostics and validation" begin
    @test SugarModel.gini_coefficient([0, 0, 0]) == 0.0
    @test SugarModel.gini_coefficient([0, 1]) == 0.5
    @test isnan(SugarModel.gini_coefficient(Int[]))

    params = SugarModel.ModelParams(width = 8, height = 8, population = 20)
    world = SugarModel.run_model(SugarModel.ModelArgs(seed = 31, params = params, steps = 5))
    logger = Ark.get_resource(world, SugarModel.Logger)
    @test !isnothing(SugarModel.plot_model_diagnostics(logger))
    @test !isnothing(SugarModel.plot_sugarscape(world))
    @test_throws ArgumentError SugarModel.plot_model_diagnostics(SugarModel.Logger())

    @test_throws ArgumentError SugarModel.setup_world(
        SugarModel.ModelArgs(
            params = SugarModel.ModelParams(width = 2, height = 2, population = 5),
        ),
    )
    @test_throws ArgumentError SugarModel.setup_world(
        SugarModel.ModelArgs(
            params = SugarModel.ModelParams(width = 3, height = 3, population = 2),
            initial_capacity = zeros(Int64, 2, 2),
        ),
    )
    @test_throws ArgumentError SugarModel.setup_world(
        SugarModel.ModelArgs(
            params = SugarModel.ModelParams(
                width = 3,
                height = 3,
                population = 2,
                replace_dead = true,
                reproduction_enabled = true,
            ),
        ),
    )
end

@testset "Sugarscape interactive visualization" begin
    params = SugarModel.ModelParams(width = 8, height = 8, population = 20)
    args = SugarModel.ModelArgs(seed = 41, params = params, steps = 4)
    visualization = SugarModel.interactive_sugarscape(args; framerate = 10)

    @test visualization.figure isa SugarModel.Figure
    @test visualization.state[].step == 0
    @test visualization.state[].population == params.population

    SugarModel.step!(visualization, 2)
    first_signature = sugarscape_signature(visualization.world)
    @test visualization.state[].step == 2
    @test visualization.state[].history_steps == [1, 2]
    @test !visualization.running[]

    SugarModel.reset!(visualization)
    SugarModel.step!(visualization, 2)
    @test sugarscape_signature(visualization.world) == first_signature

    SugarModel.step!(visualization, 10)
    @test visualization.state[].step == args.steps
    @test_throws ArgumentError SugarModel.step!(visualization, -1)

    SugarModel.reset!(visualization)
    SugarModel.play!(visualization)
    wait(visualization.task)
    @test visualization.state[].step == args.steps
    @test !visualization.running[]

    @test_throws ArgumentError SugarModel.interactive_sugarscape(
        SugarModel.ModelArgs(params = params, steps = 0),
    )
    SugarModel.stop!(visualization)
end

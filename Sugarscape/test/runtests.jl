using Ark
using EBM
using Random
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
    parent_genotypes = Dict(1 => UInt64(0b10101), 2 => UInt64(0b10011))
    for (entities, ids, positions, proposals, immunities) in Ark.Query(
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
            immunities[i] = SugarModel.ImmuneProfile(parent_genotypes[ids[i].val])
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
    child_immunity = nothing
    for (entities, ids, immunities) in
        Ark.Query(world, (SugarModel.CitizenId, SugarModel.ImmuneProfile))
        @inbounds for i in eachindex(entities)
            ids[i].val == 3 && (child_immunity = immunities[i])
        end
    end
    differing = xor(parent_genotypes[1], parent_genotypes[2])
    @test child_immunity.phenotype == child_immunity.genotype
    @test child_immunity.genotype & ~differing == parent_genotypes[1] & ~differing
end

@testset "Sugarscape adaptive immune response" begin
    disease = SugarModel.Disease(0b111, 3)
    @test SugarModel.immune_to(0b00111, disease, 5)
    @test SugarModel.immune_to(0b11100, disease, 5)
    @test !SugarModel.immune_to(0b00000, disease, 5)
    trained = UInt64(0)
    trained = SugarModel.train_immunity(trained, disease, 5)
    @test trained == 0b00001
    trained = SugarModel.train_immunity(trained, disease, 5)
    @test trained == 0b00011
    trained = SugarModel.train_immunity(trained, disease, 5)
    @test trained == 0b00111

    rng = Random.Xoshiro(41)
    mother = UInt64(0b10101)
    father = UInt64(0b10011)
    child = SugarModel.inherit_immune_genotype(rng, mother, father, 5)
    differing = xor(mother, father)
    @test child & ~differing == mother & ~differing
    @test iszero(child & ~SugarModel.bit_mask(5))
end

@testset "Sugarscape multi-disease transmission and recovery" begin
    params = SugarModel.ModelParams(
        width = 5,
        height = 5,
        population = 3,
        disease_catalog_size = 2,
        initial_diseases_per_citizen = 0,
        minimum_disease_length = 3,
        maximum_disease_length = 3,
        immune_system_length = 5,
        disease_sugar_cost = 1,
    )
    world = SugarModel.setup_world(SugarModel.ModelArgs(seed = 19, params = params, steps = 0))
    catalog = Ark.get_resource(world, SugarModel.DiseaseCatalog)
    empty!(catalog.diseases)
    append!(catalog.diseases, (SugarModel.Disease(0b111, 3), SugarModel.Disease(0b101, 3)))
    infected = Pair{Ark.Entity, SugarModel.Infection}[]
    recipient = nothing
    initial_sugar = Dict{Int64, Int64}()
    for (entities, ids, positions, proposals, immunity, sugars) in Ark.Query(
        world,
        (
            SugarModel.CitizenId,
            SugarModel.Position,
            SugarModel.ProposedPosition,
            SugarModel.ImmuneProfile,
            SugarModel.Sugar,
        ),
    )
        @inbounds for i in eachindex(entities)
            position = SugarModel.Position(ids[i].val, 3)
            positions[i] = position
            proposals[i] = SugarModel.ProposedPosition(position)
            immunity[i] = SugarModel.ImmuneProfile(0)
            initial_sugar[ids[i].val] = sugars[i].val
            ids[i].val == 1 && push!(infected, entities[i] => SugarModel.Infection(0b01))
            ids[i].val == 2 && (recipient = entities[i])
            ids[i].val == 3 && push!(infected, entities[i] => SugarModel.Infection(0b10))
        end
    end
    for (entity, infection) in infected
        Ark.add_components!(world, entity, (infection,))
    end
    SugarModel.rebuild_occupancy!(world)
    SugarModel.transmit_disease!(world)
    @test count_sugar_components(world, (SugarModel.Infection,)) == 3
    @test Ark.get_components(world, recipient, (SugarModel.Infection,))[1].diseases == 0b11
    @test Ark.get_resource(world, SugarModel.StepEvents).infections == 2

    for _ in 1:3
        SugarModel.progress_infections!(world)
    end
    @test count_sugar_components(world, (SugarModel.Infection,)) == 0
    @test Ark.get_resource(world, SugarModel.StepEvents).recoveries == 4
    final_sugar = Dict{Int64, Int64}()
    for (entities, ids, sugars) in Ark.Query(world, (SugarModel.CitizenId, SugarModel.Sugar))
        @inbounds for i in eachindex(entities)
            final_sugar[ids[i].val] = sugars[i].val
        end
    end
    @test sum(values(initial_sugar)) - sum(values(final_sugar)) == 8
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

@testset "Buffered destination selection matches reference semantics" begin
    for width in 1:4, height in 1:4
        params = SugarModel.ModelParams(width = width, height = height, population = 0)
        capacity = [Int64(mod(3x + 5y, 4)) for x in 1:width, y in 1:height]
        landscape = SugarModel.SugarLandscape(copy(capacity), capacity)
        occupancy = SugarModel.OccupancyGrid(zeros(Int64, width, height))
        buffers = SugarModel.SimulationBuffers(params)

        for x in 1:width, y in 1:height, vision in 0:10, seed in 1:5
            position = SugarModel.Position(x, y)
            reference_rng = Random.Xoshiro(seed)
            buffered_rng = Random.Xoshiro(seed)
            reference = SugarModel.select_destination(
                position,
                vision,
                1,
                landscape,
                occupancy,
                params,
                reference_rng,
            )
            buffered = SugarModel.select_destination(
                position,
                vision,
                1,
                landscape,
                occupancy,
                params,
                buffered_rng,
                buffers,
            )

            @test buffered == reference
            @test rand(buffered_rng, UInt64) == rand(reference_rng, UInt64)
        end
    end
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
    distribution = SugarModel.wealth_statistics([0, 10, 20, 30])
    @test distribution.mean_wealth == 15.0
    @test distribution.median_wealth == 15.0
    @test distribution.wealth_p25 == 7.5
    @test distribution.wealth_p75 == 22.5
    @test distribution.bottom_50_share ≈ 1 / 6
    @test distribution.top_10_share == 0.5
    @test distribution.gini ≈ 5 / 12
    lorenz = SugarModel.lorenz_curve([0, 10, 20, 30])
    @test lorenz.population_share == [0.0, 0.25, 0.5, 0.75, 1.0]
    @test lorenz.wealth_share == [0.0, 0.0, 1 / 6, 0.5, 1.0]
    @test_throws ArgumentError SugarModel.lorenz_curve([-1, 1])

    params = SugarModel.ModelParams(width = 8, height = 8, population = 20)
    world = SugarModel.run_model(SugarModel.ModelArgs(seed = 31, params = params, steps = 5))
    logger = Ark.get_resource(world, SugarModel.Logger)
    statistics = SugarModel.summary_statistics(world; burn_in = 1)
    @test statistics.period == 5
    @test statistics.population == params.population
    @test statistics.females + statistics.males == statistics.population
    @test statistics.total_agent_sugar == logger.total_agent_sugar[end]
    @test statistics.gini ≈ logger.gini[end]
    @test statistics.cumulative_deaths == sum(logger.deaths)
    @test_throws ArgumentError SugarModel.summary_statistics(world; burn_in = -1)
    @test !isnothing(SugarModel.plot_model_diagnostics(logger))
    @test !isnothing(SugarModel.plot_population_dynamics(logger))
    @test !isnothing(SugarModel.plot_sugarscape(world))
    @test !isnothing(SugarModel.plot_wealth_distribution(world))
    @test_throws ArgumentError SugarModel.plot_model_diagnostics(SugarModel.Logger())
    @test_throws ArgumentError SugarModel.plot_population_dynamics(SugarModel.Logger())

    mktempdir() do directory
        path = joinpath(directory, "statistics.tsv")
        @test SugarModel.write_summary_statistics(path, statistics) == path
        contents = read(path, String)
        @test startswith(contents, "metric\tvalue\n")
        @test occursin("gini\t", contents)
    end

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

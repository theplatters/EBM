function spawn_citizen!(
    world,
    position::Position;
    vision = nothing,
    metabolism = nothing,
    sugar = nothing,
    maximum_age = nothing,
    initial_endowment = nothing,
    sex = nothing,
    immune_genotype = nothing,
    immune_bits = nothing,
)
    params = Ark.get_resource(world, ModelParams)
    rng = simulation_rng(world)
    next_id = Ark.get_resource(world, NextCitizenId)
    citizen_id = next_id.val
    next_id.val += 1
    citizen_sugar = isnothing(sugar) ?
                    rand(rng, params.minimum_initial_sugar:params.maximum_initial_sugar) : sugar
    endowment = isnothing(initial_endowment) ? citizen_sugar : initial_endowment
    sex_component = isnothing(sex) ? (rand(rng, Bool) ? Female() : Male()) : sex
    citizen_vision =
        isnothing(vision) ? rand(rng, params.minimum_vision:params.maximum_vision) : vision
    citizen_metabolism = isnothing(metabolism) ?
                         rand(rng, params.minimum_metabolism:params.maximum_metabolism) :
                         metabolism
    citizen_maximum_age = isnothing(maximum_age) ?
                          rand(rng, params.minimum_lifespan:params.maximum_lifespan) :
                          maximum_age
    genotype = if isnothing(immune_genotype)
        rand(rng, UInt64) & bit_mask(params.immune_system_length)
    else
        UInt64(immune_genotype) & bit_mask(params.immune_system_length)
    end
    phenotype = isnothing(immune_bits) ? genotype :
                UInt64(immune_bits) & bit_mask(params.immune_system_length)
    bundle = (
        CitizenId(citizen_id),
        position,
        ProposedPosition(position),
        Vision(citizen_vision),
        Metabolism(citizen_metabolism),
        Sugar(citizen_sugar),
        Age(0),
        MaximumAge(citizen_maximum_age),
        InitialEndowment(endowment),
        ImmuneProfile(genotype, phenotype),
        sex_component,
    )
    Ark.new_entity!(world, bundle)
    return citizen_id
end

function initial_positions(params::ModelParams, rng)
    cell_indices = randperm(rng, params.width * params.height)[1:params.population]
    return [
        Position(mod1(index, params.width), (index - 1) ÷ params.width + 1)
        for index in cell_indices
    ]
end

function spawn_initial_population!(world)
    params = Ark.get_resource(world, ModelParams)
    rng = simulation_rng(world)
    for (index, position) in enumerate(initial_positions(params, rng))
        sex = isodd(index) ? Female() : Male()
        spawn_citizen!(world, position; sex = sex)
    end
    return nothing
end

function seed_initial_infections!(world)
    params = Ark.get_resource(world, ModelParams)
    count = params.initial_diseases_per_citizen
    iszero(count) && return nothing
    catalog = Ark.get_resource(world, DiseaseCatalog)
    rng = simulation_rng(world)
    candidates = Tuple{Int64, Ark.Entity, UInt64}[]
    for (entities, ids, immunity) in Query(world, (CitizenId, ImmuneProfile))
        @inbounds for i in eachindex(entities)
            push!(candidates, (ids[i].val, entities[i], immunity[i].phenotype))
        end
    end
    sort!(candidates; by = first)
    for (_, entity, immune_bits) in candidates
        diseases = initial_disease_mask(
            rng,
            catalog,
            count,
            immune_bits,
            params.immune_system_length,
        )
        iszero(diseases) || Ark.add_components!(world, entity, (Infection(diseases),))
    end
    return nothing
end

function rebuild_occupancy!(world)
    occupancy = Ark.get_resource(world, OccupancyGrid)
    fill!(occupancy.citizen_ids, 0)
    for (entities, ids, positions) in Query(world, (CitizenId, Position))
        @inbounds for i in eachindex(entities)
            position = positions[i]
            occupancy.citizen_ids[position.x, position.y] == 0 ||
                throw(ArgumentError("multiple citizens occupy $(position)"))
            occupancy.citizen_ids[position.x, position.y] = ids[i].val
        end
    end
    return nothing
end

function setup_world(args::ModelArgs = ModelArgs())
    validate(args)
    world = Ark.World(
        CitizenId,
        Position,
        ProposedPosition,
        Vision,
        Metabolism,
        Sugar,
        Age,
        MaximumAge,
        InitialEndowment,
        Female,
        Male,
        ImmuneProfile,
        Infection,
    )
    setup_resources!(world, args)
    spawn_initial_population!(world)
    seed_initial_infections!(world)
    rebuild_occupancy!(world)
    return world
end

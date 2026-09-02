bit_mask(length::Integer) =
    length == 64 ? typemax(UInt64) : (UInt64(1) << length) - UInt64(1)

function immune_to(immune_bits::UInt64, disease::Disease, immune_length::Integer)
    disease_mask = bit_mask(disease.length)
    for offset in 0:(immune_length - disease.length)
        ((immune_bits >> offset) & disease_mask) == disease.bits && return true
    end
    return false
end

immune_to(immune_bits::Integer, disease::Disease, immune_length::Integer) =
    immune_to(UInt64(immune_bits), disease, immune_length)

function train_immunity(
    immune_bits::UInt64,
    disease::Disease,
    immune_length::Integer,
)
    immune_to(immune_bits, disease, immune_length) && return immune_bits
    disease_mask = bit_mask(disease.length)
    closest_offset = 0
    closest_distance = typemax(Int64)
    closest_difference = UInt64(0)
    for offset in 0:(immune_length - disease.length)
        difference = xor((immune_bits >> offset) & disease_mask, disease.bits)
        distance = count_ones(difference)
        if distance < closest_distance
            closest_offset = offset
            closest_distance = distance
            closest_difference = difference
        end
    end
    differing_bit = trailing_zeros(closest_difference)
    target_position = closest_offset + differing_bit
    target_mask = UInt64(1) << target_position
    trained = iszero(disease.bits & (UInt64(1) << differing_bit)) ?
              immune_bits & ~target_mask : immune_bits | target_mask
    return trained & bit_mask(immune_length)
end

function inherit_immune_genotype(
    rng,
    mother_genotype::UInt64,
    father_genotype::UInt64,
    immune_length::Integer,
)
    differing = xor(mother_genotype, father_genotype)
    choices = rand(rng, UInt64)
    inherited = (mother_genotype & ~differing) | (choices & differing)
    return inherited & bit_mask(immune_length)
end

function generate_disease_catalog(rng, params::ModelParams)
    diseases = Disease[]
    sizehint!(diseases, params.disease_catalog_size)
    while length(diseases) < params.disease_catalog_size
        length = rand(rng, params.minimum_disease_length:params.maximum_disease_length)
        disease = Disease(rand(rng, UInt64), length)
        disease in diseases || push!(diseases, disease)
    end
    return diseases
end

function initial_disease_mask(
    rng,
    catalog::DiseaseCatalog,
    count::Integer,
    immune_bits::UInt64,
    immune_length::Integer,
)
    iszero(count) && return UInt64(0)
    indices = randperm(rng, length(catalog.diseases))
    diseases = UInt64(0)
    for index in @view indices[1:count]
        immune_to(immune_bits, catalog.diseases[index], immune_length) && continue
        diseases |= UInt64(1) << (index - 1)
    end
    return diseases
end

function random_disease_bit(rng, diseases::UInt64)
    ordinal = rand(rng, 1:count_ones(diseases))
    remaining = diseases
    for _ in 2:ordinal
        remaining &= remaining - UInt64(1)
    end
    return UInt64(1) << trailing_zeros(remaining)
end

function neighboring_positions(position::Position, params::ModelParams)
    neighbors = Position[
        Position(mod1(position.x + 1, params.width), position.y),
        Position(mod1(position.x - 1, params.width), position.y),
        Position(position.x, mod1(position.y + 1, params.height)),
        Position(position.x, mod1(position.y - 1, params.height)),
    ]
    unique!(neighbors)
    filter!(!=(position), neighbors)
    sort!(neighbors; by = neighbor -> (neighbor.x, neighbor.y))
    return neighbors
end

function transmit_disease!(world)
    params = Ark.get_resource(world, ModelParams)
    catalog = Ark.get_resource(world, DiseaseCatalog)
    isempty(catalog.diseases) && return nothing
    buffers = Ark.get_resource(world, SimulationBuffers)
    infectious = buffers.infectious
    empty!(infectious)
    for (entities, positions, infections) in Query(world, (Position, Infection))
        @inbounds for i in eachindex(entities)
            infectious[positions[i]] = infections[i].diseases
        end
    end
    isempty(infectious) && return nothing

    citizens = Tuple{Int64, Ark.Entity, Position, UInt64}[]
    for (entities, ids, positions, immunity) in
        Query(world, (CitizenId, Position, ImmuneProfile))
        @inbounds for i in eachindex(entities)
            push!(citizens, (ids[i].val, entities[i], positions[i], immunity[i].phenotype))
        end
    end
    sort!(citizens; by = first)
    rng = simulation_rng(world)
    new_infections = buffers.new_infections
    empty!(new_infections)
    for (_, entity, position, immune_bits) in citizens
        current = if Ark.has_components(world, entity, (Infection,))
            Ark.get_components(world, entity, (Infection,))[1].diseases
        else
            UInt64(0)
        end
        additions = UInt64(0)
        for neighbor in neighboring_positions(position, params)
            donor_diseases = get(infectious, neighbor, UInt64(0))
            iszero(donor_diseases) && continue
            disease_bit = random_disease_bit(rng, donor_diseases)
            iszero((current | additions) & disease_bit) || continue
            disease_index = trailing_zeros(disease_bit) + 1
            immune_to(immune_bits, catalog.diseases[disease_index], params.immune_system_length) &&
                continue
            additions |= disease_bit
        end
        iszero(additions) || push!(new_infections, entity => additions)
    end

    events = Ark.get_resource(world, StepEvents)
    for (entity, additions) in new_infections
        if Ark.has_components(world, entity, (Infection,))
            current = Ark.get_components(world, entity, (Infection,))[1].diseases
            Ark.set_components!(world, entity, (Infection(current | additions),))
        else
            Ark.add_components!(world, entity, (Infection(additions),))
        end
        events.infections += count_ones(additions)
    end
    return nothing
end

function progress_infections!(world)
    params = Ark.get_resource(world, ModelParams)
    catalog = Ark.get_resource(world, DiseaseCatalog)
    recovered_entities = Ark.Entity[]
    recovered_diseases = 0
    for (entities, infections, immunity, sugars) in
        Query(world, (Infection, ImmuneProfile, Sugar))
        @inbounds for i in eachindex(entities)
            diseases = infections[i].diseases
            starting_immunity = immunity[i].phenotype
            disease_cost = 0
            for index in 1:length(catalog.diseases)
                disease_bit = UInt64(1) << (index - 1)
                iszero(diseases & disease_bit) && continue
                immune_to(starting_immunity, catalog.diseases[index], params.immune_system_length) ||
                    (disease_cost += params.disease_sugar_cost)
            end
            sugars[i] = Sugar(sugars[i].val - disease_cost)

            trained_immunity = starting_immunity
            remaining = diseases
            for index in 1:length(catalog.diseases)
                disease_bit = UInt64(1) << (index - 1)
                iszero(diseases & disease_bit) && continue
                disease = catalog.diseases[index]
                if !immune_to(trained_immunity, disease, params.immune_system_length)
                    trained_immunity =
                        train_immunity(trained_immunity, disease, params.immune_system_length)
                end
                if immune_to(trained_immunity, disease, params.immune_system_length)
                    remaining &= ~disease_bit
                    recovered_diseases += 1
                end
            end
            immunity[i] = ImmuneProfile(immunity[i].genotype, trained_immunity)
            infections[i] = Infection(remaining)
            iszero(remaining) && push!(recovered_entities, entities[i])
        end
    end
    for entity in recovered_entities
        Ark.remove_components!(world, entity, (Infection,))
    end
    Ark.get_resource(world, StepEvents).recoveries += recovered_diseases
    return nothing
end

function disease!(world)
    transmit_disease!(world)
    progress_infections!(world)
    return nothing
end

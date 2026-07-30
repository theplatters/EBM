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
    transmission_probability = params.disease_transmission_probability
    transmission_probability == 0.0 && return nothing
    infectious = Dict{Position, UInt64}()
    for (entities, positions, infections) in Query(world, (Position, Infection))
        @inbounds for i in eachindex(entities)
            infectious[positions[i]] = infections[i].strain
        end
    end
    isempty(infectious) && return nothing

    susceptible = Tuple{Int64, Ark.Entity, Position, UInt64}[]
    for (entities, ids, positions, immunity) in
        Query(world, (CitizenId, Position, ImmuneProfile))
        @inbounds for i in eachindex(entities)
            Ark.has_components(world, entities[i], (Infection,)) && continue
            push!(susceptible, (ids[i].val, entities[i], positions[i], immunity[i].bits))
        end
    end
    sort!(susceptible; by = first)
    rng = simulation_rng(world)
    new_infections = Pair{Ark.Entity, Infection}[]
    for (_, entity, position, immune_bits) in susceptible
        for neighbor in neighboring_positions(position, params)
            strain = get(infectious, neighbor, nothing)
            isnothing(strain) && continue
            strain == immune_bits && continue
            if rand(rng) < transmission_probability
                push!(new_infections, entity => Infection(strain, 0))
                break
            end
        end
    end
    events = Ark.get_resource(world, StepEvents)
    for (entity, infection) in new_infections
        Ark.add_components!(world, entity, (infection,))
        events.infections += 1
    end
    return nothing
end

function progress_infections!(world)
    params = Ark.get_resource(world, ModelParams)
    recovered = Ark.Entity[]
    for (entities, infections, immunity, sugars) in
        Query(world, (Infection, ImmuneProfile, Sugar))
        @inbounds for i in eachindex(entities)
            infection_age = infections[i].age + 1
            sugars[i] = Sugar(sugars[i].val - params.disease_sugar_cost)
            if infection_age >= params.disease_duration
                immunity[i] = ImmuneProfile(infections[i].strain)
                push!(recovered, entities[i])
            else
                infections[i] = Infection(infections[i].strain, infection_age)
            end
        end
    end
    events = Ark.get_resource(world, StepEvents)
    for entity in recovered
        Ark.remove_components!(world, entity, (Infection,))
        events.recoveries += 1
    end
    return nothing
end

function disease!(world)
    transmit_disease!(world)
    progress_infections!(world)
    return nothing
end

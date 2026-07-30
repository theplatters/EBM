struct MovementRecord
    id::Int64
    position::Position
    vision::Int64
    sugar::Int64
end

function visible_cells(position::Position, vision::Integer, params::ModelParams)
    distances = Dict{Position, Int64}(position => 0)
    for distance in 1:vision
        candidates = (
            Position(mod1(position.x + distance, params.width), position.y),
            Position(mod1(position.x - distance, params.width), position.y),
            Position(position.x, mod1(position.y + distance, params.height)),
            Position(position.x, mod1(position.y - distance, params.height)),
        )
        for candidate in candidates
            distances[candidate] = min(get(distances, candidate, typemax(Int64)), distance)
        end
    end
    return distances
end

function select_destination(
    position::Position,
    vision::Integer,
    citizen_id::Integer,
    landscape::SugarLandscape,
    occupancy::OccupancyGrid,
    params::ModelParams,
    rng,
)
    best_sugar = typemin(Int64)
    best_distance = typemax(Int64)
    best = Position[]
    cells = collect(visible_cells(position, vision, params))
    sort!(cells; by = pair -> (last(pair), first(pair).x, first(pair).y))
    for (candidate, distance) in cells
        occupant = occupancy.citizen_ids[candidate.x, candidate.y]
        (occupant == 0 || occupant == citizen_id) || continue
        patch_sugar = landscape.current[candidate.x, candidate.y]
        if patch_sugar > best_sugar ||
                (patch_sugar == best_sugar && distance < best_distance)
            best_sugar = patch_sugar
            best_distance = distance
            empty!(best)
            push!(best, candidate)
        elseif patch_sugar == best_sugar && distance == best_distance
            push!(best, candidate)
        end
    end
    isempty(best) && return position
    return best[rand(rng, eachindex(best))]
end

function movement_records(world)
    records = MovementRecord[]
    for (entities, ids, positions, visions, sugars) in
        Query(world, (CitizenId, Position, Vision, Sugar))
        @inbounds for i in eachindex(entities)
            push!(
                records,
                MovementRecord(ids[i].val, positions[i], visions[i].val, sugars[i].val),
            )
        end
    end
    sort!(records; by = record -> record.id)
    return records
end

function apply_movement!(world, destinations, wealth)
    for (entities, ids, positions, proposals, sugars) in
        Query(world, (CitizenId, Position, ProposedPosition, Sugar))
        @inbounds for i in eachindex(entities)
            id = ids[i].val
            destination = destinations[id]
            positions[i] = destination
            proposals[i] = ProposedPosition(destination)
            sugars[i] = Sugar(wealth[id])
        end
    end
    return nothing
end

function move_and_harvest_sequential!(world)
    params = Ark.get_resource(world, ModelParams)
    landscape = Ark.get_resource(world, SugarLandscape)
    occupancy = Ark.get_resource(world, OccupancyGrid)
    events = Ark.get_resource(world, StepEvents)
    rng = simulation_rng(world)
    records = movement_records(world)
    shuffle!(rng, records)
    destinations = Dict{Int64, Position}()
    wealth = Dict{Int64, Int64}()

    for record in records
        occupancy.citizen_ids[record.position.x, record.position.y] = 0
        destination = select_destination(
            record.position,
            record.vision,
            record.id,
            landscape,
            occupancy,
            params,
            rng,
        )
        occupancy.citizen_ids[destination.x, destination.y] = record.id
        harvest = landscape.current[destination.x, destination.y]
        landscape.current[destination.x, destination.y] = 0
        destinations[record.id] = destination
        wealth[record.id] = record.sugar + harvest
        events.moved += destination != record.position
        events.harvested += harvest
    end
    apply_movement!(world, destinations, wealth)
    return nothing
end

function plan_movements!(world)
    params = Ark.get_resource(world, ModelParams)
    landscape = Ark.get_resource(world, SugarLandscape)
    occupancy = Ark.get_resource(world, OccupancyGrid)
    rng = simulation_rng(world)
    destinations = Dict{Int64, Position}()
    for record in movement_records(world)
        destinations[record.id] = select_destination(
            record.position,
            record.vision,
            record.id,
            landscape,
            occupancy,
            params,
            rng,
        )
    end
    for (entities, ids, proposals) in Query(world, (CitizenId, ProposedPosition))
        @inbounds for i in eachindex(entities)
            proposals[i] = ProposedPosition(destinations[ids[i].val])
        end
    end
    return nothing
end

function resolve_and_commit_movements!(world)
    landscape = Ark.get_resource(world, SugarLandscape)
    events = Ark.get_resource(world, StepEvents)
    rng = simulation_rng(world)
    origins = Dict{Int64, Position}()
    proposals = Dict{Position, Vector{Int64}}()
    sugar_by_id = Dict{Int64, Int64}()

    for (entities, ids, positions, proposed, sugars) in
        Query(world, (CitizenId, Position, ProposedPosition, Sugar))
        @inbounds for i in eachindex(entities)
            id = ids[i].val
            origins[id] = positions[i]
            sugar_by_id[id] = sugars[i].val
            push!(get!(proposals, Position(proposed[i]), Int64[]), id)
        end
    end

    destinations = copy(origins)
    proposal_destinations = sort!(
        collect(keys(proposals));
        by = position -> (position.x, position.y),
    )
    for destination in proposal_destinations
        contenders = proposals[destination]
        sort!(contenders)
        if length(contenders) == 1
            destinations[only(contenders)] = destination
        else
            winner = contenders[rand(rng, eachindex(contenders))]
            destinations[winner] = destination
            events.conflicts += length(contenders) - 1
        end
    end

    wealth = Dict{Int64, Int64}()
    for (id, destination) in destinations
        harvest = landscape.current[destination.x, destination.y]
        landscape.current[destination.x, destination.y] = 0
        wealth[id] = sugar_by_id[id] + harvest
        events.moved += destination != origins[id]
        events.harvested += harvest
    end
    apply_movement!(world, destinations, wealth)
    rebuild_occupancy!(world)
    return nothing
end

function move_and_harvest_synchronously!(world)
    plan_movements!(world)
    resolve_and_commit_movements!(world)
    return nothing
end

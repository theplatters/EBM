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

@inline function position_isless(first::Position, second::Position)
    return first.x < second.x || (first.x == second.x && first.y < second.y)
end

@inline function ordered_positions(first::Position, second::Position)
    return position_isless(second, first) ? (second, first) : (first, second)
end

@inline function sort_four_positions(first, second, third, fourth)
    first, second = ordered_positions(first, second)
    third, fourth = ordered_positions(third, fourth)
    first, third = ordered_positions(first, third)
    second, fourth = ordered_positions(second, fourth)
    second, third = ordered_positions(second, third)
    return first, second, third, fourth
end

@inline function visible_at_distance(
    candidate::Position,
    position::Position,
    distance::Integer,
    params::ModelParams,
)
    minimum_distance = typemax(Int64)
    if candidate.y == position.y
        positive = mod(candidate.x - position.x, params.width)
        negative = mod(position.x - candidate.x, params.width)
        minimum_distance = min(minimum_distance, positive, negative)
    end
    if candidate.x == position.x
        positive = mod(candidate.y - position.y, params.height)
        negative = mod(position.y - candidate.y, params.height)
        minimum_distance = min(minimum_distance, positive, negative)
    end
    return minimum_distance == distance
end

@inline function score_destination_candidate!(
    ties::Vector{Position},
    tie_offset::Int,
    tie_stride::Int,
    candidate::Position,
    distance::Int64,
    citizen_id::Integer,
    landscape::SugarLandscape,
    occupancy::OccupancyGrid,
    best_sugar::Int64,
    best_distance::Int64,
    best_count::Int,
)
    occupant = occupancy.citizen_ids[candidate.x, candidate.y]
    (occupant == 0 || occupant == citizen_id) ||
        return best_sugar, best_distance, best_count
    patch_sugar = landscape.current[candidate.x, candidate.y]
    if patch_sugar > best_sugar ||
            (patch_sugar == best_sugar && distance < best_distance)
        @inbounds ties[tie_offset + 1] = candidate
        return patch_sugar, distance, 1
    elseif patch_sugar == best_sugar && distance == best_distance
        best_count += 1
        @boundscheck best_count <= tie_stride ||
            throw(BoundsError(ties, tie_offset + best_count))
        @inbounds ties[tie_offset + best_count] = candidate
    end
    return best_sugar, best_distance, best_count
end

function score_destinations!(
    ties::Vector{Position},
    tie_offset::Int,
    tie_stride::Int,
    position::Position,
    vision::Integer,
    citizen_id::Integer,
    landscape::SugarLandscape,
    occupancy::OccupancyGrid,
    params::ModelParams,
)
    best_sugar = typemin(Int64)
    best_distance = typemax(Int64)
    best_count = 0
    best_sugar, best_distance, best_count = score_destination_candidate!(
        ties,
        tie_offset,
        tie_stride,
        position,
        0,
        citizen_id,
        landscape,
        occupancy,
        best_sugar,
        best_distance,
        best_count,
    )

    for distance in 1:vision
        first, second, third, fourth = sort_four_positions(
            Position(mod1(position.x + distance, params.width), position.y),
            Position(mod1(position.x - distance, params.width), position.y),
            Position(position.x, mod1(position.y + distance, params.height)),
            Position(position.x, mod1(position.y - distance, params.height)),
        )
        previous = position
        for candidate in (first, second, third, fourth)
            candidate == previous && continue
            previous = candidate
            visible_at_distance(candidate, position, distance, params) || continue
            best_sugar, best_distance, best_count = score_destination_candidate!(
                ties,
                tie_offset,
                tie_stride,
                candidate,
                Int64(distance),
                citizen_id,
                landscape,
                occupancy,
                best_sugar,
                best_distance,
                best_count,
            )
        end
    end
    if iszero(best_count)
        @inbounds ties[tie_offset + 1] = position
        return 1
    end
    return best_count
end

function select_destination(
    position::Position,
    vision::Integer,
    citizen_id::Integer,
    landscape::SugarLandscape,
    occupancy::OccupancyGrid,
    params::ModelParams,
    rng,
    buffers::SimulationBuffers,
)
    stride = buffers.destination_tie_stride
    length(buffers.destination_ties) < stride &&
        resize!(buffers.destination_ties, stride)
    best_count = score_destinations!(
        buffers.destination_ties,
        0,
        stride,
        position,
        vision,
        citizen_id,
        landscape,
        occupancy,
        params,
    )
    candidate_index = rand(rng, Base.OneTo(best_count))
    return @inbounds buffers.destination_ties[candidate_index]
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

function movement_records!(records::Vector{MovementRecord}, world)
    empty!(records)
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


movement_records(world) = movement_records!(MovementRecord[], world)

function ensure_citizen_capacity!(buffers::SimulationBuffers, citizen_id::Integer)
    required = Int(citizen_id)
    length(buffers.origins) < required && resize!(buffers.origins, required)
    length(buffers.destinations) < required && resize!(buffers.destinations, required)
    length(buffers.wealth_by_id) < required && resize!(buffers.wealth_by_id, required)
    return nothing
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
    buffers = Ark.get_resource(world, SimulationBuffers)
    records = movement_records!(buffers.movement_records, world)
    shuffle!(rng, records)
    destinations = buffers.destinations
    wealth = buffers.wealth_by_id

    for record in records
        ensure_citizen_capacity!(buffers, record.id)
        occupancy.citizen_ids[record.position.x, record.position.y] = 0
        destination = select_destination(
            record.position,
            record.vision,
            record.id,
            landscape,
            occupancy,
            params,
            rng,
            buffers,
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
    buffers = Ark.get_resource(world, SimulationBuffers)
    destinations = buffers.destinations
    records = movement_records!(buffers.movement_records, world)
    config = Ark.get_resource(world, ExecutionConfig)
    thread_movements = config.threaded && config.thread_count >= 4 &&
                       length(records) >= 2_000 &&
                       sum(record -> 1 + 4 * record.vision, records; init = 0) >= 20_000
    if thread_movements
        plan_movements_threaded!(world, records)
    else
        rng = simulation_rng(world)
        for record in records
            ensure_citizen_capacity!(buffers, record.id)
            destinations[record.id] = select_destination(
                record.position,
                record.vision,
                record.id,
                landscape,
                occupancy,
                params,
                rng,
                buffers,
            )
        end
    end
    for (entities, ids, proposals) in Query(world, (CitizenId, ProposedPosition))
        @inbounds for i in eachindex(entities)
            proposals[i] = ProposedPosition(destinations[ids[i].val])
        end
    end
    return nothing
end

function plan_movements_threaded!(world, records::Vector{MovementRecord})
    params = Ark.get_resource(world, ModelParams)
    landscape = Ark.get_resource(world, SugarLandscape)
    occupancy = Ark.get_resource(world, OccupancyGrid)
    buffers = Ark.get_resource(world, SimulationBuffers)
    count = length(records)
    stride = buffers.destination_tie_stride
    required_ties = count * stride
    length(buffers.destination_ties) < required_ties &&
        resize!(buffers.destination_ties, required_ties)
    length(buffers.destination_tie_counts) < count &&
        resize!(buffers.destination_tie_counts, count)

    Threads.@threads for record_index in eachindex(records)
        record = @inbounds records[record_index]
        offset = (record_index - 1) * stride
        best_count = score_destinations!(
            buffers.destination_ties,
            offset,
            stride,
            record.position,
            record.vision,
            record.id,
            landscape,
            occupancy,
            params,
        )
        @inbounds buffers.destination_tie_counts[record_index] = best_count
    end

    rng = simulation_rng(world)
    destinations = buffers.destinations
    for record_index in eachindex(records)
        record = @inbounds records[record_index]
        ensure_citizen_capacity!(buffers, record.id)
        best_count = @inbounds buffers.destination_tie_counts[record_index]
        candidate_index = rand(rng, Base.OneTo(best_count))
        offset = (record_index - 1) * stride
        @inbounds destinations[record.id] =
            buffers.destination_ties[offset + candidate_index]
    end
    return nothing
end

function resolve_and_commit_movements!(world)
    landscape = Ark.get_resource(world, SugarLandscape)
    events = Ark.get_resource(world, StepEvents)
    rng = simulation_rng(world)
    params = Ark.get_resource(world, ModelParams)
    buffers = Ark.get_resource(world, SimulationBuffers)
    origins = buffers.origins
    destinations = buffers.destinations
    sugar_by_id = buffers.wealth_by_id
    proposals = buffers.proposals
    for cell_index in buffers.touched_proposals
        empty!(proposals[cell_index])
    end
    empty!(buffers.touched_proposals)

    for (entities, ids, positions, proposed, sugars) in
        Query(world, (CitizenId, Position, ProposedPosition, Sugar))
        @inbounds for i in eachindex(entities)
            id = ids[i].val
            ensure_citizen_capacity!(buffers, id)
            origins[id] = positions[i]
            destinations[id] = positions[i]
            sugar_by_id[id] = sugars[i].val
            destination = Position(proposed[i])
            cell_index = destination.x + (destination.y - 1) * params.width
            contenders = proposals[cell_index]
            isempty(contenders) && push!(buffers.touched_proposals, cell_index)
            push!(contenders, id)
        end
    end

    for x in 1:params.width, y in 1:params.height
        destination = Position(x, y)
        contenders = proposals[x + (y - 1) * params.width]
        isempty(contenders) && continue
        sort!(contenders)
        if length(contenders) == 1
            destinations[only(contenders)] = destination
        else
            winner = contenders[rand(rng, eachindex(contenders))]
            destinations[winner] = destination
            events.conflicts += length(contenders) - 1
        end
    end

    for record in movement_records!(buffers.movement_records, world)
        id = record.id
        destination = destinations[id]
        harvest = landscape.current[destination.x, destination.y]
        landscape.current[destination.x, destination.y] = 0
        sugar_by_id[id] += harvest
        events.moved += destination != origins[id]
        events.harvested += harvest
    end
    apply_movement!(world, destinations, sugar_by_id)
    rebuild_occupancy!(world)
    return nothing
end

function move_and_harvest_synchronously!(world)
    plan_movements!(world)
    resolve_and_commit_movements!(world)
    return nothing
end

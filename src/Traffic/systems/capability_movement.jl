struct ObservedMotion
    entity::Ark.Entity
    position::Position
    direction::Direction
    speed::Int
end

@inline entity_order(entity) = (getfield(entity, :_id), getfield(entity, :_gen))

function proposed_path(position::Position, direction::Direction, lane::Int, speed::Int, ring::Ring)
    current = position
    return ntuple(3) do microstep
        if microstep <= speed
            current = Position(
                microstep == 1 ? lane : current.x,
                ahead_y(current.y, direction, 1, Int(ring.height)),
            )
        end
        current
    end
end

@inline function edges_cross(
        previous_a::Position, current_a::Position,
        previous_b::Position, current_b::Position,
    )
    current_a == previous_b && current_b == previous_a && return true
    return previous_a.y == previous_b.y &&
           current_a.y == current_b.y &&
           previous_a.x != previous_b.x &&
           current_a.x != current_b.x &&
           previous_a.x == current_b.x &&
           previous_b.x == current_a.x
end

function paths_conflict(start_a::Position, path_a::NTuple{3,Position},
                        start_b::Position, path_b::NTuple{3,Position})
    previous_a, previous_b = start_a, start_b
    @inbounds for microstep in 1:3
        current_a, current_b = path_a[microstep], path_b[microstep]
        (current_a == current_b ||
         edges_cross(previous_a, current_a, previous_b, current_b)) &&
            return true
        previous_a, previous_b = current_a, current_b
    end
    return false
end

function observed_motions(world)
    motions = ObservedMotion[]
    for (entities, positions, directions, speeds) in Query(world, (Position, Direction, Speed))
        @inbounds for index in eachindex(entities)
            push!(motions, ObservedMotion(
                entities[index], positions[index], directions[index], speeds[index].val,
            ))
        end
    end
    sort!(motions; by = motion -> entity_order(motion.entity))
    return motions
end

function observed_action_is_safe(entity, position, direction, lane, speed, motions, ring)
    candidate_path = proposed_path(position, direction, lane, speed, ring)
    for other in motions
        other.entity == entity && continue
        other_path = proposed_path(
            other.position, other.direction, other.position.x, other.speed, ring,
        )
        paths_conflict(position, candidate_path, other.position, other_path) && return false
    end
    return true
end

"""Choose speed on the LR-selected lane, using one danger decision per car.

Diagnostics count cars whose max-speed candidate is dangerous and, separately,
those that voluntarily accept that candidate. The unavoidable speed-one
fallback is never counted as voluntary danger acceptance.
"""
function propose_speeds!(world)
    ring = Ark.get_resource(world, Ring)
    rng = simulation_rng(world)
    diagnostics = Ark.get_resource(world, CapabilityTickDiagnostics)
    motions = observed_motions(world)
    decisions = Tuple{Ark.Entity,Position,Direction,Int,Float64,Int}[]
    for (entities, positions, directions, controls, risks, lane_proposals, speed_proposals, paths) in
        Query(
            world,
            (Position, Direction, SpeedAdjustment, RiskAversion, LaneProposal,
             SpeedProposal, MovementPath),
        )
        @inbounds for index in eachindex(entities)
            push!(decisions, (
                entities[index], positions[index], directions[index],
                controls[index].max_speed, risks[index].value, lane_proposals[index].lane,
            ))
        end
    end
    sort!(decisions; by = decision -> entity_order(decision[1]))
    chosen = Dict{Ark.Entity,Tuple{Int,MovementPath}}()
    for (entity, position, direction, max_speed, risk, lane) in decisions
        max_path_dangerous = !observed_action_is_safe(
            entity, position, direction, lane, max_speed, motions, ring,
        )
        accept_danger = rand(rng) > risk
        if max_path_dangerous
            diagnostics.dangerous_proposals += 1
        end

        chosen_speed = 1
        if !max_path_dangerous || accept_danger
            chosen_speed = max_speed
        else
            for speed in (max_speed - 1):-1:1
                if observed_action_is_safe(
                        entity, position, direction, lane, speed, motions, ring,
                    )
                    chosen_speed = speed
                    break
                end
            end
            # No stopping action exists: speed one is the unavoidable-risk fallback.
        end
        if max_path_dangerous && accept_danger
            diagnostics.accepted_dangerous_proposals += 1
        end
        diagnostics.proposed_speed_total += chosen_speed
        diagnostics.proposed_speed_count += 1
        chosen[entity] = (
            chosen_speed,
            MovementPath(proposed_path(position, direction, lane, chosen_speed, ring)),
        )
    end
    for (entities, speed_proposals, paths) in Query(world, (SpeedProposal, MovementPath))
        @inbounds for index in eachindex(entities)
            speed, path = chosen[entities[index]]
            speed_proposals[index] = SpeedProposal(speed)
            paths[index] = path
        end
    end
    return nothing
end

"""Resolve submitted paths synchronously and return replacement requests."""
function resolve_capability_movement!(world)
    entities = Ark.Entity[]
    starts = Dict{Ark.Entity,Position}()
    directions = Dict{Ark.Entity,Direction}()
    paths = Dict{Ark.Entity,NTuple{3,Position}}()
    proposed_speeds = Dict{Ark.Entity,Int}()
    for (batch, positions, dirs, speed_proposals, movement_paths) in
        Query(world, (Position, Direction, SpeedProposal, MovementPath))
        @inbounds for index in eachindex(batch)
            entity = batch[index]
            push!(entities, entity)
            starts[entity] = positions[index]
            directions[entity] = dirs[index]
            paths[entity] = movement_paths[index].positions
            proposed_speeds[entity] = speed_proposals[index].value
        end
    end
    sort!(entities; by = entity_order)
    active, killed = Set(entities), Set{Ark.Entity}()
    previous, final_positions = copy(starts), copy(starts)
    for microstep in 1:3
        active_entities = [entity for entity in entities if entity in active]
        current = Dict(entity => paths[entity][microstep] for entity in active_entities)
        occupants = Dict{Position,Vector{Ark.Entity}}()
        for entity in active_entities
            push!(get!(occupants, current[entity], Ark.Entity[]), entity)
        end
        newly_killed = Set{Ark.Entity}()
        for at_position in values(occupants)
            if length(at_position) > 1
                union!(newly_killed, at_position)
            end
        end
        if length(active_entities) >= 2
            for i in 1:(length(active_entities) - 1)
                for j in (i + 1):length(active_entities)
                    a, b = active_entities[i], active_entities[j]
                    if edges_cross(
                            previous[a], current[a], previous[b], current[b],
                        )
                        push!(newly_killed, a)
                        push!(newly_killed, b)
                    end
                end
            end
        end
        for entity in active_entities
            final_positions[entity] = current[entity]
            previous[entity] = current[entity]
        end
        union!(killed, newly_killed)
        setdiff!(active, newly_killed)
    end
    diagnostics = Ark.get_resource(world, CapabilityTickDiagnostics)
    for (entities_q, positions, speeds, steps) in Query(world, (Position, Speed, Step))
        @inbounds for index in eachindex(entities_q)
            entity = entities_q[index]
            if entity in killed
                continue
            end
            positions[index] = final_positions[entity]
            speeds[index] = Speed(proposed_speeds[entity])
            steps[index] = Step(steps[index].val + 1)
            diagnostics.realized_speed_total += proposed_speeds[entity]
            diagnostics.realized_speed_count += 1
        end
    end
    killed_entities = sort!(collect(killed); by = entity_order)
    survivor_entities = sort!(setdiff(entities, killed_entities); by = entity_order)
    for entity in survivor_entities
        push!(diagnostics.survivor_risks, Ark.get_components(world, entity, (RiskAversion,))[1].value)
    end
    replacements = CapabilityReplacementSpec[CapabilityReplacementSpec(directions[e]) for e in killed_entities]
    for entity in killed_entities
        Ark.remove_entity!(world, entity)
    end
    return replacements
end

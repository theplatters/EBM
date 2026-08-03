struct ObservedMotion
  entity::Ark.Entity
  position::Position
  direction::Direction
  speed::Int
end

@inline entity_order(entity) = (getfield(entity, :_id), getfield(entity, :_gen))

function proposed_path(
  position::Position, direction::Direction, lane::Int, speed::Int, ring::Ring,
)
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

function paths_conflict(
  start_a::Position, path_a::NTuple{3,Position},
  start_b::Position, path_b::NTuple{3,Position},
)
  previous_a = start_a
  previous_b = start_b
  @inbounds for microstep in 1:3
    current_a = path_a[microstep]
    current_b = path_b[microstep]
    (
      current_a == current_b ||
      edges_cross(previous_a, current_a, previous_b, current_b)
    ) && return true
    previous_a = current_a
    previous_b = current_b
  end
  return false
end

@inline function longitudinal_separation(first::Int, second::Int, ring_height::Int)
  clockwise = mod(first - second, ring_height)
  counterclockwise = mod(second - first, ring_height)
  return min(clockwise, counterclockwise)
end

function clearance_violated(
  path_a::NTuple{3,Position}, path_b::NTuple{3,Position},
  clearance::Int, ring_height::Int,
)
  clearance > 0 || return false
  @inbounds for microstep in 1:3
    first = path_a[microstep]
    second = path_b[microstep]
    first.x == second.x || continue
    longitudinal_separation(first.y, second.y, ring_height) <= clearance &&
      return true
  end
  return false
end

function observed_motions(world)
  motions = ObservedMotion[]
  for (entities, positions, directions, speeds) in Query(
    world, (Position, Direction, Speed),
  )
    @inbounds for index in eachindex(entities)
      push!(
        motions,
        ObservedMotion(
          entities[index], positions[index], directions[index], speeds[index].val,
        ),
      )
    end
  end
  sort!(motions; by=motion -> entity_order(motion.entity))
  return motions
end

function observed_action_is_safe(
  entity, position, direction, lane, speed, motions, ring, speed_clearance,
)
  candidate_path = proposed_path(position, direction, lane, speed, ring)
  clearance = ceil(Int, speed_clearance * (speed - 1))
  for other in motions
    other.entity == entity && continue
    other_path = proposed_path(
      other.position, other.direction, other.position.x, other.speed, ring,
    )
    (
      paths_conflict(position, candidate_path, other.position, other_path) ||
      clearance_violated(
        candidate_path, other_path, clearance, Int(ring.height),
      )
    ) && return false
  end
  return true
end

"""
Choose a locally safe lane-speed action from observable current motion.

With `prefer_lane_over_speed = false`, retain the historical speed-first rule:
try the fastest speed first and use the lane disposition only to break ties.
With `prefer_lane_over_speed = true`, exhaust the safe speeds on the disposed
lane before trying the other lane. `speed_clearance` adds a speed-dependent
headway to the exact path-conflict check. Other cars' private proposals remain
deliberately unavailable.
"""
function propose_speeds!(world)
  ring = Ark.get_resource(world, Ring)
  model = Ark.get_resource(world, CapabilityModel)
  motions = observed_motions(world)
  for (entities, positions, directions, controls, lane_proposals, speed_proposals, paths) in
      Query(
    world,
    (
      Position, Direction, SpeedAdjustment, LaneProposal,
      SpeedProposal, MovementPath,
    ),
  )
    @inbounds for index in eachindex(entities)
      entity = entities[index]
      position = positions[index]
      direction = directions[index]
      preferred_lane = lane_proposals[index].lane
      alternative_lane = preferred_lane == 1 ? 2 : 1
      chosen_lane = preferred_lane
      chosen_speed = 1
      found_safe_action = false

      if model.prefer_lane_over_speed
        for lane in (preferred_lane, alternative_lane)
          for speed in controls[index].max_speed:-1:1
            if observed_action_is_safe(
              entity, position, direction, lane, speed, motions, ring,
              model.speed_clearance,
            )
              chosen_lane = lane
              chosen_speed = speed
              found_safe_action = true
              break
            end
          end
          found_safe_action && break
        end
      else
        for speed in controls[index].max_speed:-1:1
          for lane in (preferred_lane, alternative_lane)
            if observed_action_is_safe(
              entity, position, direction, lane, speed, motions, ring,
              model.speed_clearance,
            )
              chosen_lane = lane
              chosen_speed = speed
              found_safe_action = true
              break
            end
          end
          found_safe_action && break
        end
      end

      lane_proposals[index] = LaneProposal(chosen_lane)
      speed_proposals[index] = SpeedProposal(chosen_speed)
      paths[index] = MovementPath(
        proposed_path(position, direction, chosen_lane, chosen_speed, ring),
      )
    end
  end
  return nothing
end

"""Resolve all submitted paths synchronously and return replacement requests."""
function resolve_capability_movement!(world)
  entities = Ark.Entity[]
  starts = Dict{Ark.Entity,Position}()
  directions = Dict{Ark.Entity,Direction}()
  paths = Dict{Ark.Entity,NTuple{3,Position}}()
  proposed_speeds = Dict{Ark.Entity,Int}()

  for (batch, positions, dirs, speed_proposals, movement_paths) in Query(
    world, (Position, Direction, SpeedProposal, MovementPath),
  )
    @inbounds for index in eachindex(batch)
      entity = batch[index]
      push!(entities, entity)
      starts[entity] = positions[index]
      directions[entity] = dirs[index]
      paths[entity] = movement_paths[index].positions
      proposed_speeds[entity] = speed_proposals[index].value
    end
  end
  sort!(entities; by=entity_order)

  active = Set(entities)
  killed = Set{Ark.Entity}()
  previous = copy(starts)
  final_positions = copy(starts)
  for microstep in 1:3
    active_entities = [entity for entity in entities if entity in active]
    current = Dict(
      entity => paths[entity][microstep] for entity in active_entities
    )
    occupants = Dict{Position,Vector{Ark.Entity}}()
    for entity in active_entities
      push!(get!(occupants, current[entity], Ark.Entity[]), entity)
    end

    newly_killed = Set{Ark.Entity}()
    for occupants_at_position in values(occupants)
      length(occupants_at_position) > 1 || continue
      union!(newly_killed, occupants_at_position)
    end
    if length(active_entities) >= 2
      for first_index in 1:(length(active_entities)-1)
        first_entity = active_entities[first_index]
        for second_index in (first_index+1):length(active_entities)
          second_entity = active_entities[second_index]
          edges_cross(
            previous[first_entity], current[first_entity],
            previous[second_entity], current[second_entity],
          ) || continue
          push!(newly_killed, first_entity)
          push!(newly_killed, second_entity)
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

  for (batch, positions, speeds, steps) in Query(
    world, (Position, Speed, Step),
  )
    @inbounds for index in eachindex(batch)
      entity = batch[index]
      entity in killed && continue
      positions[index] = final_positions[entity]
      speeds[index] = Speed(proposed_speeds[entity])
      steps[index] = Step(steps[index].val + 1)
    end
  end

  killed_entities = sort!(collect(killed); by=entity_order)
  replacements = CapabilityReplacementSpec[
    CapabilityReplacementSpec(directions[entity]) for entity in killed_entities
  ]
  for entity in killed_entities
    Ark.remove_entity!(world, entity)
  end
  return replacements
end

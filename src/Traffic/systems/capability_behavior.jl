"""Build each driver's observation solely from the committed start-of-tick state."""
function observe_capability_traffic!(world)
  occupancy = Ark.get_resource(world, Occupancy).grid
  ring = Ark.get_resource(world, Ring)
  params = Ark.get_resource(world, ModelParams)
  model = Ark.get_resource(world, CapabilityModel)
  success_traces = Ark.get_resource(world, SuccessfulDriverTrace).grid
  horizon = min(params.lookahead, Int(ring.height) - 1)

  for (entities, positions, directions, observations) in Query(
    world, (Position, Direction, LocalObservation),
  )
    @inbounds for index in eachindex(entities)
      position = positions[index]
      direction = directions[index]
      same_total = 0
      same_left = 0
      opposite_total = 0
      opposite_left = 0
      close_left = 0
      close_right = 0
      convention = 0.0
      convention_samples = 0
      success_trace = 0.0
      success_trace_samples = 0

      for distance in 1:horizon
        y = ahead_y(position.y, direction, distance, Int(ring.height))
        for lane in 1:Int(ring.width)
          trace = success_traces[lane, y]
          if abs(trace) >= SUCCESS_TRACE_CUTOFF
            success_trace += trace
            success_trace_samples += 1
          end
          other_direction = occupancy[lane, y]
          isnothing(other_direction) && continue
          left_relative = is_left_relative(lane, direction)
          if other_direction == direction
            same_total += 1
            same_left += left_relative
          else
            opposite_total += 1
            opposite_left += left_relative
          end
          if distance <= model.max_speed
            if left_relative
              close_left += 1
            else
              close_right += 1
            end
          end
          convention += relative_lane_sign(lane, other_direction)
          convention_samples += 1
        end
      end

      observations[index] = LocalObservation(
        same_total == 0 ? 0.5 : same_left / same_total,
        opposite_total == 0 ? 0.5 : opposite_left / opposite_total,
        close_left,
        close_right,
        convention_samples == 0 ? 0.0 : convention / convention_samples,
        convention_samples,
        success_trace_samples == 0 ? 0.0 : success_trace / success_trace_samples,
        success_trace_samples,
      )
    end
  end
  return nothing
end

function learn_conventions!(world)
  rng = simulation_rng(world)
  for (entities, observations, perceptions, conventions) in Query(
    world,
    (LocalObservation, ConventionPerception, PerceivedConvention),
  )
    @inbounds for index in eachindex(entities)
      observation = observations[index]
      old = conventions[index]
      if observation.convention_samples == 0
        continue
      end
      perception = perceptions[index]
      noisy_observation = clamp(
        observation.convention + perception.noise * randn(rng), -1.0, 1.0,
      )
      rate = perception.learning_rate
      conventions[index] = PerceivedConvention(
        clamp((1 - rate) * old.value + rate * noisy_observation, -1.0, 1.0),
        clamp((1 - rate) * old.confidence + rate, 0.0, 1.0),
      )
    end
  end
  return nothing
end

"""Build a persistent disposition from a history of local success traces."""
function learn_social_habits!(world)
  rng = simulation_rng(world)
  for (entities, observations, formations, habitus) in Query(
    world,
    (LocalObservation, SocialHabitFormation, SocialHabitus),
  )
    @inbounds for index in eachindex(entities)
      observation = observations[index]
      observation.success_trace_samples == 0 && continue
      formation = formations[index]
      noisy_trace = clamp(
        observation.success_trace + formation.noise * randn(rng), -1.0, 1.0,
      )
      rate = formation.learning_rate
      old = habitus[index].value
      habitus[index] = SocialHabitus(
        clamp((1 - rate) * old + rate * noisy_trace, -1.0, 1.0),
      )
    end
  end
  return nothing
end

function reset_lane_scores!(world)
  for (entities, scores) in Query(world, (LaneScore,))
    @inbounds for index in eachindex(entities)
      scores[index] = LaneScore(0.0)
    end
  end
  return nothing
end

function add_same_direction_response!(world)
  model = Ark.get_resource(world, CapabilityModel)
  weight = Ark.get_resource(world, Weights).wₛ
  for (entities, observations, responses, scores, steps) in Query(
    world, (LocalObservation, SameDirectionResponse, LaneScore, Step),
  )
    @inbounds for index in eachindex(entities)
      !isnothing(model.avoidance_disable_age) &&
          steps[index].val >= model.avoidance_disable_age && continue
      value = weight * responses[index].sensitivity *
              (2 * observations[index].same_left - 1)
      scores[index] = LaneScore(scores[index].value + value)
    end
  end
  return nothing
end

function add_opposite_direction_response!(world)
  model = Ark.get_resource(world, CapabilityModel)
  weight = Ark.get_resource(world, Weights).wₒ
  for (entities, observations, responses, scores, steps) in Query(
    world, (LocalObservation, OppositeDirectionResponse, LaneScore, Step),
  )
    @inbounds for index in eachindex(entities)
      !isnothing(model.avoidance_disable_age) &&
          steps[index].val >= model.avoidance_disable_age && continue
      value = -weight * responses[index].sensitivity *
              (2 * observations[index].opposite_left - 1)
      scores[index] = LaneScore(scores[index].value + value)
    end
  end
  return nothing
end

function add_near_field_avoidance!(world)
  model = Ark.get_resource(world, CapabilityModel)
  weight = Ark.get_resource(world, Weights).wₐ
  for (entities, observations, responses, scores, steps) in Query(
    world, (LocalObservation, NearFieldAvoidance, LaneScore, Step),
  )
    @inbounds for index in eachindex(entities)
      !isnothing(model.avoidance_disable_age) &&
          steps[index].val >= model.avoidance_disable_age && continue
      value = weight * responses[index].sensitivity *
              (observations[index].close_right - observations[index].close_left)
      scores[index] = LaneScore(scores[index].value + value)
    end
  end
  return nothing
end

function add_habit_response!(world)
  model = Ark.get_resource(world, CapabilityModel)
  for (entities, habits, habitus, scores) in Query(
    world, (HabitFormation, Habitus, LaneScore),
  )
    @inbounds for index in eachindex(entities)
      value = model.habit_weight * habits[index].disposition * habitus[index].val
      scores[index] = LaneScore(scores[index].value + value)
    end
  end
  return nothing
end

function add_social_habit_response!(world)
  model = Ark.get_resource(world, CapabilityModel)
  for (entities, formations, habitus, scores) in Query(
    world, (SocialHabitFormation, SocialHabitus, LaneScore),
  )
    @inbounds for index in eachindex(entities)
      value = model.social_habit_weight * formations[index].disposition *
              habitus[index].value
      scores[index] = LaneScore(scores[index].value + value)
    end
  end
  return nothing
end

function add_convention_response!(world)
  model = Ark.get_resource(world, CapabilityModel)
  for (entities, perceptions, scores) in Query(
    world, (PerceivedConvention, LaneScore),
  )
    @inbounds for index in eachindex(entities)
      perception = perceptions[index]
      value = model.convention_weight * perception.confidence * perception.value
      scores[index] = LaneScore(scores[index].value + value)
    end
  end
  return nothing
end

function propose_lanes!(world)
  params = Ark.get_resource(world, ModelParams)
  rng = simulation_rng(world)
  for (entities, positions, directions, scores, proposals, decisions) in Query(
    world, (Position, Direction, LaneScore, LaneProposal, LR),
  )
    @inbounds for index in eachindex(entities)
      score = scores[index].value
      go_left = if score > 0.0
        true
      elseif score < 0.0
        false
      else
        is_left_relative(positions[index].x, directions[index])
      end
      rand(rng) < params.ϵ && (go_left = !go_left)
      lane = directions[index] == Clockwise ? (go_left ? 1 : 2) : (go_left ? 2 : 1)
      # LaneProposal is the binding lane for this tick. Speed selection may
      # not revise the LR decision.
      proposals[index] = LaneProposal(lane)
      decisions[index] = LR(score)
    end
  end
  return nothing
end

function calculate_capability_proposals!(world)
  observe_capability_traffic!(world)
  learn_conventions!(world)
  learn_social_habits!(world)
  reset_lane_scores!(world)
  add_same_direction_response!(world)
  add_opposite_direction_response!(world)
  add_near_field_avoidance!(world)
  add_habit_response!(world)
  add_convention_response!(world)
  add_social_habit_response!(world)
  propose_lanes!(world)
  return nothing
end

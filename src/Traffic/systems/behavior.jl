function calculate_lr!(world)
    weights = Ark.get_resource(world, Weights)
    params = Ark.get_resource(world, ModelParams)
    ring = Ark.get_resource(world, Ring)

    occ = Ark.get_resource(world, PredictedOccupancy).grid
    for (e, pos, dir, s, o, a, hg, ha, lr) in Query(
            world,
            (Position, Direction, SSensitvity, OSensitvity, Avoidance, Habitgene, Habitus, LR)
        )
        @inbounds for i in eachindex(e)
            SL, OL, CL, CR = compute_observations(
                e[i],
                occ, pos[i], dir[i], ring, params.lookahead
            )

            s_val = weights.wₛ * s[i].val * (2 * SL - 1)
            o_val = -weights.wₒ * o[i].val * (2 * OL - 1)
            avoidance_val = weights.wₐ * a[i].val * (CR - CL)
            habit_strength = weights.wₕ * hg[i].val * ha[i].val

            lr[i] = LR(s_val + o_val + avoidance_val + habit_strength)
        end
    end

    return nothing
end

calculate_lr!(world, ::OccupancyStrategy) = calculate_lr!(world)

function calculate_lr!(world, ::HeterogeneousStrategy)
    weights = Ark.get_resource(world, Weights)
    params = Ark.get_resource(world, ModelParams)
    ring = Ark.get_resource(world, Ring)
    occupancy = Ark.get_resource(world, PredictedOccupancy).grid

    for (entities, positions, directions, same, opposite, avoidance, habitgene, habitus, strategies, lr) in Query(
            world,
            (
                Position, Direction, SSensitvity, OSensitvity, Avoidance,
                Habitgene, Habitus, DriverStrategy, LR,
            ),
        )
        @inbounds for i in eachindex(entities)
            current_score = decision_score(
                entities[i], occupancy, positions[i], directions[i], same[i].val,
                opposite[i].val, avoidance[i].val, habitgene[i].val, habitus[i].val,
                ring, params, weights,
            )
            if strategies[i].kind == TwoFrameNaiveKind
                advanced_position = predict_position(positions[i], directions[i], ring)
                advanced_score = decision_score(
                    entities[i], occupancy, advanced_position, directions[i], same[i].val,
                    opposite[i].val, avoidance[i].val, habitgene[i].val, habitus[i].val,
                    ring, params, weights,
                )
                agrees = sign(current_score) == sign(advanced_score)
                current_score = agrees ? (current_score + advanced_score) / 2 : 0.0
            end
            lr[i] = LR(current_score)
        end
    end
    return nothing
end

function calculate_lr!(world, ::TwoFrameNaiveStrategy)
    weights = Ark.get_resource(world, Weights)
    params = Ark.get_resource(world, ModelParams)
    ring = Ark.get_resource(world, Ring)
    occupancy = Ark.get_resource(world, PredictedOccupancy).grid

    for (entities, positions, directions, same, opposite, avoidance, habitgene, habitus, lr) in Query(
            world,
            (Position, Direction, SSensitvity, OSensitvity, Avoidance, Habitgene, Habitus, LR),
        )
        @inbounds for i in eachindex(entities)
            current_score = decision_score(
                entities[i], occupancy, positions[i], directions[i], same[i].val,
                opposite[i].val, avoidance[i].val, habitgene[i].val, habitus[i].val,
                ring, params, weights,
            )
            advanced_position = predict_position(positions[i], directions[i], ring)
            advanced_score = decision_score(
                entities[i], occupancy, advanced_position, directions[i], same[i].val,
                opposite[i].val, avoidance[i].val, habitgene[i].val, habitus[i].val,
                ring, params, weights,
            )
            agrees = sign(current_score) == sign(advanced_score)
            lr[i] = LR(agrees ? (current_score + advanced_score) / 2 : 0.0)
        end
    end
    return nothing
end

@inline function decision_score(
        entity,
        occupancy,
        position::Position,
        direction::Direction,
        same_sensitivity::Float64,
        opposite_sensitivity::Float64,
        avoidance::Float64,
        habitgene::Float64,
        habitus::Float64,
        ring::Ring,
        params::ModelParams,
        weights::Weights,
    )
    same_left, opposite_left, close_left, close_right = compute_observations(
        entity, occupancy, position, direction, ring, params.lookahead,
    )
    same_value = weights.wₛ * same_sensitivity * (2 * same_left - 1)
    opposite_value = -weights.wₒ * opposite_sensitivity * (2 * opposite_left - 1)
    avoidance_value = weights.wₐ * avoidance * (close_right - close_left)
    habit_strength = weights.wₕ * habitgene * habitus
    return same_value + opposite_value + avoidance_value + habit_strength
end

@inline function compute_observations(
        e,
        occ, pos::Position, dir::Direction, ring::Ring, lookahead::Int
    )
    h = Int(ring.height)

    same_total = 0
    same_left = 0
    opp_total = 0
    opp_left = 0

    CL = 0
    CR = 0

    @inbounds for d in 0:lookahead
        y = ahead_y(pos.y, dir, d, h)
        for x in 1:2
            for (dir_other, e_other, occ_weight) in occ[x, y]
                e_other == e && continue
                left_rel = is_left_relative(x, dir)

                if dir_other == dir
                    same_total += occ_weight
                    same_left += occ_weight * left_rel
                else
                    opp_total += occ_weight
                    opp_left += occ_weight * left_rel
                end

                if d ≤ 2
                    if left_rel
                        CL += occ_weight
                    else
                        CR += occ_weight
                    end
                end
            end
        end
    end

    SL = same_total == 0 ? 0.5 : same_left / same_total
    OL = opp_total == 0 ? 0.5 : opp_left / opp_total

    return SL, OL, CL, CR
end

function rebuild_predicted_occupancy!(world, ::PerEntityHabitusStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    empty!.(grid)

    ring = Ark.get_resource(world, Ring)

    for (e, pos, dir, habitus) in Query(world, (Position, Direction, Habitus))
        @inbounds for i in eachindex(e)

            pnext = predict_position(pos[i], dir[i], ring)

            h = habitus[i].val
            w = abs(h)

            # determine which lane is preferred
            prefer_lane1 = (dir[i] == Clockwise  && h > 0) || (dir[i] == Counterclockwise && h < 0)

            if prefer_lane1
                push!(grid[1, pnext.y], (dir[i], e[i], 0.5 + w / 2))
                push!(grid[2, pnext.y], (dir[i], e[i], 0.5 - w / 2))
            else
                push!(grid[2, pnext.y], (dir[i], e[i], 0.5 + w / 2))
                push!(grid[1, pnext.y], (dir[i], e[i], 0.5 - w / 2))
            end
        end
    end

    return occ
end

function rebuild_predicted_occupancy!(world, ::MeanHabitusStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    empty!.(grid)
    ring = Ark.get_resource(world, Ring)
    mean_habitus = Ark.get_resource(world, MeanHabitus)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring)
            push!(grid[pnext.x, pnext.y], (dir[i], e[i], 0.5 + mean_habitus.abs / 2))
            push!(grid[pnext.x == 1 ? 2 : 1, pnext.y], (dir[i], e[i], 0.5 - mean_habitus.abs / 2))
        end
    end

    return occ
end

function rebuild_predicted_occupancy!(world, ::NaiveStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    empty!.(grid)
    ring = Ark.get_resource(world, Ring)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring)
            push!(grid[pnext.x, pnext.y], (dir[i], e[i], 1.0))
        end
    end

    return occ
end

rebuild_predicted_occupancy!(world, ::TwoFrameNaiveStrategy) =
    rebuild_predicted_occupancy!(world, NaiveStrategy())

function rebuild_predicted_occupancy!(world, ::UnsureStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    empty!.(grid)
    ring = Ark.get_resource(world, Ring)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring)
            push!(grid[1, pnext.y], (dir[i], e[i], 0.5))
            push!(grid[2, pnext.y], (dir[i], e[i], 0.5))
        end
    end

    return occ
end
function rebuild_predicted_occupancy!(world, ::SwitchStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    empty!.(grid)
    ring = Ark.get_resource(world, Ring)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring)
            push!(grid[pnext.x == 1 ? 2 : 1, pnext.y], (dir[i], e[i], 1.0))
        end
    end

    return occ
end

function rebuild_predicted_occupancy!(world, ::RandomStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    empty!.(grid)
    ring = Ark.get_resource(world, Ring)
    rng = simulation_rng(world)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring)
            r = rand(rng)
            push!(grid[1, pnext.y], (dir[i], e[i], r))
            push!(grid[2, pnext.y], (dir[i], e[i], 1 - r))
        end
    end

    return occ
end

function rebuild_predicted_occupancy!(world, strategy::DecisionAwareStrategy)
    occupancy = Ark.get_resource(world, PredictedOccupancy)
    grid = occupancy.grid
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    weights = Ark.get_resource(world, Weights)

    entities = Ark.Entity[]
    positions = Position[]
    directions = Direction[]
    same_sensitivities = Float64[]
    opposite_sensitivities = Float64[]
    avoidances = Float64[]
    habitgenes = Float64[]
    habitus_values = Float64[]

    for (e, pos, dir, same, opposite, avoidance, habitgene, habitus) in Query(
            world,
            (Position, Direction, SSensitvity, OSensitvity, Avoidance, Habitgene, Habitus),
        )
        @inbounds for i in eachindex(e)
            push!(entities, e[i])
            push!(positions, pos[i])
            push!(directions, dir[i])
            push!(same_sensitivities, same[i].val)
            push!(opposite_sensitivities, opposite[i].val)
            push!(avoidances, avoidance[i].val)
            push!(habitgenes, habitgene[i].val)
            push!(habitus_values, habitus[i].val)
        end
    end

    lane_one_probability = [position.x == 1 ? 1.0 : 0.0 for position in positions]
    updated_probability = similar(lane_one_probability)

    for _ in 1:strategy.iterations
        write_probabilistic_forecast!(
            grid, entities, positions, directions, lane_one_probability, ring,
        )

        @inbounds for i in eachindex(entities)
            score = decision_score(
                entities[i], grid, positions[i], directions[i], same_sensitivities[i],
                opposite_sensitivities[i], avoidances[i], habitgenes[i], habitus_values[i],
                ring, params, weights,
            )
            go_left = if score > 0.0
                true
            elseif score < 0.0
                false
            else
                is_left_relative(positions[i].x, directions[i])
            end
            preferred_lane = directions[i] == Clockwise ? (go_left ? 1 : 2) : (go_left ? 2 : 1)
            desired_probability = preferred_lane == 1 ? 1.0 - params.ϵ : params.ϵ
            updated_probability[i] = lane_one_probability[i] +
                                     strategy.damping * (desired_probability - lane_one_probability[i])
        end
        lane_one_probability, updated_probability = updated_probability, lane_one_probability
    end

    write_probabilistic_forecast!(
        grid, entities, positions, directions, lane_one_probability, ring,
    )
    return occupancy
end

@inline function fixed_lane_one_probability(
        strategy::DriverStrategy,
        position::Position,
        direction::Direction,
        habitus::Float64,
        mean_abs_habitus::Float64,
        rng,
    )
    kind = strategy.kind
    if kind == NaiveKind || kind == TwoFrameNaiveKind || kind == DecisionAwareKind
        return position.x == 1 ? 1.0 : 0.0
    elseif kind == UnsureKind
        return 0.5
    elseif kind == SwitchKind
        return position.x == 1 ? 0.0 : 1.0
    elseif kind == RandomKind
        return rand(rng)
    elseif kind == MeanHabitusKind
        confidence = clamp(mean_abs_habitus, 0.0, 1.0)
        return position.x == 1 ? 0.5 + confidence / 2 : 0.5 - confidence / 2
    elseif kind == PerEntityHabitusKind
        confidence = clamp(abs(habitus), 0.0, 1.0)
        prefer_lane_one =
            (direction == Clockwise && habitus > 0.0) ||
            (direction == Counterclockwise && habitus < 0.0)
        return prefer_lane_one ? 0.5 + confidence / 2 : 0.5 - confidence / 2
    end
    error("unsupported driver strategy: $(kind)")
end

"""
Compose one predicted-occupancy field from per-car strategies.

Non-iterative cars contribute their own forecast directly. Decision-aware cars
best-respond to that combined field, so they account for both one another and
the fixed forecasts of less sophisticated drivers.
"""
function rebuild_predicted_occupancy!(world, ::HeterogeneousStrategy)
    occupancy = Ark.get_resource(world, PredictedOccupancy)
    grid = occupancy.grid
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    weights = Ark.get_resource(world, Weights)
    rng = simulation_rng(world)
    mean_abs_habitus = Ark.get_resource(world, MeanHabitus).abs

    entities = Ark.Entity[]
    positions = Position[]
    directions = Direction[]
    same_sensitivities = Float64[]
    opposite_sensitivities = Float64[]
    avoidances = Float64[]
    habitgenes = Float64[]
    habitus_values = Float64[]
    strategies = DriverStrategy[]

    for (e, pos, dir, same, opposite, avoidance, habitgene, habitus, strategy) in Query(
            world,
            (
                Position, Direction, SSensitvity, OSensitvity, Avoidance,
                Habitgene, Habitus, DriverStrategy,
            ),
        )
        @inbounds for i in eachindex(e)
            push!(entities, e[i])
            push!(positions, pos[i])
            push!(directions, dir[i])
            push!(same_sensitivities, same[i].val)
            push!(opposite_sensitivities, opposite[i].val)
            push!(avoidances, avoidance[i].val)
            push!(habitgenes, habitgene[i].val)
            push!(habitus_values, habitus[i].val)
            push!(strategies, strategy[i])
        end
    end

    lane_one_probability = [
        fixed_lane_one_probability(
            strategies[i], positions[i], directions[i], habitus_values[i],
            mean_abs_habitus, rng,
        )
            for i in eachindex(entities)
    ]
    updated_probability = similar(lane_one_probability)
    max_iterations = maximum(
        (strategy.iterations for strategy in strategies if strategy.kind == DecisionAwareKind);
        init = 0,
    )

    for iteration in 1:max_iterations
        write_probabilistic_forecast!(
            grid, entities, positions, directions, lane_one_probability, ring,
        )
        copyto!(updated_probability, lane_one_probability)

        @inbounds for i in eachindex(entities)
            strategy = strategies[i]
            if strategy.kind == DecisionAwareKind && iteration <= strategy.iterations
                score = decision_score(
                    entities[i], grid, positions[i], directions[i], same_sensitivities[i],
                    opposite_sensitivities[i], avoidances[i], habitgenes[i], habitus_values[i],
                    ring, params, weights,
                )
                go_left = score > 0.0 ? true :
                    score < 0.0 ? false : is_left_relative(positions[i].x, directions[i])
                preferred_lane = directions[i] == Clockwise ?
                    (go_left ? 1 : 2) : (go_left ? 2 : 1)
                desired_probability = preferred_lane == 1 ? 1.0 - params.ϵ : params.ϵ
                updated_probability[i] = lane_one_probability[i] +
                    strategy.damping * (desired_probability - lane_one_probability[i])
            end
        end
        lane_one_probability, updated_probability = updated_probability, lane_one_probability
    end

    write_probabilistic_forecast!(
        grid, entities, positions, directions, lane_one_probability, ring,
    )
    return occupancy
end

function write_probabilistic_forecast!(
        grid,
        entities,
        positions,
        directions,
        lane_one_probability,
        ring,
    )
    empty!.(grid)
    @inbounds for i in eachindex(entities)
        next_position = predict_position(positions[i], directions[i], ring)
        probability = lane_one_probability[i]
        push!(grid[1, next_position.y], (directions[i], entities[i], probability))
        push!(grid[2, next_position.y], (directions[i], entities[i], 1.0 - probability))
    end
    return grid
end

#every agent assumes the cars just step forward once
@inline function predict_position(
        pos::Position,
        dir::Direction,
        ring::Ring
    )

    # forward movement
    y = step_y(Position(pos.x, pos.y), ring, dir)

    return Position(pos.x, y)
end

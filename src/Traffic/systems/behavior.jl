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
    fill!(grid, nothing)

    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    rng = Ark.get_resource(world, TaskLocalRNG)

    for (e, pos, dir, habitus) in Query(world, (Position, Direction, Habitus))
        @inbounds for i in eachindex(e)

            pnext = predict_position(pos[i], dir[i], ring, params, rng)

            h = habitus[i].val
            w = abs(h)

            # determine which lane is preferred
            prefer_lane1 = (dir[i] == Clockwise  && h > 0) || (dir[i] == Counterclockwise && h < 0)

            if prefer_lane1
                push!(grid[1, pnext.y], (dir[i], e[i], w))
                push!(grid[2, pnext.y], (dir[i], e[i], 1 - w))
            else
                gpush!(rid[2, pnext.y], (dir[i], e[i], w))
                push!(grid[1, pnext.y], (dir[i], e[i], 1 - w))
            end
        end
    end

    return occ
end

function rebuild_predicted_occupancy!(world, ::MeanHabitusStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    fill!(grid, [])
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    rng = Ark.get_resource(world, TaskLocalRNG)
    mean_habitus = Ark.get_resource(world, MeanHabitus)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring, params, rng)
            push!(grid[pnext.x, pnext.y], (dir[i], e[i], mean_habitus.abs))
            push!(grid[pnext.x == 1 ? 2 : 1, pnext.y], (dir[i], e[i], 1 - mean_habitus.abs))
        end
    end

    return occ
end

function rebuild_predicted_occupancy!(world, ::NaiveStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    fill!(grid, [])
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    rng = Ark.get_resource(world, TaskLocalRNG)
    mean_habitus = Ark.get_resource(world, MeanHabitus)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring, params, rng)
            push!(grid[pnext.x, pnext.y], (dir[i], e[i], 1.0))
        end
    end

    return occ
end

function rebuild_predicted_occupancy!(world, ::UnsureStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    fill!(grid, [])
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    rng = Ark.get_resource(world, TaskLocalRNG)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring, params, rng)
            push!(grid[1, pnext.y], (dir[i], e[i], 0.5))
            push!(grid[2, pnext.y], (dir[i], e[i], 0.5))
        end
    end

    return occ
end
function rebuild_predicted_occupancy!(world, ::SwitchStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    fill!(grid, [])
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    rng = Ark.get_resource(world, TaskLocalRNG)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring, params, rng)
            push!(grid[pnext.x == 1 ? 2 : 1, pnext.y], (dir[i], e[i], 1.0))
        end
    end

    return occ
end

function rebuild_predicted_occupancy!(world, ::RandomStrategy)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    fill!(grid, nothing)
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    rng = Ark.get_resource(world, TaskLocalRNG)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring, params, rng)
            r = rand(rng)
            push!(grid[1, pnext.y], (dir[i], e[i], r))
            push!(grid[2, pnext.y], (dir[i], e[i], 1 - r))
        end
    end

    return occ
end

#every agent assumes the cars just step forward once
@inline function predict_position(
        pos::Position,
        dir::Direction,
        ring::Ring,
        params,
        rng
    )

    # forward movement
    y = step_y(Position(pos.x, pos.y), ring, dir)

    return Position(pos.x, y)
end

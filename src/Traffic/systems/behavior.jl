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
            o_val = weights.wₒ * o[i].val * (2 * OL - 1)
            avoidance_val = weights.wₐ * a[i].val * (CR - CL)
            habit_strength = weights.wₕ * hg[i].val * ha[i].val

            lr[i] = LR(s_val + o_val + avoidance_val + habit_strength)
        end
    end

    return nothing
end

@inline function compute_observations(e,
        occ, pos::Position, dir::Direction, ring::Ring, lookahead::Int
    )
    h = Int(ring.height)

    same_total = 0
    same_left = 0
    opp_total = 0
    opp_left = 0

    CL = 0
    CR = 0

    @inbounds for d in 1:lookahead
        y = ahead_y(pos.y, dir, d, h)

        for x in 1:2
            occ[x,y] != nothing || continue

            dir_other, e_other = occ[x, y]

            e_other != e || continue
            left_rel = is_left_relative(x, dir)

            if dir_other == dir
                same_total += 1
                same_left += left_rel
            else
                opp_total += 1
                opp_left += left_rel
            end

            if d ≤ 2
                if left_rel
                    CL += 1
                else
                    CR += 1
                end
            end
        end
    end

    SL = same_total == 0 ? 0.5 : same_left / same_total
    OL = opp_total == 0 ? 0.5 : opp_left / opp_total

    return SL, OL, CL, CR
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

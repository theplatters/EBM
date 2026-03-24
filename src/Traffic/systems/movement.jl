function step_y(pos::Position, ring::Ring, direction::Direction)
    h = Int(ring.height)
    return mod1(Int(pos.y) + Int(direction), h)
end

function move!(world)
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    rng = Ark.get_resource(world, TaskLocalRNG)

    for (e, pos, lr) in Query(world, (Position, LR))
        @inbounds for i in eachindex(e)
            if rand(rng) < params.ϵ

                pos[i] = Position(lr[i].val > 0.0 ? 2 : 1, pos[i].y)
            else

                pos[i] = Position(lr[i].val > 0.0 ? 1 : 2, pos[i].y)
            end
        end
    end

    for (e, pos, dir, step) in Query(world, (Position, Direction, Step))
        @inbounds for i in eachindex(e)
            pos[i] = Position(pos[i].x, step_y(pos[i], ring, dir[i]))
            step[i] = Step(step[i].val + 1)
        end
    end

    return nothing
end

function store_prev_positions!(world)
    for (_, pos, prev) in Query(world, (Position, PrevPosition))
        prev .= PrevPosition.(pos)
    end
    return
end

function rebuild_occupancy!(world)
    occ = Ark.get_resource(world, Occupancy)
    grid = occ.grid
    fill!(grid, nothing)

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            grid[pos[i].x, pos[i].y] = dir[i]
        end
    end

    return occ
end

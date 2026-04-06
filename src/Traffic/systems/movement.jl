function step_y(pos::Position, ring::Ring, direction::Direction)
    h = Int(ring.height)
    return mod1(Int(pos.y) + Int(direction), h)
end


function move!(world)
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    rng = Ark.get_resource(world, TaskLocalRNG)

    for (e, pos, dir, lr, step) in Query(world, (Position, Direction, LR, Step))
        @inbounds for i in eachindex(e)
            ynew = step_y(pos[i], ring, dir[i])

            go_left = lr[i].val > 0.0
            if rand(rng) < params.ϵ
                go_left = !go_left
            end

            xnew = dir[i] == Clockwise ? (go_left ? 2 : 1) : (go_left ? 1 : 2)

            pos[i] = Position(xnew, ynew)
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

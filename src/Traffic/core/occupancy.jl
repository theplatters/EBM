struct Occupancy
    grid::Matrix{Union{Nothing, Direction}}
end

struct PredictedOccupancy
    grid::Matrix{Union{Nothing, Tuple{Direction,Entity}}}
end

function Occupancy(ring::Ring)
    return Occupancy(fill(nothing, Int(ring.width), Int(ring.height)))
end


function PredictedOccupancy(ring::Ring)
    return PredictedOccupancy(fill(nothing, Int(ring.width), Int(ring.height)))
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

function rebuild_predicted_occupancy!(world)
    occ = Ark.get_resource(world, PredictedOccupancy)
    grid = occ.grid
    fill!(grid, nothing)
    ring = Ark.get_resource(world, Ring)
    params = Ark.get_resource(world, ModelParams)
    rng = Ark.get_resource(world, TaskLocalRNG)

    for (e, pos, dir, lr) in Query(world, (Position, Direction, LR))
        @inbounds for i in eachindex(e)
            pnext = predict_position(pos[i], dir[i], ring, params, rng)
            grid[pnext.x, pnext.y] = (dir[i],e[i])
        end
    end

    return occ
end

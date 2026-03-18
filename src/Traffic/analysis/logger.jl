mutable struct Logger
    left::Vector{Tuple{Int64, Int64}}
    right::Vector{Tuple{Int64, Int64}}
    deaths::Vector{Int64}
    positions::Vector{Matrix{Union{Nothing, Direction}}}
end

Logger() = Logger(Vector{Tuple{Int64, Int64}}(), Vector{Tuple{Int64, Int64}}(), Int64[], Matrix{Union{Nothing, Entity}}[])

function logger!(world)
    logger = Ark.get_resource(world, Logger)

    push!(logger.left, (0, 0))
    push!(logger.right, (0, 0))

    push!(logger.positions, deepcopy(Ark.get_resource(world, Occupancy).grid))

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            idx = lastindex(logger.left)

            inc = (dir[i] == Clockwise) ? (1, 0) : (0, 1)

            if pos[i].x == 1
                logger.left[idx] = logger.left[idx] .+ inc   # or: logger.left[idx] + inc
            else
                logger.right[idx] = logger.right[idx] .+ inc
            end
        end
    end
    return
end

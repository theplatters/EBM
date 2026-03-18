mutable struct Logger
    left::Vector{Tuple{Int64, Int64}}
    right::Vector{Tuple{Int64, Int64}}
    deaths::Vector{Int64}
    mean_habitus::Vector{Float64}
    positions::Vector{Matrix{Union{Nothing, Direction}}}
    mean_age::Vector{Float64}
end

Logger() = Logger(Vector{Tuple{Int64, Int64}}(), Vector{Tuple{Int64, Int64}}(), Int64[],Float64[], Matrix{Union{Nothing, Entity}}[], Float64[])

function logger!(world)
    logger = Ark.get_resource(world, Logger)

    push!(logger.left, (0, 0))
    push!(logger.right, (0, 0))

    push!(logger.positions, deepcopy(Ark.get_resource(world, Occupancy).grid))

    total_habitus = 0
    total_entities = 0
    
    for (e, habitus) in Query(world, (Habitus,))
      for i in eachindex(e)
        total_habitus += habitus[i].val
        total_entities += 1
      end
    end


    push!(logger.mean_habitus, total_habitus / total_entities)
    total_step = 0
    total_entities = 0

    for (e, step) in Query(world, (Step,))
      for i in eachindex(e)
        total_step += step[i].val
        total_entities += 1
      end
    end

    push!(logger.mean_age, total_step / total_entities)

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

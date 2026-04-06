abstract type AbstractLogger end

mutable struct Logger <: AbstractLogger
    left::Vector{Tuple{Int64, Int64}}
    right::Vector{Tuple{Int64, Int64}}
    deaths::Vector{Int64}
    mean_habitus::Vector{Float64}
    mean_abs_habitus::Vector{Float64}
    positions::Vector{Matrix{Union{Nothing, Direction}}}
    mean_age::Vector{Float64}
    distribution_S::Vector{Vector{Float64}}
    distribution_O::Vector{Vector{Float64}}
    distribution_A::Vector{Vector{Float64}}
    lr::Vector{Vector{Float64}}
    stay_ratio::Vector{Float64}
end

Logger() = Logger(
    Vector{Tuple{Int64, Int64}}(),
    Vector{Tuple{Int64, Int64}}(),
    Int64[],
    Float64[],
    Float64[],
    Matrix{Union{Nothing, Entity}}[],
    Float64[],
    Vector{Vector{Float64}}(),
    Vector{Vector{Float64}}(),
    Vector{Vector{Float64}}(),
    Vector{Vector{Float64}}(),
    Vector{Float64}(),
)

function log_lr!(world, logger)

    for (_, lr) in Query(world, (LR,))
        push!(logger.lr, collect(lr.val))
    end

    return
end

function log_habitus!(world, logger)

    mean_habitus = Ark.get_resource(world, MeanHabitus)

    push!(logger.mean_habitus, mean_habitus.total)
    push!(logger.mean_abs_habitus, mean_habitus.abs)
    return nothing
end

function log_deaths!(world, logger)
    babies = sum(count(s -> s.val == 1, step) for (_, step) in Query(world, (Step,)))
    return push!(logger.deaths, babies)
end

function log_mean_age!(world, logger)
    total_step = 0
    total_entities = 0

    for (e, step) in Query(world, (Step,))
        for i in eachindex(e)
            total_step += step[i].val
            total_entities += 1
        end
    end

    push!(logger.mean_age, total_step / total_entities)

    return nothing
end

function log_positions!(world, logger)

    push!(logger.positions, deepcopy(Ark.get_resource(world, Occupancy).grid))
    push!(logger.left, (0, 0))
    push!(logger.right, (0, 0))

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

    return nothing
end

function log_distributions!(world, logger)

    for (e, ss, os, av) in Query(world, (SSensitvity, OSensitvity, Avoidance))
        push!(logger.distribution_A, collect(av.val))
        push!(logger.distribution_S, collect(ss.val))
        push!(logger.distribution_O, collect(os.val))

    end
    return nothing
end

function log_stays!(world, logger)

    stays = 0
    total = 0
    for (e, pos, prev) in Query(world, (Position, PrevPosition))
        @inbounds for i in eachindex(e)
            stays += (pos[i].x == Position(prev[i]).x)
            total += 1
        end
    end
    push!(logger.stay_ratio, stays / total)
    return nothing
end

function logger!(world)
    logger = Ark.get_resource(world, Logger)

    log_habitus!(world, logger)
    log_deaths!(world, logger)
    log_mean_age!(world, logger)
    log_positions!(world, logger)
    log_distributions!(world, logger)
    log_lr!(world, logger)
    log_stays!(world, logger)

    return
end


struct MeanLogger <: AbstractLogger
    left::Vector{Tuple{Int64, Int64}}
    right::Vector{Tuple{Int64, Int64}}
    deaths::Vector{Int64}
    mean_habitus::Vector{Float64}
    mean_abs_habitus::Vector{Float64}
    positions::Vector{Matrix{Union{Nothing, Direction}}}
    mean_age::Vector{Float64}
    distribution_S::Vector{Vector{Float64}}
    distribution_O::Vector{Vector{Float64}}
    distribution_A::Vector{Vector{Float64}}
    mean_stay_ratio::Vector{Float64}
end


function MeanLogger(loggers::AbstractArray{T}) where {T <: AbstractLogger}
    n = length(loggers)

    L = length(loggers[1].mean_habitus)

    left = Vector{Tuple{Int64, Int64}}(undef, L)
    right = Vector{Tuple{Int64, Int64}}(undef, L)

    deaths = zeros(Int64, L)
    mean_habitus = zeros(Float64, L)
    mean_abs_habitus = zeros(Float64, L)
    mean_age = zeros(Float64, L)
    mean_stay_ratio = zeros(Float64, L)

    distribution_S = Vector{Vector{Float64}}(undef, L)
    distribution_O = Vector{Vector{Float64}}(undef, L)
    distribution_A = Vector{Vector{Float64}}(undef, L)

    for t in 1:L
        lx = ly = rx = ry = 0

        for logger in loggers
            l = logger.left[t]
            r = logger.right[t]

            lx += l[1]; ly += l[2]
            rx += r[1]; ry += r[2]

            deaths[t] += logger.deaths[t]
            mean_habitus[t] += logger.mean_habitus[t]
            mean_abs_habitus[t] += logger.mean_abs_habitus[t]
            mean_age[t] += logger.mean_age[t]
            mean_stay_ratio[t] += logger.stay_ratio[t]
        end

        left[t] = (round(Int, lx / n), round(Int, ly / n))
        right[t] = (round(Int, rx / n), round(Int, ry / n))

        deaths[t] = round(Int, deaths[t] / n)
        mean_habitus[t] /= n
        mean_abs_habitus[t] /= n
        mean_age[t] /= n
        mean_stay_ratio[t] /= n

        distribution_S[t] = mean(reduce(hcat, [l.distribution_S[t] for l in loggers]), dims = 2)[:]
        distribution_O[t] = mean(reduce(hcat, [l.distribution_O[t] for l in loggers]), dims = 2)[:]
        distribution_A[t] = mean(reduce(hcat, [l.distribution_A[t] for l in loggers]), dims = 2)[:]
    end

    positions = deepcopy(loggers[1].positions)

    return MeanLogger(
        left,
        right,
        deaths,
        mean_habitus,
        mean_abs_habitus,
        positions,
        mean_age,
        distribution_S,
        distribution_O,
        distribution_A,
        mean_stay_ratio
    )
end


function convergence(l::AbstractLogger)
    return maximum.(l.left) / 20
end

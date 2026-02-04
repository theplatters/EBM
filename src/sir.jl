import Ark
using Plots
using Ark: Entity, Query
using LinearAlgebra: diagind
using StatsBase


struct Position
    x::UInt64
    y::UInt64
end

struct Infected
    since::UInt64
end

struct Cured end

struct Grid
    width::UInt64
    height::UInt64
end

struct WeightCache
    rows::Vector{Weights}
    cols::Vector{Weights}
end

function WeightCache(m::Matrix{Float64})
    rows = Weights[]
    cols = Weights[]
    for row in eachrow(m)
        push!(rows,Weights(row))
    end

    for col in eachcol(m)
        push!(cols,Weights(col))
    end
    WeightCache(rows,cols)
end


struct TransmitionRates
    undetected::Float64
    detected::Float64
    detection_time::UInt64
    reinfection_probability::Float64
end

struct Logger
    infected::Vector{Int64}
    cured::Vector{Int64}
    dead::Vector{Int64}
    city_matrix::Vector{Matrix{Float64}}
end

function Logger()
    return Logger(Vector{Int64}(), Vector{Int64}(), Vector{Int64}(), Vector{Matrix{Float64}}())
end

function generate_migration_rates(grid)
    Ns = rand(50:5000, grid.width)

    migration_rates = zeros(grid.width, grid.height)

    for c in 1:grid.width
        for c2 in 1:grid.height
            migration_rates[c, c2] = (Ns[c] + Ns[c2]) / Ns[c]
        end
    end

    migration_rates[diagind(migration_rates)] .= 1.0
    return migration_rates
end

struct WeightCacheCDF
    cdf_rows::Vector{Vector{Float64}}  # each is length W, last element == 1 (or == total)
    cdf_cols::Vector{Vector{Float64}}  # each is length H
end

# Build once (when migration_rates changes)
function WeightCacheCDF(migration_rates::AbstractMatrix{<:Real})
    H, W = size(migration_rates)

    # Example assumes:
    # - "row conditional" for x given y0: weights = migration_rates[y0, :]
    # - "col conditional" for y given x0: weights = migration_rates[:, x0]
    # Adjust if your meaning differs.
    cdf_rows = Vector{Vector{Float64}}(undef, H)
    for y in 1:H
        w = Float64.(view(migration_rates, y, :))
        s = sum(w)
        s == 0 && (w .= 1)  # fallback to uniform if degenerate; choose your policy
        c = cumsum(w)
        c ./= c[end]
        cdf_rows[y] = c
    end

    cdf_cols = Vector{Vector{Float64}}(undef, W)
    for x in 1:W
        w = Float64.(view(migration_rates, :, x))
        s = sum(w)
        s == 0 && (w .= 1)
        c = cumsum(w)
        c ./= c[end]
        cdf_cols[x] = c
    end

    return WeightCacheCDF(cdf_rows, cdf_cols)
end

@inline function draw_cdf(cdf::AbstractVector{<:Real})
    u = rand()
    return searchsortedfirst(cdf, u)
end

function migrate!(world, grid)

    cache = Ark.get_resource(world, WeightCacheCDF)
    H = length(cache.cdf_rows)
    W = length(cache.cdf_cols)

    for (entities, positions) in Query(world, (Position,))
        @inbounds for i in eachindex(entities)
            x0 = clamp(positions[i].x, 1, W)
            y0 = clamp(positions[i].y, 1, H)

            # x conditional on y0, y conditional on x0 (consistent naming)
            x = draw_cdf(cache.cdf_rows[y0])
            y = draw_cdf(cache.cdf_cols[x0])

            positions[i] = Position(x, y)
        end
    end
    return
end


# helper for 1-based indexing safety if needed
@inline cellrate(days_infected::UInt64, tr::TransmitionRates) = (days_infected < tr.detection_time) ? tr.undetected : tr.detected

function transmit!(world, tr::TransmitionRates)
    # Infectors: have Position + Infected
    to_add = Vector{Pair{Entity, Tuple{Infected}}}()
    to_remove = Vector{Pair{Entity, Tuple}}()

    positions_to_infect = Dict{Position, Float64}()


    for (inf_entities, inf_pos, inf_inf) in Query(world, (Position, Infected))
        @inbounds for i in eachindex(inf_entities)
            rate = cellrate(inf_inf[i].since, tr)

            n = rate * abs(randn())
            n <= 0 && continue

            positions_to_infect[inf_pos[i]] = get(positions_to_infect, inf_pos[i], 0) + n

        end
    end


    for (sus_entities, sus_pos) in Query(world, (Position,), without = (Infected,))

        @inbounds for i in eachindex(sus_entities)
            n = get(positions_to_infect, sus_pos[i], 0)
            n > 0 || continue
            e = sus_entities[i]

            # If entity has Cured -> allow reinfection with probability
            if Ark.has_components(world, e, (Cured,))
                rand() ≤ tr.reinfection_probability || continue
                push!(to_remove, e => (Cured,))
            end

            push!(to_add, e => (Infected(0),))
            haskey(positions_to_infect, sus_pos[i]) && (positions_to_infect[sus_pos[i]] -= 1)
        end

    end

    return (to_add, to_remove)
end

function recover_or_die!(world, infection_period, death_rate)
    to_remove = Vector{Entity}()
    to_add_component = Vector{Entity}()
    for (entity, infected) in Query(world, (Infected,))
        @inbounds for i in eachindex(entity)
            if infected[i].since ≥ infection_period
                if rand() ≤ death_rate
                    push!(to_remove, entity[i])
                else
                    push!(to_add_component, entity[i])
                end
            end
        end
    end


    return (to_remove, to_add_component)

end


function update!(world)
    for (entity, infected) in Query(world, (Infected,))
        Threads.@threads for i in eachindex(entity)
            infected[i] = Infected(infected[i].since + 1)
        end
    end
    return
end

function spawn_entities!(world, grid, init_entities)
    for _ in 1:init_entities
        x = rand(1:grid.width)
        y = rand(1:grid.height)

        is_infected = rand() > 0.9
        bundle = is_infected ? (Position(x, y), Infected(0)) : (Position(x, y),)
        if is_infected
            Ark.new_entity!(world, bundle)
        else
            Ark.new_entity!(world, bundle)
        end
    end
    return
end


function step!(world, grid, tr)
    # These two operate on different components - can run in parallel
    migrate!(world, grid)
    update!(world)
    

    
    # Both are read-only queries - can run in parallel
    (to_add, to_remove) = transmit!(world, tr)
    (entities_to_remove, entities_to_change) = recover_or_die!(world, 50, 0.1)
    

    # World mutations must be serial
    for t in to_add
        Ark.add_components!(world, first(t), last(t))
    end

    for t in to_remove
        Ark.remove_components!(world, first(t), last(t))
    end

    for r in entities_to_remove
        Ark.remove_entity!(world, r)
    end
    
    for r in entities_to_change
        Ark.exchange_components!(world, r, remove = (Infected,), add = (Cured(),))
    end
    
    return
end

function iterate!(world, grid, tr, init_entities)
    for _ in 1:steps
        step!(world, grid, tr)
        logger = Ark.get_resource(world, Logger)

        q = Ark.Query(world, (Infected,))
        push!(
            logger.infected, Ark.count_entities(q)
        )

        Ark.close!(q)


        q = Ark.Query(world, (Cured,))
        push!(
            logger.cured, Ark.count_entities(q)
        )

        Ark.close!(q)
        q = Ark.Query(world, ())

        push!(
            logger.dead, init_entities - Ark.count_entities(q)
        )


        Ark.close!(q)
    end
    return
end


function @main(args)
    tr = TransmitionRates(args[1], args[2], args[3], args[4])

    init_entities = 10_000

    grid = Grid(24, 24)
    migration_rates = generate_migration_rates(grid)

    world = Ark.World(Position, Infected, Cured)
    Ark.add_resource!(world, Logger())
    Ark.add_resource!(world, WeightCacheCDF(migration_rates))
    @time spawn_entities!(world, grid, init_entities)

    @time iterate!(world, grid, tr, init_entities)

    logger = Ark.get_resource(world, Logger)

    plot(logger.infected)
    plot!(logger.dead, label = "dead")
    plot!(logger.cured, label ="cured")
end

@main([0.01, 0.001, 10, 0.1])


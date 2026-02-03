import Ark
using Plots
using Ark: Query, Entity
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
end

function Logger()
    return Logger(Vector{Int64}(), Vector{Int64}(), Vector{Int64}())
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


function migrate!(world, grid, migration_rates)
    H, W = size(migration_rates)

    for (entities, positions) in Query(world, (Position,))
        @inbounds for i in eachindex(entities)
            # ensure current position is valid for 1-based arrays
            x0 = clamp(positions[i].x, 1, W)
            y0 = clamp(positions[i].y, 1, H)

            x = sample(1:W, Weights(@view migration_rates[y0, :]))   # row: fixed y
            y = sample(1:H, Weights(@view migration_rates[:, x0]))   # col: fixed x

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

            # stochastic "budget" of infections (same structure as your ABM version)
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
        @inbounds for i in eachindex(entity)
            infected[i] = Infected(infected[i].since + 1)
        end
    end
    return
end

function spawn_entities!(world, grid)
    for _ in 1:1000
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


function step!(world, grid, migration_rates, tr)
    migrate!(world, grid, migration_rates)
    (to_add, to_remove) = transmit!(world, tr)

    for t in to_add
        Ark.add_components!(world, first(t), last(t))
    end

    for t in to_remove
        Ark.remove_components!(world, first(t), last(t))
    end
    update!(world)

    to_remove, to_change = recover_or_die!(world, 10, 0.2)

    for r in to_remove
        Ark.remove_entity!(world, r)
    end
    for r in to_change
        Ark.exchange_components!(world, r, remove = (Infected,), add = (Cured(),))
    end
    return
end

function iterate!(world, grid, migration_rates, tr)
    for _ in 1:100
        step!(world, grid, migration_rates, tr)
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
            logger.dead, 1000 - Ark.count_entities(q)
        )

        Ark.close!(q)
    end
    return
end


function @main(rgs)
    tr1 = 0.9
    tr = TransmitionRates(tr1, tr1 / 10, 10, 0.05)


    grid = Grid(64, 64)
    migration_rates = generate_migration_rates(grid)

    world = Ark.World(Position, Infected, Cured)
    Ark.add_resource!(world, Logger())

    spawn_entities!(world, grid)

    iterate!(world, grid, migration_rates, tr)

    logger = Ark.get_resource(world, Logger)

    plot(logger.infected)
    plot!(logger.dead, label = "dead")
    plot!(logger.cured, label ="cured")
end

@main(1)


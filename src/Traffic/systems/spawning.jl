function random_unique_positions(r::Ring, n::Int; rng = Random.default_rng())
    w = r.width
    h = r.height
    total = w * h
    n <= total || throw(ArgumentError("n=$n exceeds ring capacity=$total"))
    idxs = randperm(rng, total)[1:n]
    return Position.([(mod1(i, w), (i - 1) ÷ w + 1) for i in idxs])
end

@inline function spawn_car!(
        world, pos, dir, ssens, osense, avoid, habitgene, habitus, strategy,
    )
    return Ark.new_entity!(
        world, (
            pos,
            PrevPosition(pos),
            dir,
            ssens,
            osense,
            avoid,
            habitgene,
            habitus,
            strategy,
            LR(0.0),
            Step(1),
        )
    )
end

function initial_driver_strategies(
        strategy::OccupancyStrategy, amount::Integer, rng,
    )
    return fill(DriverStrategy(strategy), amount)
end

function initial_driver_strategies(
        mixture::HeterogeneousStrategy, amount::Integer, rng,
    )
    exact_counts = mixture.shares .* amount
    counts = floor.(Int, exact_counts)
    remaining = amount - sum(counts)
    remainder_order = sortperm(
        eachindex(exact_counts);
        by = index -> (-(exact_counts[index] - counts[index]), index),
    )
    for index in Iterators.take(remainder_order, remaining)
        counts[index] += 1
    end

    assignments = DriverStrategy[]
    for (strategy, count) in zip(mixture.strategies, counts)
        append!(assignments, fill(strategy, count))
    end
    shuffle!(rng, assignments)
    return assignments
end

function spawn_init_entities!(world, strategy::OccupancyStrategy)
    ring = Ark.get_resource(world, Ring)
    rng = simulation_rng(world)
    params = Ark.get_resource(world, ModelParams)

    amount = params.init_agents

    positions = random_unique_positions(ring, amount, rng = rng)
    directions = shuffle(
        rng,
        repeat([Clockwise, Counterclockwise], cld(amount, 2))[1:amount],
    )
    draws = rand(rng, Normal(1.0, params.δ), amount, 4)
    # Keep physical initial conditions paired across homogeneous and mixed runs.
    # Only the strategy assignment consumes additional random numbers.
    strategies = initial_driver_strategies(strategy, amount, rng)

    @inbounds for i in 1:amount
        spawn_car!(
            world,
            positions[i],
            directions[i],
            SSensitvity(draws[i, 1]),
            OSensitvity(draws[i, 2]),
            Avoidance(draws[i, 3]),
            Habitgene(draws[i, 4]),
            Habitus(0.0),
            strategies[i],
        )
    end
    return nothing
end


struct ReplacementSpec
    direction::Direction
    strategy::DriverStrategy
end

function spawn_new_entities!(world, replacements::AbstractVector{ReplacementSpec})
    amount = length(replacements)
    rng = simulation_rng(world)
    params = Ark.get_resource(world, ModelParams)
    ring = Ark.get_resource(world, Ring)
    occupied_positions = Set{Position}()
    for (e, pos) in Query(world, (Position,))
        @inbounds for i in eachindex(e)
            push!(occupied_positions, pos[i])
        end
    end
    unoccupied_positions = Position[
        Position(x, y)
            for y in 1:ring.height for x in 1:ring.width
                if Position(x, y) ∉ occupied_positions
    ]

    replacements = shuffle(rng, replacements)
    draws = rand(rng, Normal(1.0, params.δ), amount, 4)

    for i in 1:amount
        position_index = rand(rng, eachindex(unoccupied_positions))
        position = unoccupied_positions[position_index]

        spawn_car!(
            world,
            position,
            replacements[i].direction,
            SSensitvity(draws[i, 1]),
            OSensitvity(draws[i, 2]),
            Avoidance(draws[i, 3]),
            Habitgene(draws[i, 4]),
            Habitus(0.0),
            replacements[i].strategy,

        )

        deleteat!(unoccupied_positions, position_index)
    end

    return nothing
end

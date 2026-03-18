function random_unique_positions(r::Ring, n::Int; rng = Random.default_rng())
    w = r.width
    h = r.height
    total = w * h
    n <= total || throw(ArgumentError("n=$n exceeds ring capacity=$total"))
    idxs = randperm(rng, total)[1:n]
    return Position.([(mod1(i, w), (i - 1) ÷ w + 1) for i in idxs])
end

@inline function spawn_car!(world, pos, dir, ssens, osense, avoid, habitgene, habitus)
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
            LR(0.0),
            Step(1),
        )
    )
end

function spawn_init_entities!(world)
    ring = Ark.get_resource(world, Ring)
    rng = Ark.get_resource(world, TaskLocalRNG)
    params = Ark.get_resource(world, ModelParams)

    amount = params.init_agents

    positions = random_unique_positions(ring, amount, rng = rng)
    directions = shuffle(rng, repeat([Clockwise, Counterclockwise], amount ÷ 2))
    draws = rand(rng, Normal(1.0, params.δ), amount, 4)

    return @inbounds for i in 1:amount
        spawn_car!(
            world,
            positions[i],
            directions[i],
            SSensitvity(draws[i, 1]),
            OSensitvity(draws[i, 2]),
            Avoidance(draws[i, 3]),
            Habitgene(draws[i, 4]),
            Habitus(0.0),
        )
    end
end


function spawn_new_entities!(world, directions)
    amount = length(directions)
    rng = Ark.get_resource(world, TaskLocalRNG)
    params = Ark.get_resource(world, ModelParams)
    ring = Ark.get_resource(world, Ring)
    unoccupied_positions = Set{Position}(Position(x, y) for x in 1:ring.width, y in 1:ring.height)
    for (e, pos) in Query(world, (Position,))
        @inbounds for i in eachindex(e)
            delete!(unoccupied_positions, pos[i])
        end
    end

    directions = shuffle(rng, directions)
    draws = rand(rng, Normal(1.0, params.δ), amount, 4)

    for i in 1:amount
        position = rand(unoccupied_positions)

        spawn_car!(
            world,
            position,
            directions[i],
            SSensitvity(draws[i, 1]),
            OSensitvity(draws[i, 2]),
            Avoidance(draws[i, 3]),
            Habitgene(draws[i, 4]),
            Habitus(0.0),

        )

        delete!(unoccupied_positions, position)
    end

    return
end

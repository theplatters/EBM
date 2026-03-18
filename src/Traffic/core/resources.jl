function setup_resources!(world, args)
    params = get!(args, :params, ModelParams())
    ring = Ring(params.ring_x, params.ring_y)
    Ark.add_resource!(world, ring)

    Ark.add_resource!(world, PredictedOccupancy(ring))
    Ark.add_resource!(world, Occupancy(ring))
    Ark.add_resource!(
        world,
        haskey(args, :seed) ? Random.default_rng(args[:seed]) : Random.default_rng()
    )

    Ark.add_resource!(world, params)
    Ark.add_resource!(world, get!(args, :weights, Weights()))
    Ark.add_resource!(world, get!(args, :logger, Logger()))


    return nothing
end

function setup_resources!(world, args::ModelArgs{T}) where {T}
    params = args.params
    ring = Ring(params.ring_x, params.ring_y)
    Ark.add_resource!(world, ring)

    Ark.add_resource!(world, PredictedOccupancy(ring))
    Ark.add_resource!(world, Occupancy(ring))

    Ark.add_resource!(world, MeanHabitus(0, 0))
    Ark.add_resource!(
        world,
        Random.default_rng(args.seed)
    )

    Ark.add_resource!(world, params)
    Ark.add_resource!(world, args.weights)
    Ark.add_resource!(world, Logger())


    return nothing
end

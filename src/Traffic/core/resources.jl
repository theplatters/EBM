function setup_resources!(world, args::ModelArgs)
    params = args.params
    ring = Ring(params.ring_x, params.ring_y)
    Ark.add_resource!(world, ring)

    Ark.add_resource!(world, PredictedOccupancy(ring))
    Ark.add_resource!(world, Occupancy(ring))

    Ark.add_resource!(world, MeanHabitus(0, 0))
    Ark.add_resource!(
        world,
        args.seed
    )

    Ark.add_resource!(world, params)
    Ark.add_resource!(world, args.weight)
    Ark.add_resource!(world, Logger())


    return nothing
end

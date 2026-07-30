function setup_world(args::ModelArgs{T}) where {T}
    world = Ark.World(
        Position,
        PrevPosition,
        Direction,
        SSensitvity,
        OSensitvity,
        Avoidance,
        Habitgene,
        Habitus,
        DriverStrategy,
        LR,
        Step,
    )
    setup_resources!(world, args)
    spawn_init_entities!(world, args.prediction_strategy)
    rebuild_occupancy!(world)
    rebuild_predicted_occupancy!(world, args.prediction_strategy)
    return world
end

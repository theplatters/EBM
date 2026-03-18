function setup_world(args)
    world = Ark.World(Position, PrevPosition, Direction, SSensitvity, OSensitvity, Avoidance, Habitgene, Habitus, LR, Step)
    setup_resources!(world, args)
    spawn_init_entities!(world)
    return world
end

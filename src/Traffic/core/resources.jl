mutable struct SimulationRNG
    rng::Random.Xoshiro
end

SimulationRNG(seed::Integer) = SimulationRNG(Random.Xoshiro(seed))

simulation_rng(world) = Ark.get_resource(world, SimulationRNG).rng

function setup_resources!(world, args::ModelArgs{T}) where {T}
    params = args.params
    ring = Ring(params.ring_x, params.ring_y)
    Ark.add_resource!(world, ring)

    Ark.add_resource!(world, PredictedOccupancy(ring))
    Ark.add_resource!(world, Occupancy(ring))

    Ark.add_resource!(world, MeanHabitus(0, 0))
    Ark.add_resource!(world, SimulationRNG(args.seed))

    Ark.add_resource!(world, params)
    Ark.add_resource!(world, args.weights)
    Ark.add_resource!(world, Logger())


    return nothing
end

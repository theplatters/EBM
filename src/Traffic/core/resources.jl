mutable struct SimulationRNG
    rng::Random.Xoshiro
end

SimulationRNG(seed::Integer) = SimulationRNG(Random.Xoshiro(seed))

simulation_rng(world) = Ark.get_resource(world, SimulationRNG).rng

"""
Signed, spatial information left by drivers that completed a movement safely.

Positive values indicate successful left-side use and negative values indicate
successful right-side use, both relative to the depositing driver's direction.
"""
mutable struct SuccessfulDriverTrace
    grid::Matrix{Float64}
end

SuccessfulDriverTrace(ring::Ring) = SuccessfulDriverTrace(
    zeros(Float64, Int(ring.width), Int(ring.height)),
)

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

function setup_resources!(world, args::ModelArgs{CapabilityModel})
    model = validate(args.prediction_strategy)
    params = args.params
    params.ring_x == 2 ||
        throw(ArgumentError("CapabilityModel currently requires a two-lane ring"))
    params.ring_y >= 2 * model.max_speed + 1 ||
        throw(ArgumentError("ring_y is too small for swept speed-aware movement"))
    params.init_agents <= params.ring_x * params.ring_y ||
        throw(ArgumentError("init_agents exceeds ring capacity"))

    ring = Ring(params.ring_x, params.ring_y)
    Ark.add_resource!(world, ring)
    Ark.add_resource!(world, Occupancy(ring))
    Ark.add_resource!(world, SuccessfulDriverTrace(ring))
    Ark.add_resource!(world, MeanHabitus(0.0, 0.0))
    Ark.add_resource!(world, SimulationRNG(args.seed))
    Ark.add_resource!(world, params)
    Ark.add_resource!(world, args.weights)
    Ark.add_resource!(world, model)
    Ark.add_resource!(world, Logger())
    return nothing
end

mutable struct SimulationRNG
    rng::Random.Xoshiro
end

SimulationRNG(seed::Integer) = SimulationRNG(Random.Xoshiro(seed))

mutable struct SimulationClock
    step::Int64
end

mutable struct NextCitizenId
    val::Int64
end

mutable struct SugarLandscape
    current::Matrix{Int64}
    capacity::Matrix{Int64}
end

mutable struct OccupancyGrid
    citizen_ids::Matrix{Int64}
end

mutable struct StepEvents
    moved::Int64
    conflicts::Int64
    harvested::Int64
    deaths::Int64
    starvation_deaths::Int64
    old_age_deaths::Int64
    replacements::Int64
    births::Int64
    infections::Int64
    recoveries::Int64
end

StepEvents() = StepEvents(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

function reset!(events::StepEvents)
    events.moved = 0
    events.conflicts = 0
    events.harvested = 0
    events.deaths = 0
    events.starvation_deaths = 0
    events.old_age_deaths = 0
    events.replacements = 0
    events.births = 0
    events.infections = 0
    events.recoveries = 0
    return nothing
end

simulation_rng(world) = Ark.get_resource(world, SimulationRNG).rng

function canonical_landscape(params::ModelParams)
    width = params.width
    height = params.height
    maximum = params.maximum_patch_sugar
    maximum == 0 && return zeros(Int64, width, height)

    peaks = (
        (max(1, round(Int, 0.3 * width)), max(1, round(Int, 0.3 * height))),
        (max(1, round(Int, 0.7 * width)), max(1, round(Int, 0.7 * height))),
    )
    band_width = max(1.0, min(width, height) / (2.0 * maximum))
    capacity = zeros(Int64, width, height)
    for x in 1:width, y in 1:height
        distance = minimum(hypot(x - peak_x, y - peak_y) for (peak_x, peak_y) in peaks)
        capacity[x, y] = clamp(maximum - floor(Int64, distance / band_width), 0, maximum)
    end
    return capacity
end

function setup_resources!(world, args::ModelArgs)
    params = args.params
    capacity = if isnothing(args.initial_capacity)
        canonical_landscape(params)
    else
        copy(args.initial_capacity)
    end
    Ark.add_resource!(world, params)
    Ark.add_resource!(world, SimulationRNG(args.seed))
    Ark.add_resource!(world, SimulationClock(0))
    Ark.add_resource!(world, NextCitizenId(1))
    Ark.add_resource!(world, SugarLandscape(copy(capacity), capacity))
    Ark.add_resource!(world, OccupancyGrid(zeros(Int64, params.width, params.height)))
    Ark.add_resource!(world, StepEvents())
    Ark.add_resource!(world, Logger())
    return nothing
end

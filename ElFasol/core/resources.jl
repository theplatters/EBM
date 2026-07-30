mutable struct SimulationRNG
    rng::Random.Xoshiro
end

SimulationRNG(seed::Integer) = SimulationRNG(Random.Xoshiro(seed))

mutable struct AttendanceHistory
    values::Vector{Int64}
end

mutable struct CurrentAttendance
    val::Int64
end

mutable struct SimulationClock
    step::Int64
end

mutable struct BestPredictions
    errors::Vector{Float64}
    tie_breakers::Vector{Float64}
    predictor_ids::Vector{Int64}
    forecasts::Vector{Float64}
end

function BestPredictions(population::Integer)
    return BestPredictions(
        fill(Inf, population),
        fill(Inf, population),
        fill(Int64(0), population),
        fill(NaN, population),
    )
end

function reset!(best::BestPredictions)
    fill!(best.errors, Inf)
    fill!(best.tie_breakers, Inf)
    fill!(best.predictor_ids, 0)
    fill!(best.forecasts, NaN)
    return nothing
end

simulation_rng(world) = Ark.get_resource(world, SimulationRNG).rng

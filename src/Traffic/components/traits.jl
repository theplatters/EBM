struct SSensitvity
    val::Float64
end

struct OSensitvity
    val::Float64
end

struct Avoidance
    val::Float64
end

struct Habitgene
    val::Float64
end

struct Habitus
    val::Float64
end

mutable struct MeanHabitus
    abs::Float64
    total::Float64
end

abstract type OccupancyStrategy end
struct PerEntityHabitusStrategy <: OccupancyStrategy end
struct MeanHabitusStrategy <: OccupancyStrategy end
struct NaiveStrategy <: OccupancyStrategy end
struct UnsureStrategy <: OccupancyStrategy end
struct RandomStrategy <: OccupancyStrategy end
struct SwitchStrategy <: OccupancyStrategy end

struct PredictedOccupancy
    grid::Matrix{Vector{Tuple{Direction, Entity, Float64}}}
end

function PredictedOccupancy(ring::Ring)
    return PredictedOccupancy(fill([], Int(ring.width), Int(ring.height)))
end

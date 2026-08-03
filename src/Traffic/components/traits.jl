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

"""Disposition accumulated from the driver's own realized lane history."""
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

"""
Use Naive's constant-lane forecast in two timeframes.

The driver changes lanes only when the decision score agrees when measured from
both its current position and its one-step-ahead position. The strategy uses
only current positions, directions, and the driver's own traits.
"""
struct TwoFrameNaiveStrategy <: OccupancyStrategy end

struct UnsureStrategy <: OccupancyStrategy end
struct RandomStrategy <: OccupancyStrategy end
struct SwitchStrategy <: OccupancyStrategy end

"""
Predict the lane choices induced by the model's own decision rule.

The forecast starts from current lanes and applies `iterations` damped
best-response updates. `damping` controls how far each update moves toward the
new lane-choice probability, which stabilizes mutually dependent predictions.
"""
struct DecisionAwareStrategy <: OccupancyStrategy
    iterations::Int
    damping::Float64

    function DecisionAwareStrategy(iterations::Integer, damping::Real)
        iterations > 0 || throw(ArgumentError("iterations must be positive"))
        0.0 < damping <= 1.0 || throw(ArgumentError("damping must be in (0, 1]"))
        return new(Int(iterations), Float64(damping))
    end
end

DecisionAwareStrategy(; iterations::Integer = 5, damping::Real = 0.5) =
    DecisionAwareStrategy(iterations, damping)

@enum StrategyKind begin
    PerEntityHabitusKind
    MeanHabitusKind
    NaiveKind
    TwoFrameNaiveKind
    UnsureKind
    RandomKind
    SwitchKind
    DecisionAwareKind
end

"""Compact per-car representation of an occupancy strategy."""
struct DriverStrategy
    kind::StrategyKind
    iterations::Int
    damping::Float64
end

DriverStrategy(::PerEntityHabitusStrategy) = DriverStrategy(PerEntityHabitusKind, 0, 0.0)
DriverStrategy(::MeanHabitusStrategy) = DriverStrategy(MeanHabitusKind, 0, 0.0)
DriverStrategy(::NaiveStrategy) = DriverStrategy(NaiveKind, 0, 0.0)
DriverStrategy(::TwoFrameNaiveStrategy) = DriverStrategy(TwoFrameNaiveKind, 0, 0.0)
DriverStrategy(::UnsureStrategy) = DriverStrategy(UnsureKind, 0, 0.0)
DriverStrategy(::RandomStrategy) = DriverStrategy(RandomKind, 0, 0.0)
DriverStrategy(::SwitchStrategy) = DriverStrategy(SwitchKind, 0, 0.0)
DriverStrategy(strategy::DecisionAwareStrategy) =
    DriverStrategy(DecisionAwareKind, strategy.iterations, strategy.damping)

const STRATEGY_NAMES = Dict(
    PerEntityHabitusKind => "Per-entity habitus",
    MeanHabitusKind => "Mean habitus",
    NaiveKind => "Naive",
    TwoFrameNaiveKind => "Two-frame naive",
    UnsureKind => "Unsure",
    RandomKind => "Random",
    SwitchKind => "Switch",
    DecisionAwareKind => "Decision-aware",
)

strategy_name(strategy::DriverStrategy) = STRATEGY_NAMES[strategy.kind]
strategy_name(strategy::OccupancyStrategy) = strategy_name(DriverStrategy(strategy))

"""
    HeterogeneousStrategy(strategy => share, ...)

Assign different occupancy strategies to cars using deterministic, stratified
counts followed by a seeded shuffle. Each car contributes its own forecast to
the shared occupancy field. A replacement inherits the strategy and direction
of the car it replaces, keeping the population composition fixed.
"""
struct HeterogeneousStrategy <: OccupancyStrategy
    strategies::Vector{DriverStrategy}
    shares::Vector{Float64}

    function HeterogeneousStrategy(
            strategies::Vector{DriverStrategy}, shares::Vector{Float64},
        )
        length(strategies) == length(shares) ||
            throw(ArgumentError("strategies and shares must have the same length"))
        isempty(strategies) && throw(ArgumentError("at least one strategy is required"))
        all(>(0.0), shares) || throw(ArgumentError("strategy shares must be positive"))
        isapprox(sum(shares), 1.0; atol = 1.0e-10) ||
            throw(ArgumentError("strategy shares must sum to one"))
        length(unique(strategy.kind for strategy in strategies)) == length(strategies) ||
            throw(ArgumentError("each strategy kind may appear only once"))
        return new(strategies, shares)
    end
end

function HeterogeneousStrategy(pairs::Pair...)
    all(pair -> pair.first isa OccupancyStrategy && pair.second isa Real, pairs) ||
        throw(ArgumentError("expected OccupancyStrategy => share pairs"))
    return HeterogeneousStrategy(
        DriverStrategy[pair.first |> DriverStrategy for pair in pairs],
        Float64[pair.second for pair in pairs],
    )
end

function HeterogeneousStrategy(strategies::OccupancyStrategy...)
    isempty(strategies) && throw(ArgumentError("at least one strategy is required"))
    share = 1.0 / length(strategies)
    return HeterogeneousStrategy((strategy => share for strategy in strategies)...)
end

struct PredictedOccupancy
    grid::Matrix{Vector{Tuple{Direction, Entity, Float64}}}
end

function PredictedOccupancy(ring::Ring)
    grid = [
        Tuple{Direction, Entity, Float64}[]
            for _ in 1:Int(ring.width), _ in 1:Int(ring.height)
    ]
    return PredictedOccupancy(grid)
end

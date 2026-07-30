mutable struct DividendRNG
    rng::Random.Xoshiro
end

mutable struct LearningRNG
    rng::Random.Xoshiro
end

mutable struct SimulationClock
    step::Int64
end

mutable struct MarketState
    price::Float64
    previous_price::Float64
    dividend::Float64
    fundamental_price::Float64
    volume::Float64
end

mutable struct MarketHistory
    prices::Vector{Float64}
    dividends::Vector{Float64}
    fundamental_prices::Vector{Float64}
    volumes::Vector{Float64}
end

mutable struct MarketDescriptors
    values::NTuple{DESCRIPTOR_COUNT, Bool}
end

mutable struct BestForecasts
    errors::Vector{Float64}
    tie_breakers::Vector{Float64}
    predictor_ids::Vector{Int64}
    forecasts::Vector{Float64}
    variances::Vector{Float64}
end

function BestForecasts(population::Integer)
    return BestForecasts(
        fill(Inf, population),
        fill(Inf, population),
        fill(Int64(0), population),
        fill(NaN, population),
        fill(NaN, population),
    )
end

function reset!(best::BestForecasts)
    fill!(best.errors, Inf)
    fill!(best.tie_breakers, Inf)
    fill!(best.predictor_ids, 0)
    fill!(best.forecasts, NaN)
    fill!(best.variances, NaN)
    return nothing
end

struct PredictorRegistry
    entities::Vector{Vector{Ark.Entity}}
end

mutable struct EvolutionSchedule
    next_steps::Vector{Int64}
end

mutable struct EvolutionEvents
    count::Int64
end

dividend_rng(world) = Ark.get_resource(world, DividendRNG).rng
learning_rng(world) = Ark.get_resource(world, LearningRNG).rng
total_share_supply(params::ModelParams) = params.population * params.shares_per_trader

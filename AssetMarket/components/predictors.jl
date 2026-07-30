const DESCRIPTOR_COUNT = 12
const WILDCARD = Int8(-1)

struct PredictorId
    val::Int64
end

struct PredictorOwner
    val::Int64
end

struct ConditionBits
    values::NTuple{DESCRIPTOR_COUNT, Int8}
end

struct ForecastParameters
    a::Float64
    b::Float64
end

struct PredictorForecast
    val::Float64
end

struct ForecastErrorVariance
    val::Float64
end

struct DemandVariance
    val::Float64
end

struct Matched
    val::Bool
end

struct DefaultPredictor
    val::Bool
end

struct TieBreaker
    val::Float64
end

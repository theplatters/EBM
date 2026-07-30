struct PredictorId
    val::Int64
end

struct PredictorOwner
    val::Int64
end

struct Forecast
    val::Float64
end

struct SquaredError
    val::Float64
end

struct TieBreaker
    val::Float64
end

"""Forecast the attendance observed `lag` periods ago."""
struct LagPredictor
    lag::Int64
end

"""Forecast the arithmetic mean over the most recent `window` periods."""
struct MeanPredictor
    window::Int64
end

"""Reflect lagged attendance around `center`."""
struct MirrorPredictor
    center::Float64
    lag::Int64
end

"""Extrapolate a least-squares linear trend fitted to a recent window."""
struct TrendPredictor
    window::Int64
end

"""Always issue the same attendance forecast."""
struct ConstantPredictor
    value::Float64
end

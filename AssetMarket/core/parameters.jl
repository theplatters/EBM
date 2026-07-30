Base.@kwdef struct ModelParams
    population::Int64 = 25
    predictors_per_trader::Int64 = 100
    shares_per_trader::Float64 = 1.0
    interest_rate::Float64 = 0.1
    risk_aversion::Float64 = 0.5
    dividend_mean::Float64 = 10.0
    dividend_persistence::Float64 = 0.95
    dividend_variance::Float64 = 0.0743
    initial_forecast_variance::Float64 = 4.0
    minimum_forecast_variance::Float64 = 0.05
    accuracy_rate::Float64 = 1 / 75
    evolution_interval::Float64 = 250.0
    replacement_fraction::Float64 = 0.20
    crossover_probability::Float64 = 0.10
    condition_probability::Float64 = 0.10
    bit_mutation_probability::Float64 = 0.03
    a_mutation_std::Float64 = 0.05
    b_mutation_std::Float64 = 1.0
    specificity_cost::Float64 = 0.005
    history_length::Int64 = 500
    price_floor::Float64 = 0.01
end

Base.@kwdef struct ModelArgs
    seed::Int64 = 42
    params::ModelParams = ModelParams()
    steps::Int64 = 1_000
end

complex_market_params(; kwargs...) = ModelParams(; kwargs...)

slow_market_params(; kwargs...) = ModelParams(
    accuracy_rate = 1 / 150,
    evolution_interval = 1_000.0,
    crossover_probability = 0.30;
    kwargs...,
)

function validate(params::ModelParams)
    params.population > 0 || throw(ArgumentError("population must be positive"))
    params.predictors_per_trader >= 2 ||
        throw(ArgumentError("predictors_per_trader must be at least 2"))
    params.shares_per_trader > 0.0 ||
        throw(ArgumentError("shares_per_trader must be positive"))
    params.interest_rate > 0.0 || throw(ArgumentError("interest_rate must be positive"))
    params.risk_aversion > 0.0 || throw(ArgumentError("risk_aversion must be positive"))
    0.0 <= params.dividend_persistence < 1.0 ||
        throw(ArgumentError("dividend_persistence must be in [0, 1)"))
    params.dividend_variance > 0.0 ||
        throw(ArgumentError("dividend_variance must be positive"))
    params.initial_forecast_variance > 0.0 ||
        throw(ArgumentError("initial_forecast_variance must be positive"))
    params.minimum_forecast_variance > 0.0 ||
        throw(ArgumentError("minimum_forecast_variance must be positive"))
    0.0 < params.accuracy_rate <= 1.0 ||
        throw(ArgumentError("accuracy_rate must be in (0, 1]"))
    params.evolution_interval > 0.0 ||
        throw(ArgumentError("evolution_interval must be positive"))
    0.0 < params.replacement_fraction < 1.0 ||
        throw(ArgumentError("replacement_fraction must be in (0, 1)"))
    0.0 <= params.crossover_probability <= 1.0 ||
        throw(ArgumentError("crossover_probability must be in [0, 1]"))
    0.0 <= params.condition_probability <= 1.0 ||
        throw(ArgumentError("condition_probability must be in [0, 1]"))
    0.0 <= params.bit_mutation_probability <= 1.0 ||
        throw(ArgumentError("bit_mutation_probability must be in [0, 1]"))
    params.history_length >= 500 || throw(ArgumentError("history_length must be at least 500"))
    params.price_floor > 0.0 || throw(ArgumentError("price_floor must be positive"))
    return nothing
end

function validate(args::ModelArgs)
    validate(args.params)
    args.steps >= 0 || throw(ArgumentError("steps cannot be negative"))
    return nothing
end

function homogeneous_equilibrium(params::ModelParams)
    denominator = 1.0 + params.interest_rate - params.dividend_persistence
    price_slope = params.dividend_persistence / denominator
    payoff_variance = (1.0 + price_slope)^2 * params.dividend_variance
    price_intercept = (
        (1.0 + price_slope) * (1.0 - params.dividend_persistence) *
        params.dividend_mean - params.risk_aversion * payoff_variance
    ) / params.interest_rate
    forecast_a = params.dividend_persistence
    forecast_b = (1.0 - params.dividend_persistence) * (
        (1.0 + price_slope) * params.dividend_mean + price_intercept
    )
    return (
        price_slope = price_slope,
        price_intercept = price_intercept,
        payoff_variance = payoff_variance,
        forecast_a = forecast_a,
        forecast_b = forecast_b,
    )
end

function fundamental_price(params::ModelParams, dividend::Real)
    equilibrium = homogeneous_equilibrium(params)
    return equilibrium.price_slope * dividend + equilibrium.price_intercept
end

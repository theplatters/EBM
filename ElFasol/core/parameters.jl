Base.@kwdef struct ModelParams
    population::Int64 = 100
    capacity::Int64 = 60
    predictors_per_agent::Int64 = 12
    history_length::Int64 = 20
    successful_attendance_payoff::Float64 = 1.0
    successful_absence_payoff::Float64 = 1.0
end

Base.@kwdef struct ModelArgs
    seed::Int64 = 42
    params::ModelParams = ModelParams()
    steps::Int64 = 100
    initial_history::Union{Nothing, Vector{Int64}} = nothing
end

function validate(params::ModelParams)
    params.population > 0 || throw(ArgumentError("population must be positive"))
    0 < params.capacity <= params.population ||
        throw(ArgumentError("capacity must be between 1 and population"))
    params.predictors_per_agent > 0 ||
        throw(ArgumentError("predictors_per_agent must be positive"))
    params.history_length >= 2 || throw(ArgumentError("history_length must be at least 2"))
    params.successful_attendance_payoff >= 0.0 ||
        throw(ArgumentError("successful_attendance_payoff cannot be negative"))
    params.successful_absence_payoff >= 0.0 ||
        throw(ArgumentError("successful_absence_payoff cannot be negative"))
    return nothing
end

function validate(args::ModelArgs)
    validate(args.params)
    args.steps >= 0 || throw(ArgumentError("steps cannot be negative"))
    history = args.initial_history
    isnothing(history) && return nothing
    length(history) >= args.params.history_length || throw(
        ArgumentError("initial_history must contain at least history_length values"),
    )
    all(0 <= attendance <= args.params.population for attendance in history) ||
        throw(ArgumentError("initial_history values must be between 0 and population"))
    return nothing
end

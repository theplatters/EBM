@enum MovementMode begin
    ShuffledSequentialMovement
    SynchronousMovement
end

Base.@kwdef struct ModelParams
    width::Int64 = 50
    height::Int64 = 50
    population::Int64 = 400
    maximum_patch_sugar::Int64 = 4
    growback_rate::Int64 = 1
    minimum_vision::Int64 = 1
    maximum_vision::Int64 = 6
    minimum_metabolism::Int64 = 1
    maximum_metabolism::Int64 = 4
    minimum_initial_sugar::Int64 = 5
    maximum_initial_sugar::Int64 = 25
    minimum_lifespan::Int64 = 60
    maximum_lifespan::Int64 = 100
    replace_dead::Bool = true
    movement_mode::MovementMode = ShuffledSequentialMovement
    reproduction_enabled::Bool = false
    minimum_fertility_age::Int64 = 12
    maximum_fertility_age::Int64 = 50
    reproduction_probability::Float64 = 0.05
    initial_infection_probability::Float64 = 0.0
    disease_transmission_probability::Float64 = 0.15
    disease_duration::Int64 = 15
    disease_sugar_cost::Int64 = 1
end

Base.@kwdef struct ModelArgs
    seed::Int64 = 42
    params::ModelParams = ModelParams()
    steps::Int64 = 100
    initial_capacity::Union{Nothing, Matrix{Int64}} = nothing
end

function validate(params::ModelParams)
    params.width > 0 || throw(ArgumentError("width must be positive"))
    params.height > 0 || throw(ArgumentError("height must be positive"))
    0 < params.population <= params.width * params.height ||
        throw(ArgumentError("population must be between 1 and the number of cells"))
    params.maximum_patch_sugar >= 0 ||
        throw(ArgumentError("maximum_patch_sugar cannot be negative"))
    params.growback_rate >= 0 || throw(ArgumentError("growback_rate cannot be negative"))
    0 <= params.minimum_vision <= params.maximum_vision ||
        throw(ArgumentError("vision bounds must be ordered and nonnegative"))
    0 <= params.minimum_metabolism <= params.maximum_metabolism ||
        throw(ArgumentError("metabolism bounds must be ordered and nonnegative"))
    0 <= params.minimum_initial_sugar <= params.maximum_initial_sugar ||
        throw(ArgumentError("initial sugar bounds must be ordered and nonnegative"))
    0 <= params.minimum_lifespan <= params.maximum_lifespan ||
        throw(ArgumentError("lifespan bounds must be ordered and nonnegative"))
    params.reproduction_enabled && params.replace_dead && throw(
        ArgumentError("reproduction_enabled and replace_dead cannot both be true"),
    )
    0 <= params.minimum_fertility_age <= params.maximum_fertility_age ||
        throw(ArgumentError("fertility-age bounds must be ordered and nonnegative"))
    0.0 <= params.reproduction_probability <= 1.0 ||
        throw(ArgumentError("reproduction_probability must be between zero and one"))
    0.0 <= params.initial_infection_probability <= 1.0 || throw(
        ArgumentError("initial_infection_probability must be between zero and one"),
    )
    0.0 <= params.disease_transmission_probability <= 1.0 || throw(
        ArgumentError("disease_transmission_probability must be between zero and one"),
    )
    params.disease_duration > 0 || throw(ArgumentError("disease_duration must be positive"))
    params.disease_sugar_cost >= 0 ||
        throw(ArgumentError("disease_sugar_cost cannot be negative"))
    return nothing
end

function validate(args::ModelArgs)
    validate(args.params)
    args.steps >= 0 || throw(ArgumentError("steps cannot be negative"))
    isnothing(args.initial_capacity) && return nothing
    size(args.initial_capacity) == (args.params.width, args.params.height) ||
        throw(ArgumentError("initial_capacity dimensions must match width and height"))
    all(>=(0), args.initial_capacity) ||
        throw(ArgumentError("initial_capacity cannot contain negative values"))
    return nothing
end

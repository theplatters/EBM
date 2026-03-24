Base.@kwdef struct ModelArgs{T <: OccupancyStrategy}
    seed::Union{Nothing, Int} = nothing
    params::ModelParams = ModelParams()
    weights::Weights = Weights()
    prediction_strategy::T = PerEntityHabitusStrategy
    steps::Int64 = 100
end

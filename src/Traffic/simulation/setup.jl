Base.@kwdef struct ModelArgs{T <: OccupancyStrategy}
    seed::Int64 = rand(Int64)
    params::ModelParams = ModelParams()
    weights::Weights = Weights()
    prediction_strategy::T = T()
    steps::Int64 = 100
end

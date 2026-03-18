Base.@kwdef struct Weights
    wₛ::Float64 = 1.4
    wₒ::Float64 = 0.9
    wₐ::Float64 = 0.7
    wₕ::Float64 = 0.12
end

Base.@kwdef struct ModelParams
    δ::Float64 = 0.1
    ϵ::Float64 = 0.01
    init_agents::Int64 = 40
    K::Float64 = 10.0
    lookahead::Int64 = 30
    ring_x::Int64 = 2
    ring_y::Int64 = 100
end


Base.@kwdef struct ModelArgs
    seed::Union{Nothing, Int} = nothing
    params::ModelParams = ModelParams()
    weights::Weights = Weights()
end

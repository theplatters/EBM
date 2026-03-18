Base.@kwdef struct Weights
    wₛ::Float64 = 0.1
    wₒ::Float64 = 0.1
    wₐ::Float64 = 0.3
    wₕ::Float64 = 0.5
end

Base.@kwdef struct ModelParams
    δ::Float64 = 0.1
    ϵ::Float64 = 0.01
    init_agents::Int64 = 40
    K::Float64 = 2.0
    lookahead::Int64 = 10
    ring_x::Int64 = 2
    ring_y::Int64 = 100
end


Base.@kwdef struct ModelArgs
    seed::Union{Nothing, Int} = nothing
    params::ModelParams = ModelParams()
    weights::Weights = Weights()
end

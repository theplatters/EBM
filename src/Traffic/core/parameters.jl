Base.@kwdef struct Weights
    wₛ::Float64 = 0.5
    wₒ::Float64 = 0.5
    wₐ::Float64 = 0.5
    wₕ::Float64 = 1.0
end

Base.@kwdef struct ModelParams
    δ::Float64 = 0.2
    ϵ::Float64 = 0.01
    init_agents::Int64 = 40
    K::Float64 = 10.0
    lookahead::Int64 = 20
    ring_x::Int64 = 2
    ring_y::Int64 = 100
end

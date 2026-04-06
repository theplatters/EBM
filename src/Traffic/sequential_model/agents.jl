using Agents
using Random
using Distributions

include("../core/parameters.jl")
include("../simulation/setup.jl")


@enum Direction begin
    Clockwise = 1
    Counterclockwise = -1
end

struct Sensitivities
    s_sensitvity::Float64
    o_sensitvity::Float64
    avoidance::Float64
    habitgene::Float64
end

function normal_sensitivities(rng, ; μ = 1.0, σ = 0.2)
    draws::Vector{Float64} = rand(rng, Normal(μ, σ), 4)
    return Sensitivities(draws[1], draws[2], draws[3], draws[4])
end

@agent struct Car(GridAgent{2})
    lr::Float64
    sensitivities::Sensitivities
    habitus::Float64
    weights::Weights
    direction::Direction
end


function calculate_lr!(agent, model)

    (;
        s_sensitvity,
        o_sensitvity,
        avoidance,
        habitgene,
    ) = agent.sensitivities
    weights = agent.weights
    SL, OL, CL, CR = compute_observations(
        agent,
        model
    )

    s_val = weights.wₛ * s_sensitvity * (2 * SL - 1)
    o_val = weights.wₒ * o_sensitvity * (2 * OL - 1)
    avoidance_val = weights.wₐ * avoidance * (CR - CL)
    habit_strength = weights.wₕ * habitgene * agent.habit

    agent.lr[i] = s_val + o_val + avoidance_val + habit_strength
    return nothing
end

function move!(agent, model)

    new_pos = agent.pos .+ (0, Int(agent.direction))
    go_left = agent.lr > 0.0

    if rand(abmrng(model)) < model.params.ϵ
        go_left = !go_left
    end
    new_pos[1] = agent.direction == Clockwise ? (go_left ? 1 : 2) : (go_left ? 2 : 1)

    move_agent!(agent, normalize_position(new_pos, model))
    return nothing
end

function update_habitus!(agent, model)
end


function car_step!(agent, model)
    calculate_lr!(agent, model)
    move!(agent, model)
    update_habitus!(agent, model)
    return nothing
end

function init_model(params, weights)

    model = Agents.StandardABM(Car, GridSpace((2, 100)), agent_step! = car_step!)

    directions = shuffle(abmrng(model), repeat([Clockwise, Counterclockwise], params.init_agents ÷ 2))
    for i in 1:params.init_agents
        add_agent!(
            model;
            lr = 0.0,
            sensitivities = normal_sensitivities(abmrng(model)),
            habitus = 0.0,
            weights = weights,
            direction = directions[i]
        )
    end

    return model
end


init_model(ModelParams(), Weights())

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
    age::Int64
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
    habit_strength = weights.wₕ * habitgene * agent.habitus

    agent.lr = s_val + o_val + avoidance_val + habit_strength
    return nothing
end


function move!(agent, model)
    go_left = agent.lr > 0.0

    if rand(abmrng(model)) < model.params.ϵ
        go_left = !go_left
    end

    y = agent.pos[2] + Int(agent.direction)
    x = agent.direction == Clockwise ? (go_left ? 1 : 2) : (go_left ? 2 : 1)
    new_pos = normalize_position((x, y), model)

    # Check whether someone is already at the target position
    occupants = agents_in_position(new_pos, model)

    if !isempty(occupants)
        # kill the moving agent and all agents at destination
        ids_to_remove = [agent.id; [a.id for a in occupants]...]
        for id in ids_to_remove
            remove_agent!(id, model)
        end

        directions = shuffle(abmrng(model), [Clockwise, Counterclockwise])
        # spawn two new random agents at empty positions
        for i in 1:2
            pos = random_empty(model)
            add_agent!(
                pos,
                model;
                lr = 0.0,
                sensitivities = normal_sensitivities(abmrng(model)),
                habitus = 0.0,
                weights = agent.weights,
                direction = directions[i],
                age = 0
            )
        end

        return false
    end

    move_agent!(agent, new_pos, model)
    return true
end


@inline relative_lane_sign(x::Int, dir::Direction) =
    dir == Clockwise ?
    (x == 1 ? 1.0 : -1.0) :
    (x == 2 ? 1.0 : -1.0)

function update_habitus!(agent, model)
    agent.habitus = clamp(
        agent.habitus + relative_lane_sign(agent.pos[1], agent.direction) / (model.params.K + agent.age),
        -1.0,
        1.0,
    )
    return nothing
end


function car_step!(agent, model)
    calculate_lr!(agent, model)
    if move!(agent, model)
        update_habitus!(agent, model)
        agent.age += 1
    end
    return nothing
end

function init_model(params, weights)

    model = StandardABM(
        Car,
        GridSpace((2, 100));
        agent_step! = car_step!,
        properties = (params = params,)
    )

    directions = shuffle(abmrng(model), repeat([Clockwise, Counterclockwise], params.init_agents ÷ 2))
    for i in 1:params.init_agents
        add_agent!(
            model;
            lr = 0.0,
            sensitivities = normal_sensitivities(abmrng(model)),
            habitus = 0.0,
            weights = weights,
            direction = directions[i],
            age = 0
        )
    end

    return model
end


m = init_model(ModelParams(), Weights())

run_model(m, 20)

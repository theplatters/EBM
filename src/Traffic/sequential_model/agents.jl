module SequentialModel
using Agents
using Random
using Distributions

using ..Traffic: Weights
using ..Traffic: ModelParams

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

function normal_sensitivities(rng; μ = 1.0, σ = 0.2)
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


function next_agents_in_direction(agent, model, lookahead)
    found = Tuple{typeof(agent), Int}[]
    x, y = agent.pos
    _, h = spacesize(model)

    dy = agent.direction == Clockwise ? 1 : -1
    yy = y

    for d in 0:lookahead
        d > 0 && (yy = mod1(yy + dy, h))
        for lane in 1:2
            for other in agents_in_position((lane, yy), model)
                other.id == agent.id && continue
                push!(found, (other, d))
            end
        end
    end

    return found
end


is_left_relative(x::Int, dir::Direction) =
    (dir == Clockwise) ? (x == 1) : (x == 2)


function forward_distance(agent, other, model)
    _, h = spacesize(model)
    y0 = agent.pos[2]
    y1 = other.pos[2]

    if agent.direction == Clockwise
        return mod1(y1 - y0, h)
    else
        return mod1(y0 - y1, h)
    end
end


function compute_observations(agent, model)
    same_total = 0
    same_left = 0
    opp_total = 0
    opp_left = 0

    CL = 0
    CR = 0

    for (other, d) in next_agents_in_direction(agent, model, model.params.lookahead)
        left_rel = is_left_relative(other.pos[1], agent.direction)

        if other.direction == agent.direction
            same_total += 1
            same_left += left_rel ? 1 : 0
        else
            opp_total += 1
            opp_left += left_rel ? 1 : 0
        end

        if d <= 2
            if left_rel
                CL += 1
            else
                CR += 1
            end
        end
    end

    SL = same_total == 0 ? 0.5 : same_left / same_total
    OL = opp_total == 0 ? 0.5 : opp_left / opp_total

    return SL, OL, CL, CR
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
    o_val = -weights.wₒ * o_sensitvity * (2 * OL - 1)
    avoidance_val = weights.wₐ * avoidance * (CR - CL)
    habit_strength = weights.wₕ * habitgene * agent.habitus

    agent.lr = s_val + o_val + avoidance_val + habit_strength
    return nothing
end


function intended_position(agent, model)
    go_left = if agent.lr > 0.0
        true
    elseif agent.lr < 0.0
        false
    else
        is_left_relative(agent.pos[1], agent.direction)
    end

    if rand(abmrng(model)) < model.params.ϵ
        go_left = !go_left
    end

    y = agent.pos[2] + Int(agent.direction)
    x = agent.direction == Clockwise ? (go_left ? 1 : 2) : (go_left ? 2 : 1)
    return normalize_position((x, y), model)
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

function collision_ids(previous_positions, intended_positions)
    final_occupancy = Dict{NTuple{2, Int}, Vector{Int}}()
    for (id, position) in intended_positions
        push!(get!(final_occupancy, position, Int[]), id)
    end

    killed = Set{Int}()
    for ids in values(final_occupancy)
        length(ids) > 1 && union!(killed, ids)
    end

    ids = sort!(collect(keys(previous_positions)))
    for i in 1:(length(ids) - 1)
        id_a = ids[i]
        previous_a = previous_positions[id_a]
        intended_a = intended_positions[id_a]

        for j in (i + 1):length(ids)
            id_b = ids[j]
            previous_b = previous_positions[id_b]
            intended_b = intended_positions[id_b]

            exact_swap = intended_a == previous_b && intended_b == previous_a
            diagonal_crossing =
                previous_a[2] == previous_b[2] &&
                intended_a[2] == intended_b[2] &&
                previous_a[1] != previous_b[1] &&
                intended_a[1] != intended_b[1] &&
                previous_a[1] == intended_b[1] &&
                previous_b[1] == intended_a[1]

            (exact_swap || diagonal_crossing) && union!(killed, (id_a, id_b))
        end
    end

    return killed
end

function spawn_replacements!(model, directions, weights)
    isempty(directions) && return nothing

    rng = abmrng(model)
    shuffled_directions = shuffle(rng, collect(directions))
    draws = rand(rng, Normal(1.0, model.params.δ), length(directions), 4)

    for i in eachindex(shuffled_directions)
        add_agent!(
            random_empty(model),
            model;
            lr = 0.0,
            sensitivities = Sensitivities(draws[i, 1], draws[i, 2], draws[i, 3], draws[i, 4]),
            habitus = 0.0,
            weights = weights,
            direction = shuffled_directions[i],
            age = 1,
        )
    end
    return nothing
end

"""Advance every scheduled car once, then resolve the completed trajectories."""
function sequential_step!(model)
    ids = sort!(collect(allids(model)))
    isempty(ids) && return nothing

    previous_positions = Dict(id => model[id].pos for id in ids)
    for id in ids
        agent = model[id]
        calculate_lr!(agent, model)
        move_agent!(agent, intended_position(agent, model), model)
    end

    intended_positions = Dict(id => model[id].pos for id in ids)
    killed = collision_ids(previous_positions, intended_positions)
    killed_directions = [model[id].direction for id in ids if id in killed]
    weights = model[first(ids)].weights

    for id in ids
        id in killed && continue
        agent = model[id]
        agent.age += 1
        update_habitus!(agent, model)
    end

    for id in killed
        remove_agent!(id, model)
    end
    spawn_replacements!(model, killed_directions, weights)
    return nothing
end

function initial_positions(rng, params)
    total = params.ring_x * params.ring_y
    params.init_agents <= total ||
        throw(ArgumentError("init_agents=$(params.init_agents) exceeds ring capacity=$total"))
    indices = randperm(rng, total)[1:params.init_agents]
    return [
        (mod1(index, params.ring_x), (index - 1) ÷ params.ring_x + 1)
            for index in indices
    ]
end

function init_model(params, weights; seed::Integer = rand(Int64))

    model = StandardABM(
        Car,
        GridSpace((params.ring_x, params.ring_y));
        model_step! = sequential_step!,
        properties = (params = params,),
        rng = Random.Xoshiro(seed),
    )

    rng = abmrng(model)
    positions = initial_positions(rng, params)
    directions = shuffle(rng, repeat([Clockwise, Counterclockwise], params.init_agents ÷ 2))
    draws = rand(rng, Normal(1.0, params.δ), params.init_agents, 4)
    for i in 1:params.init_agents
        add_agent!(
            positions[i],
            model;
            lr = 0.0,
            sensitivities = Sensitivities(draws[i, 1], draws[i, 2], draws[i, 3], draws[i, 4]),
            habitus = 0.0,
            weights = weights,
            direction = directions[i],
            age = 1,
        )
    end

    return model
end


end

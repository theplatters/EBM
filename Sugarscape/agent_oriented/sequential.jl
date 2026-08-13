module AgentSequential

using Random
import Agents
using Agents: @agent,
    ABM,
    GridAgent,
    GridSpace,
    StandardABM,
    abmrng,
    add_agent!,
    allids,
    ids_in_position,
    move_agent!,
    remove_agent!

using ..Sugarscape: CitizenState,
    Infection,
    Logger,
    ModelArgs,
    ModelParams,
    Position,
    ShuffledSequentialMovement,
    SimulationClock,
    StepEvents,
    SugarLandscape,
    canonical_landscape,
    gini_coefficient,
    median_value,
    reset!,
    validate

export Citizen,
    citizen_snapshot,
    landscape,
    logger,
    run_model,
    setup_model,
    step!

"""A mutable Agents.jl citizen carrying all individual Sugarscape state."""
@agent struct Citizen(GridAgent{2})
    vision::Int64
    metabolism::Int64
    sugar::Int64
    age::Int64
    maximum_age::Int64
    initial_endowment::Int64
    sex::Symbol
    immune_bits::UInt64
    infection::Union{Nothing, Infection}
end

struct ReproductiveRecord
    id::Int64
    position::Position
    female::Bool
    vision::Int64
    metabolism::Int64
    sugar::Int64
    maximum_age::Int64
    initial_endowment::Int64
end

struct BirthSpec
    position::Position
    mother_id::Int64
    father_id::Int64
    mother_contribution::Int64
    father_contribution::Int64
    vision::Int64
    metabolism::Int64
    maximum_age::Int64
    initial_endowment::Int64
    sex::Symbol
end

logger(model) = model.logger
landscape(model) = model.landscape

to_position(position::NTuple{2, Int}) = Position(position...)
to_tuple(position::Position) = (position.x, position.y)

function initial_positions(params::ModelParams, rng)
    cell_indices = randperm(rng, params.width * params.height)[1:params.population]
    return [
        Position(mod1(index, params.width), (index - 1) ÷ params.width + 1)
        for index in cell_indices
    ]
end

function spawn_citizen!(
    model,
    position::Position;
    vision = nothing,
    metabolism = nothing,
    sugar = nothing,
    maximum_age = nothing,
    initial_endowment = nothing,
    sex = nothing,
    immune_bits = nothing,
)
    params = model.params
    rng = abmrng(model)
    citizen_sugar = isnothing(sugar) ?
                    rand(rng, params.minimum_initial_sugar:params.maximum_initial_sugar) : sugar
    endowment = isnothing(initial_endowment) ? citizen_sugar : initial_endowment
    citizen_sex = isnothing(sex) ? (rand(rng, Bool) ? :female : :male) : sex
    agent = add_agent!(
        to_tuple(position),
        model;
        vision = isnothing(vision) ?
                 rand(rng, params.minimum_vision:params.maximum_vision) : vision,
        metabolism = isnothing(metabolism) ?
                     rand(rng, params.minimum_metabolism:params.maximum_metabolism) : metabolism,
        sugar = citizen_sugar,
        age = 0,
        maximum_age = isnothing(maximum_age) ?
                      rand(rng, params.minimum_lifespan:params.maximum_lifespan) : maximum_age,
        initial_endowment = endowment,
        sex = citizen_sex,
        immune_bits = isnothing(immune_bits) ? rand(rng, UInt64) : immune_bits,
        infection = nothing,
    )
    return agent.id
end

function spawn_initial_population!(model)
    rng = abmrng(model)
    for (index, position) in enumerate(initial_positions(model.params, rng))
        spawn_citizen!(model, position; sex = isodd(index) ? :female : :male)
    end
    return nothing
end

function seed_initial_infections!(model)
    probability = model.params.initial_infection_probability
    probability == 0.0 && return nothing
    rng = abmrng(model)
    for id in sort!(collect(allids(model)))
        agent = model[id]
        rand(rng) < probability || continue
        strain = rand(rng, UInt64)
        strain == agent.immune_bits && (strain = ~strain)
        agent.infection = Infection(strain, 0)
    end
    return nothing
end

function visible_cells(position::Position, vision::Integer, params::ModelParams)
    distances = Dict{Position, Int64}(position => 0)
    for distance in 1:vision
        candidates = (
            Position(mod1(position.x + distance, params.width), position.y),
            Position(mod1(position.x - distance, params.width), position.y),
            Position(position.x, mod1(position.y + distance, params.height)),
            Position(position.x, mod1(position.y - distance, params.height)),
        )
        for candidate in candidates
            distances[candidate] = min(
                get(distances, candidate, typemax(Int64)),
                distance,
            )
        end
    end
    return distances
end

function available_to(candidate::Position, citizen_id::Integer, model)
    occupants = ids_in_position(to_tuple(candidate), model)
    return isempty(occupants) || (length(occupants) == 1 && only(occupants) == citizen_id)
end

function select_destination(agent::Citizen, model)
    position = to_position(agent.pos)
    best_sugar = typemin(Int64)
    best_distance = typemax(Int64)
    best = Position[]
    cells = collect(visible_cells(position, agent.vision, model.params))
    sort!(cells; by = pair -> (last(pair), first(pair).x, first(pair).y))
    for (candidate, distance) in cells
        available_to(candidate, agent.id, model) || continue
        patch_sugar = model.landscape.current[candidate.x, candidate.y]
        if patch_sugar > best_sugar ||
                (patch_sugar == best_sugar && distance < best_distance)
            best_sugar = patch_sugar
            best_distance = distance
            empty!(best)
            push!(best, candidate)
        elseif patch_sugar == best_sugar && distance == best_distance
            push!(best, candidate)
        end
    end
    isempty(best) && return position
    return best[rand(abmrng(model), eachindex(best))]
end

function growback!(model)
    rate = model.params.growback_rate
    iszero(rate) && return nothing
    sugar_landscape = model.landscape
    @inbounds for index in eachindex(sugar_landscape.current)
        sugar_landscape.current[index] = min(
            sugar_landscape.capacity[index],
            sugar_landscape.current[index] + rate,
        )
    end
    return nothing
end

function move_and_harvest!(model)
    ids = sort!(collect(allids(model)))
    shuffle!(abmrng(model), ids)
    for id in ids
        agent = model[id]
        origin = to_position(agent.pos)
        destination = select_destination(agent, model)
        move_agent!(agent, to_tuple(destination), model)
        harvest = model.landscape.current[destination.x, destination.y]
        model.landscape.current[destination.x, destination.y] = 0
        agent.sugar += harvest
        model.events.moved += destination != origin
        model.events.harvested += harvest
    end
    return nothing
end

function neighboring_positions(position::Position, params::ModelParams)
    neighbors = Position[
        Position(mod1(position.x + 1, params.width), position.y),
        Position(mod1(position.x - 1, params.width), position.y),
        Position(position.x, mod1(position.y + 1, params.height)),
        Position(position.x, mod1(position.y - 1, params.height)),
    ]
    unique!(neighbors)
    filter!(!=(position), neighbors)
    sort!(neighbors; by = neighbor -> (neighbor.x, neighbor.y))
    return neighbors
end

function transmit_disease!(model)
    probability = model.params.disease_transmission_probability
    probability == 0.0 && return nothing
    infectious = Dict{Position, UInt64}()
    for id in allids(model)
        agent = model[id]
        isnothing(agent.infection) && continue
        infectious[to_position(agent.pos)] = agent.infection.strain
    end
    isempty(infectious) && return nothing

    new_infections = Pair{Int64, Infection}[]
    rng = abmrng(model)
    for id in sort!(collect(allids(model)))
        agent = model[id]
        isnothing(agent.infection) || continue
        for neighbor in neighboring_positions(to_position(agent.pos), model.params)
            strain = get(infectious, neighbor, nothing)
            isnothing(strain) && continue
            strain == agent.immune_bits && continue
            if rand(rng) < probability
                push!(new_infections, id => Infection(strain, 0))
                break
            end
        end
    end
    for (id, infection) in new_infections
        model[id].infection = infection
        model.events.infections += 1
    end
    return nothing
end

function progress_infections!(model)
    for id in allids(model)
        agent = model[id]
        isnothing(agent.infection) && continue
        infection_age = agent.infection.age + 1
        agent.sugar -= model.params.disease_sugar_cost
        if infection_age >= model.params.disease_duration
            agent.immune_bits = agent.infection.strain
            agent.infection = nothing
            model.events.recoveries += 1
        else
            agent.infection = Infection(agent.infection.strain, infection_age)
        end
    end
    return nothing
end

function disease!(model)
    transmit_disease!(model)
    progress_infections!(model)
    return nothing
end

function lifecycle!(model)
    dead = Tuple{Int64, Bool, Bool}[]
    for id in sort!(collect(allids(model)))
        agent = model[id]
        agent.sugar -= agent.metabolism
        agent.age += 1
        starving = agent.sugar <= 0
        too_old = agent.age > agent.maximum_age
        (starving || too_old) && push!(dead, (id, starving, too_old && !starving))
    end
    for (id, starving, old_age) in dead
        remove_agent!(id, model)
        model.events.deaths += 1
        model.events.starvation_deaths += starving
        model.events.old_age_deaths += old_age
    end
    replace_dead!(model, length(dead))
    return nothing
end

function replace_dead!(model, amount::Integer)
    (amount == 0 || !model.params.replace_dead) && return nothing
    empty_positions = Position[
        Position(x, y)
        for x in 1:model.params.width for y in 1:model.params.height
        if isempty(ids_in_position((x, y), model))
    ]
    amount <= length(empty_positions) ||
        throw(ArgumentError("not enough empty cells to replace dead citizens"))
    rng = abmrng(model)
    for _ in 1:amount
        position_index = rand(rng, eachindex(empty_positions))
        spawn_citizen!(model, empty_positions[position_index])
        deleteat!(empty_positions, position_index)
        model.events.replacements += 1
    end
    return nothing
end

function reproductive_records(model)
    records = ReproductiveRecord[]
    params = model.params
    for id in sort!(collect(allids(model)))
        agent = model[id]
        params.minimum_fertility_age <= agent.age <= params.maximum_fertility_age ||
            continue
        contribution = cld(agent.initial_endowment, 2)
        agent.sugar > contribution || continue
        push!(
            records,
            ReproductiveRecord(
                id,
                to_position(agent.pos),
                agent.sex == :female,
                agent.vision,
                agent.metabolism,
                agent.sugar,
                agent.maximum_age,
                agent.initial_endowment,
            ),
        )
    end
    return records
end

choose_inherited(rng, mother_value, father_value) =
    rand(rng, Bool) ? mother_value : father_value

function plan_births(model)
    params = model.params
    params.reproduction_enabled || return BirthSpec[]
    rng = abmrng(model)
    records = reproductive_records(model)
    by_position = Dict(record.position => record for record in records)
    used = BitSet()
    reserved = Set{Position}()
    births = BirthSpec[]

    for mother in records
        mother.female || continue
        mother.id in used && continue
        rand(rng) < params.reproduction_probability || continue
        possible_fathers = ReproductiveRecord[]
        for position in neighboring_positions(mother.position, params)
            father = get(by_position, position, nothing)
            isnothing(father) && continue
            father.female && continue
            father.id in used && continue
            push!(possible_fathers, father)
        end
        isempty(possible_fathers) && continue
        sort!(possible_fathers; by = father -> father.id)
        father = possible_fathers[rand(rng, eachindex(possible_fathers))]
        birth_positions = [
            position
            for position in neighboring_positions(mother.position, params)
            if isempty(ids_in_position(to_tuple(position), model)) && position ∉ reserved
        ]
        isempty(birth_positions) && continue
        birth_position = birth_positions[rand(rng, eachindex(birth_positions))]
        mother_contribution = cld(mother.initial_endowment, 2)
        father_contribution = cld(father.initial_endowment, 2)
        child_endowment = mother_contribution + father_contribution
        push!(
            births,
            BirthSpec(
                birth_position,
                mother.id,
                father.id,
                mother_contribution,
                father_contribution,
                choose_inherited(rng, mother.vision, father.vision),
                choose_inherited(rng, mother.metabolism, father.metabolism),
                choose_inherited(rng, mother.maximum_age, father.maximum_age),
                child_endowment,
                rand(rng, Bool) ? :female : :male,
            ),
        )
        push!(used, mother.id)
        push!(used, father.id)
        push!(reserved, birth_position)
    end
    return births
end

function commit_births!(model, births)
    isempty(births) && return nothing
    for birth in births
        model[birth.mother_id].sugar -= birth.mother_contribution
        model[birth.father_id].sugar -= birth.father_contribution
    end
    for birth in births
        spawn_citizen!(
            model,
            birth.position;
            vision = birth.vision,
            metabolism = birth.metabolism,
            sugar = birth.initial_endowment,
            maximum_age = birth.maximum_age,
            initial_endowment = birth.initial_endowment,
            sex = birth.sex,
        )
        model.events.births += 1
    end
    return nothing
end

function reproduce!(model)
    commit_births!(model, plan_births(model))
    return nothing
end

function citizen_snapshot(model)
    citizens = CitizenState[]
    for id in sort!(collect(allids(model)))
        agent = model[id]
        push!(
            citizens,
            CitizenState(
                id,
                to_position(agent.pos),
                agent.vision,
                agent.metabolism,
                agent.sugar,
                agent.age,
                agent.maximum_age,
                agent.sex,
                !isnothing(agent.infection),
            ),
        )
    end
    return citizens
end

function record_logger!(model)
    citizens = citizen_snapshot(model)
    wealth = [citizen.sugar for citizen in citizens]
    ages = [citizen.age for citizen in citizens]
    log = model.logger
    events = model.events

    push!(log.step, model.clock.step)
    push!(log.population, length(citizens))
    push!(log.mean_wealth, isempty(wealth) ? NaN : sum(wealth) / length(wealth))
    push!(log.median_wealth, median_value(wealth))
    push!(log.gini, gini_coefficient(wealth))
    push!(log.mean_age, isempty(ages) ? NaN : sum(ages) / length(ages))
    push!(log.total_agent_sugar, sum(wealth))
    push!(log.total_landscape_sugar, sum(model.landscape.current))
    push!(log.moved, events.moved)
    push!(log.conflicts, events.conflicts)
    push!(log.harvested, events.harvested)
    push!(log.deaths, events.deaths)
    push!(log.starvation_deaths, events.starvation_deaths)
    push!(log.old_age_deaths, events.old_age_deaths)
    push!(log.replacements, events.replacements)
    push!(log.births, events.births)
    push!(log.infected, count(citizen -> citizen.infected, citizens))
    push!(log.infections, events.infections)
    push!(log.recoveries, events.recoveries)
    return nothing
end

function advance!(model)
    reset!(model.events)
    growback!(model)
    move_and_harvest!(model)
    disease!(model)
    lifecycle!(model)
    reproduce!(model)
    model.clock.step += 1
    record_logger!(model)
    return nothing
end

function setup_model(args::ModelArgs = ModelArgs())
    validate(args)
    args.params.movement_mode == ShuffledSequentialMovement || throw(
        ArgumentError("AgentSequential requires ShuffledSequentialMovement"),
    )
    capacity = isnothing(args.initial_capacity) ?
               canonical_landscape(args.params) : copy(args.initial_capacity)
    properties = (
        params = args.params,
        landscape = SugarLandscape(copy(capacity), capacity),
        events = StepEvents(),
        logger = Logger(),
        clock = SimulationClock(0),
    )
    model = StandardABM(
        Citizen,
        GridSpace((args.params.width, args.params.height); periodic = true);
        model_step! = advance!,
        properties = properties,
        rng = Random.Xoshiro(args.seed),
    )
    spawn_initial_population!(model)
    seed_initial_infections!(model)
    return model
end

"""Advance the shuffled-sequential Agents.jl Sugarscape by one period."""
function step!(model)
    Agents.step!(model)
    return nothing
end

function run_model(args::ModelArgs = ModelArgs())
    model = setup_model(args)
    for _ in 1:args.steps
        step!(model)
    end
    return model
end

end # module AgentSequential

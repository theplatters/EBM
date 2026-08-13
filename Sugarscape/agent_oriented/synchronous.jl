module AgentSynchronous

import Agents
using Agents: @agent,
    ABM,
    GridAgent,
    GridSpace,
    StandardABM,
    abmrng,
    add_agent!,
    allids,
    move_agent!,
    remove_agent!
using Random

using ..Sugarscape: CitizenState,
    Infection,
    Logger,
    ModelArgs,
    ModelParams,
    Position,
    SynchronousMovement,
    StepEvents,
    SugarLandscape,
    canonical_landscape,
    gini_coefficient,
    median_value,
    reset!,
    validate

export BirthSpec,
    Citizen,
    citizen_snapshot,
    landscape,
    logger,
    plan_movements!,
    resolve_and_commit_movements!,
    run_model,
    setup_model,
    step!

@agent struct Citizen(GridAgent{2})
    proposed_pos::NTuple{2, Int}
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

struct MovementRecord
    id::Int64
    position::Position
    vision::Int64
    sugar::Int64
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

sorted_ids(model) = sort!(collect(allids(model)))

function rebuild_occupancy!(model)
    occupancy = model.occupancy
    fill!(occupancy, 0)
    for id in sorted_ids(model)
        agent = model[id]
        occupancy[agent.pos...] == 0 ||
            throw(ArgumentError("multiple citizens occupy $(Position(agent.pos...))"))
        occupancy[agent.pos...] = id
    end
    return nothing
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
        (position.x, position.y),
        model;
        proposed_pos = (position.x, position.y),
        vision = Int64(
            isnothing(vision) ?
            rand(rng, params.minimum_vision:params.maximum_vision) : vision,
        ),
        metabolism = Int64(
            isnothing(metabolism) ?
            rand(rng, params.minimum_metabolism:params.maximum_metabolism) : metabolism,
        ),
        sugar = Int64(citizen_sugar),
        age = Int64(0),
        maximum_age = Int64(
            isnothing(maximum_age) ?
            rand(rng, params.minimum_lifespan:params.maximum_lifespan) : maximum_age,
        ),
        initial_endowment = Int64(endowment),
        sex = citizen_sex,
        immune_bits = isnothing(immune_bits) ? rand(rng, UInt64) : immune_bits,
        infection = nothing,
    )
    return agent.id
end

function initial_positions(params::ModelParams, rng)
    cell_indices = randperm(rng, params.width * params.height)[1:params.population]
    return [
        Position(mod1(index, params.width), (index - 1) ÷ params.width + 1)
        for index in cell_indices
    ]
end

function spawn_initial_population!(model)
    for (index, position) in enumerate(initial_positions(model.params, abmrng(model)))
        spawn_citizen!(model, position; sex = isodd(index) ? :female : :male)
    end
    return nothing
end

function seed_initial_infections!(model)
    probability = model.params.initial_infection_probability
    probability == 0.0 && return nothing
    rng = abmrng(model)
    for id in sorted_ids(model)
        agent = model[id]
        rand(rng) < probability || continue
        strain = rand(rng, UInt64)
        strain == agent.immune_bits && (strain = ~strain)
        agent.infection = Infection(strain, 0)
    end
    return nothing
end

function setup_model(args::ModelArgs = ModelArgs())
    validate(args)
    params = args.params
    params.movement_mode == SynchronousMovement || throw(
        ArgumentError("AgentSynchronous requires movement_mode = SynchronousMovement"),
    )
    capacity = isnothing(args.initial_capacity) ?
               canonical_landscape(params) : copy(args.initial_capacity)
    properties = (
        params = params,
        landscape = SugarLandscape(copy(capacity), capacity),
        occupancy = zeros(Int64, params.width, params.height),
        events = StepEvents(),
        logger = Logger(),
        clock = Ref{Int64}(0),
    )
    model = StandardABM(
        Citizen,
        GridSpace((params.width, params.height); periodic = true, metric = :manhattan);
        model_step! = synchronous_step!,
        properties = properties,
        rng = Random.Xoshiro(args.seed),
    )
    spawn_initial_population!(model)
    seed_initial_infections!(model)
    rebuild_occupancy!(model)
    return model
end

function growback!(model)
    rate = model.params.growback_rate
    iszero(rate) && return nothing
    sugar = landscape(model)
    @inbounds for index in eachindex(sugar.current)
        sugar.current[index] = min(sugar.capacity[index], sugar.current[index] + rate)
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
            distances[candidate] = min(get(distances, candidate, typemax(Int64)), distance)
        end
    end
    return distances
end

function select_destination(
    position::Position,
    vision::Integer,
    citizen_id::Integer,
    sugar::SugarLandscape,
    occupancy::Matrix{Int64},
    params::ModelParams,
    rng,
)
    best_sugar = typemin(Int64)
    best_distance = typemax(Int64)
    best = Position[]
    cells = collect(visible_cells(position, vision, params))
    sort!(cells; by = pair -> (last(pair), first(pair).x, first(pair).y))
    for (candidate, distance) in cells
        occupant = occupancy[candidate.x, candidate.y]
        (occupant == 0 || occupant == citizen_id) || continue
        patch_sugar = sugar.current[candidate.x, candidate.y]
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
    return best[rand(rng, eachindex(best))]
end

function movement_records(model)
    return [
        MovementRecord(
            id,
            Position(model[id].pos...),
            model[id].vision,
            model[id].sugar,
        )
        for id in sorted_ids(model)
    ]
end

function plan_movements!(model)
    params = model.params
    sugar = landscape(model)
    occupancy = model.occupancy
    rng = abmrng(model)
    for record in movement_records(model)
        destination = select_destination(
            record.position,
            record.vision,
            record.id,
            sugar,
            occupancy,
            params,
            rng,
        )
        model[record.id].proposed_pos = (destination.x, destination.y)
    end
    return nothing
end

function resolve_and_commit_movements!(model)
    sugar = landscape(model)
    events = model.events
    rng = abmrng(model)
    ids = sorted_ids(model)
    origins = Dict(id => Position(model[id].pos...) for id in ids)
    proposals = Dict{Position, Vector{Int64}}()
    sugar_by_id = Dict(id => model[id].sugar for id in ids)
    for id in ids
        push!(get!(proposals, Position(model[id].proposed_pos...), Int64[]), id)
    end

    destinations = copy(origins)
    proposal_destinations = sort!(
        collect(keys(proposals));
        by = position -> (position.x, position.y),
    )
    for destination in proposal_destinations
        contenders = proposals[destination]
        sort!(contenders)
        if length(contenders) == 1
            destinations[only(contenders)] = destination
        else
            winner = contenders[rand(rng, eachindex(contenders))]
            destinations[winner] = destination
            events.conflicts += length(contenders) - 1
        end
    end

    wealth = Dict{Int64, Int64}()
    for (id, destination) in destinations
        harvest = sugar.current[destination.x, destination.y]
        sugar.current[destination.x, destination.y] = 0
        wealth[id] = sugar_by_id[id] + harvest
        events.moved += destination != origins[id]
        events.harvested += harvest
    end
    for id in ids
        agent = model[id]
        destination = destinations[id]
        move_agent!(agent, (destination.x, destination.y), model)
        agent.proposed_pos = agent.pos
        agent.sugar = wealth[id]
    end
    rebuild_occupancy!(model)
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
    for id in sorted_ids(model)
        agent = model[id]
        isnothing(agent.infection) && continue
        infectious[Position(agent.pos...)] = agent.infection.strain
    end
    isempty(infectious) && return nothing

    rng = abmrng(model)
    new_infections = Pair{Int64, Infection}[]
    for id in sorted_ids(model)
        agent = model[id]
        isnothing(agent.infection) || continue
        for neighbor in neighboring_positions(Position(agent.pos...), model.params)
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
    params = model.params
    for id in sorted_ids(model)
        agent = model[id]
        isnothing(agent.infection) && continue
        infection = agent.infection
        infection_age = infection.age + 1
        agent.sugar -= params.disease_sugar_cost
        if infection_age >= params.disease_duration
            agent.immune_bits = infection.strain
            agent.infection = nothing
            model.events.recoveries += 1
        else
            agent.infection = Infection(infection.strain, infection_age)
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
    deaths = Tuple{Int64, Bool, Bool}[]
    for id in sorted_ids(model)
        agent = model[id]
        agent.sugar -= agent.metabolism
        agent.age += 1
        starving = agent.sugar <= 0
        too_old = agent.age > agent.maximum_age
        (starving || too_old) && push!(deaths, (id, starving, too_old && !starving))
    end

    for (id, starvation, old_age) in deaths
        remove_agent!(id, model)
        model.events.deaths += 1
        model.events.starvation_deaths += starvation
        model.events.old_age_deaths += old_age
    end
    rebuild_occupancy!(model)
    replace_dead!(model, length(deaths))
    return nothing
end

function replace_dead!(model, amount::Integer)
    (amount == 0 || !model.params.replace_dead) && return nothing
    params = model.params
    empty_positions = Position[
        Position(x, y)
        for x in 1:params.width for y in 1:params.height
        if model.occupancy[x, y] == 0
    ]
    amount <= length(empty_positions) ||
        throw(ArgumentError("not enough empty cells to replace dead citizens"))
    rng = abmrng(model)
    for _ in 1:amount
        position_index = rand(rng, eachindex(empty_positions))
        position = empty_positions[position_index]
        citizen_id = spawn_citizen!(model, position)
        model.occupancy[position.x, position.y] = citizen_id
        deleteat!(empty_positions, position_index)
        model.events.replacements += 1
    end
    return nothing
end

function reproductive_records(model)
    params = model.params
    records = ReproductiveRecord[]
    for id in sorted_ids(model)
        agent = model[id]
        params.minimum_fertility_age <= agent.age <= params.maximum_fertility_age ||
            continue
        contribution = cld(agent.initial_endowment, 2)
        agent.sugar > contribution || continue
        push!(
            records,
            ReproductiveRecord(
                id,
                Position(agent.pos...),
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
            if model.occupancy[position.x, position.y] == 0 && position ∉ reserved
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
    deductions = Dict{Int64, Int64}()
    for birth in births
        deductions[birth.mother_id] = birth.mother_contribution
        deductions[birth.father_id] = birth.father_contribution
    end
    for id in sorted_ids(model)
        model[id].sugar -= get(deductions, id, 0)
    end
    for birth in births
        citizen_id = spawn_citizen!(
            model,
            birth.position;
            vision = birth.vision,
            metabolism = birth.metabolism,
            sugar = birth.initial_endowment,
            maximum_age = birth.maximum_age,
            initial_endowment = birth.initial_endowment,
            sex = birth.sex,
        )
        model.occupancy[birth.position.x, birth.position.y] = citizen_id
        model.events.births += 1
    end
    return nothing
end

function reproduce!(model)
    commit_births!(model, plan_births(model))
    return nothing
end

function citizen_snapshot(model)
    return [
        CitizenState(
            id,
            Position(model[id].pos...),
            model[id].vision,
            model[id].metabolism,
            model[id].sugar,
            model[id].age,
            model[id].maximum_age,
            model[id].sex,
            !isnothing(model[id].infection),
        )
        for id in sorted_ids(model)
    ]
end

function log_step!(model)
    log = logger(model)
    events = model.events
    citizens = citizen_snapshot(model)
    wealth = [citizen.sugar for citizen in citizens]
    ages = [citizen.age for citizen in citizens]

    push!(log.step, model.clock[])
    push!(log.population, length(citizens))
    push!(log.mean_wealth, isempty(wealth) ? NaN : sum(wealth) / length(wealth))
    push!(log.median_wealth, median_value(wealth))
    push!(log.gini, gini_coefficient(wealth))
    push!(log.mean_age, isempty(ages) ? NaN : sum(ages) / length(ages))
    push!(log.total_agent_sugar, sum(wealth))
    push!(log.total_landscape_sugar, sum(landscape(model).current))
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

function synchronous_step!(model)
    reset!(model.events)
    growback!(model)
    rebuild_occupancy!(model)
    plan_movements!(model)
    resolve_and_commit_movements!(model)
    disease!(model)
    lifecycle!(model)
    reproduce!(model)
    model.clock[] += 1
    log_step!(model)
    return nothing
end

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

end # module AgentSynchronous

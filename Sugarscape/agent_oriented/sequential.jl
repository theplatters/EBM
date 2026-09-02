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
    Disease,
    DiseaseCatalog,
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
    bit_mask,
    generate_disease_catalog,
    gini_coefficient,
    immune_to,
    inherit_immune_genotype,
    initial_disease_mask,
    median_value,
    random_disease_bit,
    reset!,
    train_immunity,
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
    immune_genotype::UInt64
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
    immune_genotype::UInt64
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
    immune_genotype::UInt64
    sex::Symbol
end

mutable struct SequentialBuffers
    ids::Vector{Int64}
    visibility_marks::Matrix{Int64}
    visibility_epoch::Int64
    visible_candidates::Vector{Position}
    best_positions::Vector{Position}
    neighbors::Vector{Position}
    infectious::Dict{Position, UInt64}
    new_infections::Vector{Pair{Int64, UInt64}}
    dead::Vector{Tuple{Int64, Bool, Bool}}
    empty_positions::Vector{Position}
    reproductive_records::Vector{ReproductiveRecord}
    records_by_position::Dict{Position, ReproductiveRecord}
    used_parents::BitSet
    reserved_positions::Set{Position}
    births::Vector{BirthSpec}
    possible_fathers::Vector{ReproductiveRecord}
    birth_positions::Vector{Position}
    wealth::Vector{Int64}
    sorted_wealth::Vector{Float64}
end

SequentialBuffers(params::ModelParams) = SequentialBuffers(
    Int64[],
    zeros(Int64, params.width, params.height),
    0,
    Position[],
    Position[],
    Position[],
    Dict{Position, UInt64}(),
    Pair{Int64, UInt64}[],
    Tuple{Int64, Bool, Bool}[],
    Position[],
    ReproductiveRecord[],
    Dict{Position, ReproductiveRecord}(),
    BitSet(),
    Set{Position}(),
    BirthSpec[],
    ReproductiveRecord[],
    Position[],
    Int64[],
    Float64[],
)

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
    immune_genotype = nothing,
    immune_bits = nothing,
)
    params = model.params
    rng = abmrng(model)
    citizen_sugar = isnothing(sugar) ?
                    rand(rng, params.minimum_initial_sugar:params.maximum_initial_sugar) : sugar
    endowment = isnothing(initial_endowment) ? citizen_sugar : initial_endowment
    citizen_sex = isnothing(sex) ? (rand(rng, Bool) ? :female : :male) : sex
    citizen_vision =
        isnothing(vision) ? rand(rng, params.minimum_vision:params.maximum_vision) : vision
    citizen_metabolism = isnothing(metabolism) ?
                         rand(rng, params.minimum_metabolism:params.maximum_metabolism) :
                         metabolism
    citizen_maximum_age = isnothing(maximum_age) ?
                          rand(rng, params.minimum_lifespan:params.maximum_lifespan) :
                          maximum_age
    genotype = isnothing(immune_genotype) ? rand(rng, UInt64) : UInt64(immune_genotype)
    genotype &= bit_mask(params.immune_system_length)
    phenotype = isnothing(immune_bits) ? genotype :
                UInt64(immune_bits) & bit_mask(params.immune_system_length)
    agent = add_agent!(
        to_tuple(position),
        model;
        vision = citizen_vision,
        metabolism = citizen_metabolism,
        sugar = citizen_sugar,
        age = 0,
        maximum_age = citizen_maximum_age,
        initial_endowment = endowment,
        sex = citizen_sex,
        immune_genotype = genotype,
        immune_bits = phenotype,
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
    params = model.params
    count = params.initial_diseases_per_citizen
    iszero(count) && return nothing
    rng = abmrng(model)
    for id in sort!(collect(allids(model)))
        agent = model[id]
        diseases = initial_disease_mask(
            rng,
            model.disease_catalog,
            count,
            agent.immune_bits,
            params.immune_system_length,
        )
        iszero(diseases) || (agent.infection = Infection(diseases))
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
    buffers = model.buffers
    best = buffers.best_positions
    empty!(best)
    if buffers.visibility_epoch == typemax(Int64)
        fill!(buffers.visibility_marks, 0)
        buffers.visibility_epoch = 1
    else
        buffers.visibility_epoch += 1
    end
    epoch = buffers.visibility_epoch
    marks = buffers.visibility_marks
    candidates = buffers.visible_candidates
    params = model.params

    for distance in 0:agent.vision
        empty!(candidates)
        if iszero(distance)
            push!(candidates, position)
        else
            push!(candidates, Position(mod1(position.x + distance, params.width), position.y))
            push!(candidates, Position(mod1(position.x - distance, params.width), position.y))
            push!(candidates, Position(position.x, mod1(position.y + distance, params.height)))
            push!(candidates, Position(position.x, mod1(position.y - distance, params.height)))
            sort!(candidates; by = candidate -> (candidate.x, candidate.y))
        end
        for candidate in candidates
            marks[candidate.x, candidate.y] == epoch && continue
            marks[candidate.x, candidate.y] = epoch
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
    ids = model.buffers.ids
    empty!(ids)
    append!(ids, allids(model))
    sort!(ids)
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

function neighboring_positions!(buffers::SequentialBuffers, position::Position, params::ModelParams)
    neighbors = buffers.neighbors
    empty!(neighbors)
    push!(neighbors, Position(mod1(position.x + 1, params.width), position.y))
    push!(neighbors, Position(mod1(position.x - 1, params.width), position.y))
    push!(neighbors, Position(position.x, mod1(position.y + 1, params.height)))
    push!(neighbors, Position(position.x, mod1(position.y - 1, params.height)))
    sort!(neighbors; by = neighbor -> (neighbor.x, neighbor.y))
    write_index = 0
    for neighbor in neighbors
        neighbor == position && continue
        write_index > 0 && neighbor == neighbors[write_index] && continue
        write_index += 1
        neighbors[write_index] = neighbor
    end
    resize!(neighbors, write_index)
    return neighbors
end

function transmit_disease!(model)
    isempty(model.disease_catalog.diseases) && return nothing
    buffers = model.buffers
    infectious = buffers.infectious
    empty!(infectious)
    for id in allids(model)
        agent = model[id]
        isnothing(agent.infection) && continue
        infectious[to_position(agent.pos)] = agent.infection.diseases
    end
    isempty(infectious) && return nothing

    new_infections = buffers.new_infections
    empty!(new_infections)
    rng = abmrng(model)
    ids = buffers.ids
    empty!(ids)
    append!(ids, allids(model))
    sort!(ids)
    for id in ids
        agent = model[id]
        current = isnothing(agent.infection) ? UInt64(0) : agent.infection.diseases
        additions = UInt64(0)
        for neighbor in neighboring_positions!(buffers, to_position(agent.pos), model.params)
            donor_diseases = get(infectious, neighbor, UInt64(0))
            iszero(donor_diseases) && continue
            disease_bit = random_disease_bit(rng, donor_diseases)
            iszero((current | additions) & disease_bit) || continue
            disease_index = trailing_zeros(disease_bit) + 1
            immune_to(
                agent.immune_bits,
                model.disease_catalog.diseases[disease_index],
                model.params.immune_system_length,
            ) && continue
            additions |= disease_bit
        end
        iszero(additions) || push!(new_infections, id => additions)
    end
    for (id, additions) in new_infections
        current = isnothing(model[id].infection) ? UInt64(0) : model[id].infection.diseases
        model[id].infection = Infection(current | additions)
        model.events.infections += count_ones(additions)
    end
    return nothing
end

function progress_infections!(model)
    params = model.params
    for id in sort!(collect(allids(model)))
        agent = model[id]
        isnothing(agent.infection) && continue
        diseases = agent.infection.diseases
        starting_immunity = agent.immune_bits
        disease_cost = 0
        for index in 1:length(model.disease_catalog.diseases)
            disease_bit = UInt64(1) << (index - 1)
            iszero(diseases & disease_bit) && continue
            immune_to(
                starting_immunity,
                model.disease_catalog.diseases[index],
                params.immune_system_length,
            ) || (disease_cost += params.disease_sugar_cost)
        end
        agent.sugar -= disease_cost

        remaining = diseases
        for index in 1:length(model.disease_catalog.diseases)
            disease_bit = UInt64(1) << (index - 1)
            iszero(diseases & disease_bit) && continue
            disease = model.disease_catalog.diseases[index]
            immune_to(agent.immune_bits, disease, params.immune_system_length) ||
                (agent.immune_bits =
                    train_immunity(agent.immune_bits, disease, params.immune_system_length))
            if immune_to(agent.immune_bits, disease, params.immune_system_length)
                remaining &= ~disease_bit
                model.events.recoveries += 1
            end
        end
        agent.infection = iszero(remaining) ? nothing : Infection(remaining)
    end
    return nothing
end

function disease!(model)
    transmit_disease!(model)
    progress_infections!(model)
    return nothing
end

function lifecycle!(model)
    buffers = model.buffers
    dead = buffers.dead
    empty!(dead)
    ids = buffers.ids
    empty!(ids)
    append!(ids, allids(model))
    sort!(ids)
    for id in ids
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
    empty_positions = model.buffers.empty_positions
    empty!(empty_positions)
    for x in 1:model.params.width, y in 1:model.params.height
        isempty(ids_in_position((x, y), model)) && push!(empty_positions, Position(x, y))
    end
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
    records = model.buffers.reproductive_records
    empty!(records)
    params = model.params
    ids = model.buffers.ids
    empty!(ids)
    append!(ids, allids(model))
    sort!(ids)
    for id in ids
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
                agent.immune_genotype,
            ),
        )
    end
    return records
end

choose_inherited(rng, mother_value, father_value) =
    rand(rng, Bool) ? mother_value : father_value

function plan_births(model)
    params = model.params
    buffers = model.buffers
    births = buffers.births
    empty!(births)
    params.reproduction_enabled || return births
    rng = abmrng(model)
    records = reproductive_records(model)
    by_position = buffers.records_by_position
    empty!(by_position)
    for record in records
        by_position[record.position] = record
    end
    used = buffers.used_parents
    empty!(used)
    reserved = buffers.reserved_positions
    empty!(reserved)

    for mother in records
        mother.female || continue
        mother.id in used && continue
        rand(rng) < params.reproduction_probability || continue
        possible_fathers = buffers.possible_fathers
        empty!(possible_fathers)
        for position in neighboring_positions!(buffers, mother.position, params)
            father = get(by_position, position, nothing)
            isnothing(father) && continue
            father.female && continue
            father.id in used && continue
            push!(possible_fathers, father)
        end
        isempty(possible_fathers) && continue
        sort!(possible_fathers; by = father -> father.id)
        father = possible_fathers[rand(rng, eachindex(possible_fathers))]
        birth_positions = buffers.birth_positions
        empty!(birth_positions)
        for position in neighboring_positions!(buffers, mother.position, params)
            isempty(ids_in_position(to_tuple(position), model)) && position ∉ reserved &&
                push!(birth_positions, position)
        end
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
                inherit_immune_genotype(
                    rng,
                    mother.immune_genotype,
                    father.immune_genotype,
                    params.immune_system_length,
                ),
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
            immune_genotype = birth.immune_genotype,
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
    wealth = model.buffers.wealth
    sorted_wealth = model.buffers.sorted_wealth
    empty!(wealth)
    empty!(sorted_wealth)
    total_age = 0
    infected = 0
    for id in allids(model)
        agent = model[id]
        push!(wealth, agent.sugar)
        push!(sorted_wealth, agent.sugar)
        total_age += agent.age
        infected += !isnothing(agent.infection)
    end
    sort!(wealth)
    sort!(sorted_wealth)
    population = length(wealth)
    total_wealth = sum(wealth)
    median_wealth = if isempty(wealth)
        NaN
    else
        midpoint = length(wealth) ÷ 2
        isodd(length(wealth)) ? Float64(wealth[midpoint + 1]) :
        (wealth[midpoint] + wealth[midpoint + 1]) / 2.0
    end
    total_float_wealth = sum(sorted_wealth)
    gini = if isempty(sorted_wealth)
        NaN
    elseif total_float_wealth == 0.0
        0.0
    else
        n = length(sorted_wealth)
        weighted_sum = sum(index * value for (index, value) in enumerate(sorted_wealth))
        (2.0 * weighted_sum) / (n * total_float_wealth) - (n + 1.0) / n
    end
    log = model.logger
    events = model.events

    push!(log.step, model.clock.step)
    push!(log.population, population)
    push!(log.mean_wealth, isempty(wealth) ? NaN : total_wealth / population)
    push!(log.median_wealth, median_wealth)
    push!(log.gini, gini)
    push!(log.mean_age, population == 0 ? NaN : total_age / population)
    push!(log.total_agent_sugar, total_wealth)
    push!(log.total_landscape_sugar, sum(model.landscape.current))
    push!(log.moved, events.moved)
    push!(log.conflicts, events.conflicts)
    push!(log.harvested, events.harvested)
    push!(log.deaths, events.deaths)
    push!(log.starvation_deaths, events.starvation_deaths)
    push!(log.old_age_deaths, events.old_age_deaths)
    push!(log.replacements, events.replacements)
    push!(log.births, events.births)
    push!(log.infected, infected)
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
               canonical_landscape(args.params) :
               copy(args.initial_capacity)
    properties = (
        params = args.params,
        landscape = SugarLandscape(copy(capacity), capacity),
        events = StepEvents(),
        logger = Logger(),
        clock = SimulationClock(0),
        buffers = SequentialBuffers(args.params),
        disease_catalog = DiseaseCatalog(Disease[]),
    )
    model = StandardABM(
        Citizen,
        GridSpace((args.params.width, args.params.height); periodic = true);
        model_step! = advance!,
        properties = properties,
        rng = Random.Xoshiro(args.seed),
    )
    append!(
        model.disease_catalog.diseases,
        generate_disease_catalog(abmrng(model), args.params),
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

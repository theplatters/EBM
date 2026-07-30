struct ReproductiveRecord
    entity::Ark.Entity
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
    sex::Union{Female, Male}
end

function reproductive_records(world)
    params = Ark.get_resource(world, ModelParams)
    records = ReproductiveRecord[]
    for (
        entities,
        ids,
        positions,
        visions,
        metabolisms,
        sugars,
        ages,
        maximum_ages,
        endowments,
    ) in Query(
        world,
        (
            CitizenId,
            Position,
            Vision,
            Metabolism,
            Sugar,
            Age,
            MaximumAge,
            InitialEndowment,
        ),
    )
        @inbounds for i in eachindex(entities)
            params.minimum_fertility_age <= ages[i].val <= params.maximum_fertility_age ||
                continue
            contribution = cld(endowments[i].val, 2)
            sugars[i].val > contribution || continue
            push!(
                records,
                ReproductiveRecord(
                    entities[i],
                    ids[i].val,
                    positions[i],
                    Ark.has_components(world, entities[i], (Female,)),
                    visions[i].val,
                    metabolisms[i].val,
                    sugars[i].val,
                    maximum_ages[i].val,
                    endowments[i].val,
                ),
            )
        end
    end
    sort!(records; by = record -> record.id)
    return records
end

function choose_inherited(rng, mother_value, father_value)
    return rand(rng, Bool) ? mother_value : father_value
end

function plan_births(world)
    params = Ark.get_resource(world, ModelParams)
    params.reproduction_enabled || return BirthSpec[]
    occupancy = Ark.get_resource(world, OccupancyGrid)
    rng = simulation_rng(world)
    records = reproductive_records(world)
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
            if occupancy.citizen_ids[position.x, position.y] == 0 && position ∉ reserved
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
                rand(rng, Bool) ? Female() : Male(),
            ),
        )
        push!(used, mother.id)
        push!(used, father.id)
        push!(reserved, birth_position)
    end
    return births
end

function commit_births!(world, births)
    isempty(births) && return nothing
    deductions = Dict{Int64, Int64}()
    for birth in births
        deductions[birth.mother_id] = birth.mother_contribution
        deductions[birth.father_id] = birth.father_contribution
    end
    for (entities, ids, sugars) in Query(world, (CitizenId, Sugar))
        @inbounds for i in eachindex(entities)
            deduction = get(deductions, ids[i].val, 0)
            deduction == 0 && continue
            sugars[i] = Sugar(sugars[i].val - deduction)
        end
    end

    occupancy = Ark.get_resource(world, OccupancyGrid)
    events = Ark.get_resource(world, StepEvents)
    for birth in births
        citizen_id = spawn_citizen!(
            world,
            birth.position;
            vision = birth.vision,
            metabolism = birth.metabolism,
            sugar = birth.initial_endowment,
            maximum_age = birth.maximum_age,
            initial_endowment = birth.initial_endowment,
            sex = birth.sex,
        )
        occupancy.citizen_ids[birth.position.x, birth.position.y] = citizen_id
        events.births += 1
    end
    return nothing
end

function reproduce!(world)
    births = plan_births(world)
    commit_births!(world, births)
    return nothing
end

function metabolize_and_age!(world)
    deaths = Ark.get_resource(world, SimulationBuffers).deaths
    empty!(deaths)
    for (entities, ids, sugars, metabolisms, ages, maximum_ages) in
        Query(world, (CitizenId, Sugar, Metabolism, Age, MaximumAge))
        @inbounds for i in eachindex(entities)
            remaining_sugar = sugars[i].val - metabolisms[i].val
            new_age = ages[i].val + 1
            sugars[i] = Sugar(remaining_sugar)
            ages[i] = Age(new_age)
            starving = remaining_sugar <= 0
            too_old = new_age > maximum_ages[i].val
            if starving || too_old
                push!(
                    deaths,
                    DeathRecord(entities[i], ids[i].val, starving, too_old && !starving),
                )
            end
        end
    end
    sort!(deaths; by = death -> death.id)
    return deaths
end

function remove_dead!(world, deaths)
    events = Ark.get_resource(world, StepEvents)
    for death in deaths
        Ark.remove_entity!(world, death.entity)
        events.deaths += 1
        events.starvation_deaths += death.starvation
        events.old_age_deaths += death.old_age
    end
    rebuild_occupancy!(world)
    return nothing
end

function replace_dead!(world, amount::Integer)
    amount == 0 && return nothing
    params = Ark.get_resource(world, ModelParams)
    params.replace_dead || return nothing
    occupancy = Ark.get_resource(world, OccupancyGrid)
    rng = simulation_rng(world)
    empty_positions = Ark.get_resource(world, SimulationBuffers).empty_positions
    empty!(empty_positions)
    for x in 1:params.width, y in 1:params.height
        occupancy.citizen_ids[x, y] == 0 && push!(empty_positions, Position(x, y))
    end
    amount <= length(empty_positions) ||
        throw(ArgumentError("not enough empty cells to replace dead citizens"))
    events = Ark.get_resource(world, StepEvents)
    for _ in 1:amount
        position_index = rand(rng, eachindex(empty_positions))
        position = empty_positions[position_index]
        citizen_id = spawn_citizen!(world, position)
        occupancy.citizen_ids[position.x, position.y] = citizen_id
        deleteat!(empty_positions, position_index)
        events.replacements += 1
    end
    return nothing
end

function lifecycle!(world)
    deaths = metabolize_and_age!(world)
    remove_dead!(world, deaths)
    replace_dead!(world, length(deaths))
    return nothing
end

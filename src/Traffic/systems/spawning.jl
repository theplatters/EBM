function random_unique_positions(r::Ring, n::Int; rng = Random.default_rng())
    w = r.width
    h = r.height
    total = w * h
    n <= total || throw(ArgumentError("n=$n exceeds ring capacity=$total"))
    idxs = randperm(rng, total)[1:n]
    return Position.([(mod1(i, w), (i - 1) ÷ w + 1) for i in idxs])
end

@inline function spawn_car!(
        world, pos, dir, ssens, osense, avoid, habitgene, habitus, strategy,
    )
    return Ark.new_entity!(
        world, (
            pos,
            PrevPosition(pos),
            dir,
            ssens,
            osense,
            avoid,
            habitgene,
            habitus,
            strategy,
            LR(0.0),
            Step(1),
        )
    )
end

function initial_driver_strategies(
        strategy::OccupancyStrategy, amount::Integer, rng,
    )
    return fill(DriverStrategy(strategy), amount)
end

function initial_driver_strategies(
        mixture::HeterogeneousStrategy, amount::Integer, rng,
    )
    exact_counts = mixture.shares .* amount
    counts = floor.(Int, exact_counts)
    remaining = amount - sum(counts)
    remainder_order = sortperm(
        eachindex(exact_counts);
        by = index -> (-(exact_counts[index] - counts[index]), index),
    )
    for index in Iterators.take(remainder_order, remaining)
        counts[index] += 1
    end

    assignments = DriverStrategy[]
    for (strategy, count) in zip(mixture.strategies, counts)
        append!(assignments, fill(strategy, count))
    end
    shuffle!(rng, assignments)
    return assignments
end

function spawn_init_entities!(world, strategy::OccupancyStrategy)
    ring = Ark.get_resource(world, Ring)
    rng = simulation_rng(world)
    params = Ark.get_resource(world, ModelParams)

    amount = params.init_agents

    positions = random_unique_positions(ring, amount, rng = rng)
    directions = shuffle(
        rng,
        repeat([Clockwise, Counterclockwise], cld(amount, 2))[1:amount],
    )
    draws = rand(rng, Normal(1.0, params.δ), amount, 4)
    # Keep physical initial conditions paired across homogeneous and mixed runs.
    # Only the strategy assignment consumes additional random numbers.
    strategies = initial_driver_strategies(strategy, amount, rng)

    @inbounds for i in 1:amount
        spawn_car!(
            world,
            positions[i],
            directions[i],
            SSensitvity(draws[i, 1]),
            OSensitvity(draws[i, 2]),
            Avoidance(draws[i, 3]),
            Habitgene(draws[i, 4]),
            Habitus(0.0),
            strategies[i],
        )
    end
    return nothing
end


struct ReplacementSpec
    direction::Direction
    strategy::DriverStrategy
end

function spawn_new_entities!(world, replacements::AbstractVector{ReplacementSpec})
    amount = length(replacements)
    rng = simulation_rng(world)
    params = Ark.get_resource(world, ModelParams)
    ring = Ark.get_resource(world, Ring)
    occupied_positions = Set{Position}()
    for (e, pos) in Query(world, (Position,))
        @inbounds for i in eachindex(e)
            push!(occupied_positions, pos[i])
        end
    end
    unoccupied_positions = Position[
        Position(x, y)
            for y in 1:ring.height for x in 1:ring.width
                if Position(x, y) ∉ occupied_positions
    ]

    replacements = shuffle(rng, replacements)
    draws = rand(rng, Normal(1.0, params.δ), amount, 4)

    for i in 1:amount
        position_index = rand(rng, eachindex(unoccupied_positions))
        position = unoccupied_positions[position_index]

        spawn_car!(
            world,
            position,
            replacements[i].direction,
            SSensitvity(draws[i, 1]),
            OSensitvity(draws[i, 2]),
            Avoidance(draws[i, 3]),
            Habitgene(draws[i, 4]),
            Habitus(0.0),
            replacements[i].strategy,

        )

        deleteat!(unoccupied_positions, position_index)
    end

    return nothing
end

struct CapabilityReplacementSpec
    direction::Direction
end

function entry_capability_genome(model::CapabilityModel, draws, rng)
    return CapabilityGenome(
        rand(rng) < model.same_direction_share ? draws[1] : nothing,
        rand(rng) < model.opposite_direction_share ? draws[2] : nothing,
        rand(rng) < model.avoidance_share ? draws[3] : nothing,
        rand(rng) < model.habit_share ? draws[4] : nothing,
        rand(rng) < model.convention_share ?
            ConventionPerception(model.convention_learning_rate, model.convention_noise) :
            nothing,
    )
end

function capability_components(genome::CapabilityGenome)
    components = ()
    !isnothing(genome.same_direction) &&
        (components = (components..., SameDirectionResponse(genome.same_direction)))
    !isnothing(genome.opposite_direction) &&
        (components = (components..., OppositeDirectionResponse(genome.opposite_direction)))
    !isnothing(genome.avoidance) &&
        (components = (components..., NearFieldAvoidance(genome.avoidance)))
    if !isnothing(genome.habit)
        components = (components..., HabitFormation(genome.habit), Habitus(0.0))
    end
    if !isnothing(genome.convention)
        components = (
            components...,
            genome.convention,
            PerceivedConvention(0.0, 0.0),
        )
    end
    return components
end

function spawn_capability_car!(world, position, direction, speed, genome, model)
    lane = position.x
    base = (
        position,
        PrevPosition(position),
        direction,
        Speed(speed),
        SpeedAdjustment(model.max_speed),
        LocalObservation(),
        LaneScore(0.0),
        LaneProposal(lane),
        SpeedProposal(speed),
        MovementPath(position),
        LR(0.0),
        Step(1),
    )
    return Ark.new_entity!(world, (base..., capability_components(genome)...))
end

function spawn_init_entities!(world, model::CapabilityModel)
    ring = Ark.get_resource(world, Ring)
    rng = simulation_rng(world)
    params = Ark.get_resource(world, ModelParams)
    amount = params.init_agents

    positions = random_unique_positions(ring, amount; rng = rng)
    directions = shuffle(
        rng,
        repeat([Clockwise, Counterclockwise], cld(amount, 2))[1:amount],
    )
    speeds = rand(rng, 1:model.max_speed, amount)
    draws = rand(rng, Normal(1.0, params.δ), amount, 4)
    @inbounds for index in 1:amount
        spawn_capability_car!(
            world,
            positions[index],
            directions[index],
            speeds[index],
            entry_capability_genome(model, view(draws, index, :), rng),
            model,
        )
    end
    return nothing
end

function optional_component(world, entity, ::Type{T}) where {T}
    Ark.has_components(world, entity, (T,)) || return nothing
    return Ark.get_components(world, entity, (T,))[1]
end

function capability_genome(world, entity)
    same = optional_component(world, entity, SameDirectionResponse)
    opposite = optional_component(world, entity, OppositeDirectionResponse)
    avoidance = optional_component(world, entity, NearFieldAvoidance)
    habit = optional_component(world, entity, HabitFormation)
    convention = optional_component(world, entity, ConventionPerception)
    return CapabilityGenome(
        isnothing(same) ? nothing : same.sensitivity,
        isnothing(opposite) ? nothing : opposite.sensitivity,
        isnothing(avoidance) ? nothing : avoidance.sensitivity,
        isnothing(habit) ? nothing : habit.disposition,
        convention,
    )
end

@inline mutate_nonnegative(value, scale, rng) =
    max(0.0, value + scale * randn(rng))

function mutate_optional_trait(parent, entry_value, enabled, policy, rng)
    if rand(rng) < policy.capability_mutation_rate
        return isnothing(parent) ? (enabled ? entry_value : nothing) : nothing
    end
    isnothing(parent) && return nothing
    return mutate_nonnegative(parent, policy.trait_mutation_scale, rng)
end

function mutate_optional_convention(parent, model, policy, rng)
    if rand(rng) < policy.capability_mutation_rate
        return isnothing(parent) && model.convention_share > 0.0 ?
               ConventionPerception(model.convention_learning_rate, model.convention_noise) :
               nothing
    end
    isnothing(parent) && return nothing
    return ConventionPerception(
        clamp(
            parent.learning_rate + policy.trait_mutation_scale * randn(rng),
            0.0,
            1.0,
        ),
        mutate_nonnegative(parent.noise, policy.trait_mutation_scale, rng),
    )
end

function inherit_capability_genome(
        parent::CapabilityGenome, model::CapabilityModel, draws,
        policy::EvolutionaryReplacement, rng,
    )
    return CapabilityGenome(
        mutate_optional_trait(
            parent.same_direction, draws[1], model.same_direction_share > 0.0, policy, rng,
        ),
        mutate_optional_trait(
            parent.opposite_direction, draws[2], model.opposite_direction_share > 0.0,
            policy, rng,
        ),
        mutate_optional_trait(
            parent.avoidance, draws[3], model.avoidance_share > 0.0, policy, rng,
        ),
        mutate_optional_trait(parent.habit, draws[4], model.habit_share > 0.0, policy, rng),
        mutate_optional_convention(parent.convention, model, policy, rng),
    )
end

function replacement_genome(
        ::EntryDrawReplacement, parent_genomes, model, draws, rng,
    )
    return entry_capability_genome(model, draws, rng)
end

function replacement_genome(
        policy::EvolutionaryReplacement, parent_genomes, model, draws, rng,
    )
    isempty(parent_genomes) && return entry_capability_genome(model, draws, rng)
    parent = rand(rng, parent_genomes)
    return inherit_capability_genome(parent, model, draws, policy, rng)
end

function spawn_new_entities!(
        world, replacements::AbstractVector{CapabilityReplacementSpec},
    )
    amount = length(replacements)
    amount == 0 && return nothing

    rng = simulation_rng(world)
    params = Ark.get_resource(world, ModelParams)
    ring = Ark.get_resource(world, Ring)
    model = Ark.get_resource(world, CapabilityModel)
    occupied_positions = Set{Position}()
    parent_genomes = CapabilityGenome[]
    for (entities, positions) in Query(world, (Position,))
        @inbounds for index in eachindex(entities)
            push!(occupied_positions, positions[index])
            push!(parent_genomes, capability_genome(world, entities[index]))
        end
    end
    unoccupied_positions = Position[
        Position(x, y)
            for y in 1:ring.height for x in 1:ring.width
                if Position(x, y) ∉ occupied_positions
    ]

    shuffled_replacements = shuffle(rng, collect(replacements))
    draws = rand(rng, Normal(1.0, params.δ), amount, 4)
    for index in 1:amount
        position_index = rand(rng, eachindex(unoccupied_positions))
        position = unoccupied_positions[position_index]
        speed = rand(rng, 1:model.max_speed)
        genome = replacement_genome(
            model.replacement_policy,
            parent_genomes,
            model,
            view(draws, index, :),
            rng,
        )
        spawn_capability_car!(
            world,
            position,
            shuffled_replacements[index].direction,
            speed,
            genome,
            model,
        )
        deleteat!(unoccupied_positions, position_index)
    end
    return nothing
end

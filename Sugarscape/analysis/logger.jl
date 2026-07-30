mutable struct Logger
    step::Vector{Int64}
    population::Vector{Int64}
    mean_wealth::Vector{Float64}
    median_wealth::Vector{Float64}
    gini::Vector{Float64}
    mean_age::Vector{Float64}
    total_agent_sugar::Vector{Int64}
    total_landscape_sugar::Vector{Int64}
    moved::Vector{Int64}
    conflicts::Vector{Int64}
    harvested::Vector{Int64}
    deaths::Vector{Int64}
    starvation_deaths::Vector{Int64}
    old_age_deaths::Vector{Int64}
    replacements::Vector{Int64}
    births::Vector{Int64}
    infected::Vector{Int64}
    infections::Vector{Int64}
    recoveries::Vector{Int64}
end

Logger() = Logger(
    Int64[],
    Int64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
    Int64[],
)

function gini_coefficient(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    sorted = sort(Float64.(values))
    total = sum(sorted)
    total == 0.0 && return 0.0
    n = length(sorted)
    weighted_sum = sum(index * value for (index, value) in enumerate(sorted))
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n
end

function median_value(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    sorted = sort(values)
    midpoint = length(sorted) ÷ 2
    return isodd(length(sorted)) ? Float64(sorted[midpoint + 1]) :
           (sorted[midpoint] + sorted[midpoint + 1]) / 2.0
end

function citizen_snapshot(world)
    citizens = CitizenState[]
    for (
        entities,
        ids,
        positions,
        visions,
        metabolisms,
        sugars,
        ages,
        maximum_ages,
    ) in Query(
        world,
        (CitizenId, Position, Vision, Metabolism, Sugar, Age, MaximumAge),
    )
        @inbounds for i in eachindex(entities)
            push!(
                citizens,
                CitizenState(
                    ids[i].val,
                    positions[i],
                    visions[i].val,
                    metabolisms[i].val,
                    sugars[i].val,
                    ages[i].val,
                    maximum_ages[i].val,
                    Ark.has_components(world, entities[i], (Female,)) ? :female : :male,
                    Ark.has_components(world, entities[i], (Infection,)),
                ),
            )
        end
    end
    sort!(citizens; by = citizen -> citizen.id)
    return citizens
end

function logger!(world)
    logger = Ark.get_resource(world, Logger)
    clock = Ark.get_resource(world, SimulationClock)
    events = Ark.get_resource(world, StepEvents)
    landscape = Ark.get_resource(world, SugarLandscape)
    citizens = citizen_snapshot(world)
    wealth = [citizen.sugar for citizen in citizens]
    ages = [citizen.age for citizen in citizens]

    push!(logger.step, clock.step)
    push!(logger.population, length(citizens))
    push!(logger.mean_wealth, isempty(wealth) ? NaN : sum(wealth) / length(wealth))
    push!(logger.median_wealth, median_value(wealth))
    push!(logger.gini, gini_coefficient(wealth))
    push!(logger.mean_age, isempty(ages) ? NaN : sum(ages) / length(ages))
    push!(logger.total_agent_sugar, sum(wealth))
    push!(logger.total_landscape_sugar, sum(landscape.current))
    push!(logger.moved, events.moved)
    push!(logger.conflicts, events.conflicts)
    push!(logger.harvested, events.harvested)
    push!(logger.deaths, events.deaths)
    push!(logger.starvation_deaths, events.starvation_deaths)
    push!(logger.old_age_deaths, events.old_age_deaths)
    push!(logger.replacements, events.replacements)
    push!(logger.births, events.births)
    push!(logger.infected, count(citizen -> citizen.infected, citizens))
    push!(logger.infections, events.infections)
    push!(logger.recoveries, events.recoveries)
    return nothing
end

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

function percentile_value(values::AbstractVector{<:Real}, probability::Real)
    0.0 <= probability <= 1.0 ||
        throw(ArgumentError("probability must be between zero and one"))
    isempty(values) && return NaN
    sorted = sort(Float64.(values))
    position = 1.0 + (length(sorted) - 1) * probability
    lower = floor(Int, position)
    upper = ceil(Int, position)
    lower == upper && return sorted[lower]
    weight = position - lower
    return (1.0 - weight) * sorted[lower] + weight * sorted[upper]
end

"""
    lorenz_curve(values)

Return population and cumulative-value shares for a Lorenz curve. Values must be
nonnegative. A zero-total distribution is represented by the line of equality, which is
consistent with `gini_coefficient(values) == 0`.
"""
function lorenz_curve(values::AbstractVector{<:Real})
    any(<(0), values) && throw(ArgumentError("Lorenz curves require nonnegative values"))
    isempty(values) && return (population_share = Float64[], wealth_share = Float64[])
    sorted = sort(Float64.(values))
    population_share = collect(0:length(sorted)) ./ length(sorted)
    total = sum(sorted)
    wealth_share = total == 0.0 ? copy(population_share) : [0.0; cumsum(sorted) ./ total]
    return (population_share = population_share, wealth_share = wealth_share)
end

function wealth_statistics(values::AbstractVector{<:Real})
    isempty(values) && return (
        population = 0,
        total_wealth = 0.0,
        mean_wealth = NaN,
        wealth_std = NaN,
        minimum_wealth = NaN,
        wealth_p10 = NaN,
        wealth_p25 = NaN,
        median_wealth = NaN,
        wealth_p75 = NaN,
        wealth_p90 = NaN,
        maximum_wealth = NaN,
        gini = NaN,
        bottom_50_share = NaN,
        top_10_share = NaN,
    )
    any(<(0), values) && throw(ArgumentError("wealth values must be nonnegative"))
    sorted = sort(Float64.(values))
    population = length(sorted)
    total_wealth = sum(sorted)
    mean_wealth = total_wealth / population
    wealth_std = sqrt(sum((value - mean_wealth)^2 for value in sorted) / population)
    bottom_count = floor(Int, population / 2)
    top_count = max(1, ceil(Int, population / 10))
    share_denominator = total_wealth == 0.0 ? nothing : total_wealth
    return (
        population = population,
        total_wealth = total_wealth,
        mean_wealth = mean_wealth,
        wealth_std = wealth_std,
        minimum_wealth = first(sorted),
        wealth_p10 = percentile_value(sorted, 0.10),
        wealth_p25 = percentile_value(sorted, 0.25),
        median_wealth = percentile_value(sorted, 0.50),
        wealth_p75 = percentile_value(sorted, 0.75),
        wealth_p90 = percentile_value(sorted, 0.90),
        maximum_wealth = last(sorted),
        gini = gini_coefficient(sorted),
        bottom_50_share = isnothing(share_denominator) ? 0.5 :
                          sum(@view sorted[1:bottom_count]) / share_denominator,
        top_10_share = isnothing(share_denominator) ? 0.1 :
                       sum(@view sorted[(end - top_count + 1):end]) / share_denominator,
    )
end

function _mean_or_nan(values)
    isempty(values) && return NaN
    return sum(values) / length(values)
end

"""
    summary_statistics(world; burn_in = 0)

Compute final cross-sectional statistics and aggregate event statistics for a completed
or partially completed Sugarscape world. `burn_in` excludes that many initial logged
periods from time averages; cumulative event counts always cover the complete run.
"""
function summary_statistics(world; burn_in::Integer = 0)
    burn_in >= 0 || throw(ArgumentError("burn_in cannot be negative"))
    citizens = citizen_snapshot(world)
    wealth = [citizen.sugar for citizen in citizens]
    distribution = wealth_statistics(wealth)
    logger = Ark.get_resource(world, Logger)
    landscape = Ark.get_resource(world, SugarLandscape)
    period = Ark.get_resource(world, SimulationClock).step
    sample = isempty(logger.step) || burn_in >= length(logger.step) ? (1:0) :
             ((burn_in + 1):length(logger.step))
    females = count(citizen -> citizen.sex == :female, citizens)
    infected = count(citizen -> citizen.infected, citizens)
    population = length(citizens)

    return (
        period = period,
        population = population,
        females = females,
        males = population - females,
        infected = infected,
        infection_prevalence = population == 0 ? NaN : infected / population,
        total_agent_sugar = round(Int64, distribution.total_wealth),
        total_landscape_sugar = sum(landscape.current),
        total_sugar = round(Int64, distribution.total_wealth) + sum(landscape.current),
        mean_wealth = distribution.mean_wealth,
        wealth_std = distribution.wealth_std,
        minimum_wealth = distribution.minimum_wealth,
        wealth_p10 = distribution.wealth_p10,
        wealth_p25 = distribution.wealth_p25,
        median_wealth = distribution.median_wealth,
        wealth_p75 = distribution.wealth_p75,
        wealth_p90 = distribution.wealth_p90,
        maximum_wealth = distribution.maximum_wealth,
        gini = distribution.gini,
        bottom_50_wealth_share = distribution.bottom_50_share,
        top_10_wealth_share = distribution.top_10_share,
        mean_age = _mean_or_nan([citizen.age for citizen in citizens]),
        mean_vision = _mean_or_nan([citizen.vision for citizen in citizens]),
        mean_metabolism = _mean_or_nan([citizen.metabolism for citizen in citizens]),
        burn_in = burn_in,
        mean_population = _mean_or_nan(logger.population[sample]),
        mean_gini = _mean_or_nan(logger.gini[sample]),
        mean_harvested = _mean_or_nan(logger.harvested[sample]),
        mean_moved = _mean_or_nan(logger.moved[sample]),
        mean_conflicts = _mean_or_nan(logger.conflicts[sample]),
        cumulative_harvested = sum(logger.harvested),
        cumulative_deaths = sum(logger.deaths),
        cumulative_starvation_deaths = sum(logger.starvation_deaths),
        cumulative_old_age_deaths = sum(logger.old_age_deaths),
        cumulative_replacements = sum(logger.replacements),
        cumulative_births = sum(logger.births),
        cumulative_infections = sum(logger.infections),
        cumulative_recoveries = sum(logger.recoveries),
    )
end

"""Write a flat named tuple of summary statistics as a two-column TSV file."""
function write_summary_statistics(path::AbstractString, statistics::NamedTuple)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "metric\tvalue")
        for name in propertynames(statistics)
            println(io, name, '\t', getproperty(statistics, name))
        end
    end
    return path
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
    buffers = Ark.get_resource(world, SimulationBuffers)
    wealth = buffers.wealth
    sorted_wealth = buffers.sorted_wealth
    empty!(wealth)
    empty!(sorted_wealth)
    total_age = 0
    for (entities, sugars, ages) in Query(world, (Sugar, Age))
        @inbounds for i in eachindex(entities)
            push!(wealth, sugars[i].val)
            push!(sorted_wealth, sugars[i].val)
            total_age += ages[i].val
        end
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
    infected = 0
    for result in Query(world, (Infection,))
        infected += length(first(result))
    end

    push!(logger.step, clock.step)
    push!(logger.population, population)
    push!(logger.mean_wealth, isempty(wealth) ? NaN : total_wealth / population)
    push!(logger.median_wealth, median_wealth)
    push!(logger.gini, gini)
    push!(logger.mean_age, population == 0 ? NaN : total_age / population)
    push!(logger.total_agent_sugar, total_wealth)
    push!(logger.total_landscape_sugar, sum(landscape.current))
    push!(logger.moved, events.moved)
    push!(logger.conflicts, events.conflicts)
    push!(logger.harvested, events.harvested)
    push!(logger.deaths, events.deaths)
    push!(logger.starvation_deaths, events.starvation_deaths)
    push!(logger.old_age_deaths, events.old_age_deaths)
    push!(logger.replacements, events.replacements)
    push!(logger.births, events.births)
    push!(logger.infected, infected)
    push!(logger.infections, events.infections)
    push!(logger.recoveries, events.recoveries)
    return nothing
end

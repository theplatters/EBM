"""State needed to draw one car without retaining the simulation world."""
struct TrafficCarState
    entity::Ark.Entity
    lane::Int
    cell::Int
    direction::Direction
    age::Int
    habitus::Float64
    decision::Float64
    strategy::Union{Nothing, DriverStrategy}
    speed::Int
    capabilities::UInt8
    social_habitus::Float64
    risk_aversion::Float64
end

"""Immutable, renderer-independent view of the Traffic simulation at one step."""
struct TrafficSnapshot
    step::Int
    ring_width::Int
    ring_height::Int
    cars::Vector{TrafficCarState}
    replacements::Int
    cumulative_replacements::Int
    treatment::Symbol
end

"""
    traffic_snapshot(world; step = 0)

Copy the current car and ring state out of an Ark world. The returned snapshot can
be plotted, stored, or passed between tasks without retaining the mutable world.
"""
function traffic_snapshot(world; step::Integer = 0)
    ring = Ark.get_resource(world, Ring)
    cars = TrafficCarState[]

    if Ark.has_resource(world, CapabilityModel)
        for (entities, positions, directions, ages, decisions, speeds) in Query(
                world, (Position, Direction, Step, LR, Speed),
            )
            @inbounds for index in eachindex(entities)
                entity = entities[index]
                habitus = if Ark.has_components(world, entity, (Habitus,))
                    Ark.get_components(world, entity, (Habitus,))[1].val
                else
                    0.0
                end
                social_habitus = if Ark.has_components(world, entity, (SocialHabitus,))
                    Ark.get_components(world, entity, (SocialHabitus,))[1].value
                else
                    0.0
                end
                push!(
                    cars,
                    TrafficCarState(
                        entity,
                        positions[index].x,
                        positions[index].y,
                        directions[index],
                        ages[index].val,
                        habitus,
                        decisions[index].val,
                        nothing,
                        speeds[index].val,
                        capability_mask(world, entity),
                        social_habitus,
                        Ark.get_components(world, entity, (RiskAversion,))[1].value,
                    ),
                )
            end
        end
    else
        for (entities, positions, directions, ages, habitus, decisions, strategies) in Query(
                world,
                (Position, Direction, Step, Habitus, LR, DriverStrategy),
            )
            @inbounds for index in eachindex(entities)
                push!(
                    cars,
                    TrafficCarState(
                        entities[index],
                        positions[index].x,
                        positions[index].y,
                        directions[index],
                        ages[index].val,
                        habitus[index].val,
                        decisions[index].val,
                        strategies[index],
                        1,
                        UInt8(0),
                        0.0,
                        0.0,
                    ),
                )
            end
        end
    end

    sort!(cars; by = car -> (car.cell, car.lane, Int(car.direction)))
    logger = Ark.get_resource(world, Logger)
    replacements = isempty(logger.deaths) ? 0 : last(logger.deaths)
    treatment = if Ark.has_resource(world, CapabilityModel)
        model = Ark.get_resource(world, CapabilityModel)
        model.replacement_policy isa EvolutionaryReplacement ? :evolutionary : :entry_draw
    else
        :legacy
    end
    return TrafficSnapshot(
        Int(step), Int(ring.width), Int(ring.height), cars,
        replacements, sum(logger.deaths), treatment,
    )
end

"""
    traffic_history(args; every = 1, include_initial = true)

Run a simulation and capture visualization snapshots every `every` steps. The
final state is always included, even when it falls between capture intervals.
"""
function traffic_history(args::ModelArgs; every::Integer = 1, include_initial::Bool = true)
    every > 0 || throw(ArgumentError("every must be positive"))

    world = setup_world(args)
    history = TrafficSnapshot[]
    include_initial && push!(history, traffic_snapshot(world; step = 0))

    for iteration in 1:args.steps
        step!(world, args.prediction_strategy)
        if iteration % every == 0 || iteration == args.steps
            push!(history, traffic_snapshot(world; step = iteration))
        end
    end

    return history
end

const TRAFFIC_DIRECTION_COLORS = Dict(
    Clockwise => Makie.wong_colors()[1],
    Counterclockwise => Makie.wong_colors()[2],
)

const TRAFFIC_STRATEGY_COLORS = Dict(
    kind => color
        for (kind, color) in zip(instances(StrategyKind), Makie.resample_cmap(:tab10, 8))
)

const TRAFFIC_SPEED_COLORS = Makie.resample_cmap(:viridis, 3)

const TRAFFIC_CAPABILITY_LABELS = (
    "Same-dir.",
    "Opposite-dir.",
    "Avoidance",
    "Habit",
    "Convention",
    "Social habit",
)

const TRAFFIC_CAPABILITY_COLORS = Makie.resample_cmap(:Set2, 6)

_is_capability_snapshot(snapshot::TrafficSnapshot) =
    any(car -> isnothing(car.strategy), snapshot.cars)

function _treatment_label(snapshot::TrafficSnapshot)
    snapshot.treatment == :evolutionary && return "evolutionary replacement"
    snapshot.treatment == :entry_draw && return "entry-draw replacement"
    return "legacy forecast treatment"
end

@inline _has_capability(car::TrafficCarState, index::Int) =
    car.capabilities & (UInt8(1) << (index - 1)) != 0

function _resolve_color_by(snapshot::TrafficSnapshot, color_by::Symbol)
    color_by == :auto && return _is_capability_snapshot(snapshot) ? :speed : :direction
    color_by in (:direction, :strategy, :speed) ||
        throw(ArgumentError("color_by must be :auto, :direction, :strategy, or :speed"))
    return color_by
end

function _car_point(snapshot::TrafficSnapshot, car::TrafficCarState)
    lane_spacing = 0.24
    radius = 1.0 - (snapshot.ring_width - car.lane) * lane_spacing
    angle = π / 2 - 2π * (car.cell - 1) / snapshot.ring_height
    return Point2f(radius * cos(angle), radius * sin(angle))
end

_car_points(snapshot::TrafficSnapshot) = [_car_point(snapshot, car) for car in snapshot.cars]

function _car_rotations(snapshot::TrafficSnapshot)
    return map(snapshot.cars) do car
        angle = π / 2 - 2π * (car.cell - 1) / snapshot.ring_height
        car.direction == Clockwise ? angle - π : angle
    end
end

function _car_colors(snapshot::TrafficSnapshot, color_by::Symbol)
    color_by == :direction &&
        return [TRAFFIC_DIRECTION_COLORS[car.direction] for car in snapshot.cars]
    if color_by == :strategy
        all(!isnothing(car.strategy) for car in snapshot.cars) ||
            throw(ArgumentError("capability cars do not have forecast strategies"))
        return [TRAFFIC_STRATEGY_COLORS[something(car.strategy).kind] for car in snapshot.cars]
    end
    color_by == :speed && return [TRAFFIC_SPEED_COLORS[car.speed] for car in snapshot.cars]
    throw(ArgumentError("color_by must be :direction, :strategy, or :speed"))
end

function _lane_counts(snapshot::TrafficSnapshot)
    return [count(car -> car.lane == lane, snapshot.cars) for lane in 1:snapshot.ring_width]
end

function _direction_counts(snapshot::TrafficSnapshot)
    return [
        count(car -> car.direction == Clockwise, snapshot.cars),
        count(car -> car.direction == Counterclockwise, snapshot.cars),
    ]
end

_speed_counts(snapshot::TrafficSnapshot) =
    [count(car -> car.speed == speed, snapshot.cars) for speed in 1:3]

function _relative_side_counts(snapshot::TrafficSnapshot)
    return [
        count(car -> relative_lane_sign(car.lane, car.direction) > 0, snapshot.cars),
        count(car -> relative_lane_sign(car.lane, car.direction) < 0, snapshot.cars),
    ]
end

_capability_counts(snapshot::TrafficSnapshot) = [
    count(car -> _has_capability(car, index), snapshot.cars)
        for index in eachindex(TRAFFIC_CAPABILITY_LABELS)
]

function _convention_strength(snapshot::TrafficSnapshot)
    return abs(_safe_mean(
        relative_lane_sign(car.lane, car.direction) for car in snapshot.cars
    ))
end

function _mean_abs_habitus(snapshot::TrafficSnapshot)
    values = [
        abs(car.habitus) for car in snapshot.cars if _has_capability(car, 4)
    ]
    return _safe_mean(values)
end

function _mean_abs_social_habitus(snapshot::TrafficSnapshot)
    values = [
        abs(car.social_habitus) for car in snapshot.cars if _has_capability(car, 6)
    ]
    return _safe_mean(values)
end

_safe_mean(values) = isempty(values) ? 0.0 : mean(values)

function _snapshot_summary(snapshot::TrafficSnapshot)
    cars = snapshot.cars
    capacity = snapshot.ring_width * snapshot.ring_height
    occupancy = isempty(cars) ? 0.0 : 100 * length(cars) / capacity
    mean_age = _safe_mean(car.age for car in cars)
    mean_speed = _safe_mean(car.speed for car in cars)
    prefer_left = count(car -> car.decision > 0, cars)

    if _is_capability_snapshot(snapshot)
        speed_three_share = isempty(cars) ? 0.0 : count(car -> car.speed == 3, cars) / length(cars)
        return join(
            [
                "Cars                       $(length(cars)) / $capacity",
                "Road occupancy            $(round(occupancy; digits = 1))%",
                "Mean speed                 $(round(mean_speed; digits = 2)) cells/tick",
                "Share at speed 3           $(round(100 * speed_three_share; digits = 1))%",
                "Convention strength        $(round(_convention_strength(snapshot); digits = 3))",
                "Mean |habitus| (carriers)  $(round(_mean_abs_habitus(snapshot); digits = 3))",
                "Mean |social habit|        $(round(_mean_abs_social_habitus(snapshot); digits = 3))",
                "Replacements this tick     $(snapshot.replacements)",
                "Cumulative replacements    $(snapshot.cumulative_replacements)",
                "Mean car age               $(round(mean_age; digits = 1)) steps",
            ],
            '\n',
        )
    end

    mean_habitus = _safe_mean(car.habitus for car in cars)
    mean_abs_habitus = _safe_mean(abs(car.habitus) for car in cars)

    return join(
        [
            "Cars                 $(length(cars)) / $capacity",
            "Road occupancy      $(round(occupancy; digits = 1))%",
            "Mean car age         $(round(mean_age; digits = 1)) steps",
            "Mean habitus         $(round(mean_habitus; digits = 3))",
            "Mean |habitus|       $(round(mean_abs_habitus; digits = 3))",
            "Mean speed           $(round(mean_speed; digits = 2)) cells/tick",
            "Left-lane intent    $prefer_left",
        ],
        '\n',
    )
end

function _traffic_dashboard(
        snapshot::Observable; size = (1180, 760), car_size = 18,
        color_by::Symbol = :auto,
    )
    initial = snapshot[]
    capability_mode = _is_capability_snapshot(initial)
    resolved_color_by = _resolve_color_by(initial, color_by)
    figure = Figure(size = size, backgroundcolor = :white)

    title = lift(snapshot) do state
        "Traffic simulation overview — $(_treatment_label(state)) — step $(state.step)"
    end
    Label(figure[1, 1:2], title; fontsize = 22, font = :bold, tellwidth = false)

    ring_axis = Axis(
        figure[2:5, 1];
        aspect = DataAspect(),
        title = "Individual cars on the two-lane ring (color: $(resolved_color_by))",
        backgroundcolor = RGBf(0.97, 0.97, 0.97),
    )
    hidedecorations!(ring_axis)
    hidespines!(ring_axis)
    limits!(ring_axis, -1.18, 1.18, -1.18, 1.18)

    angles = range(0, 2π; length = 361)
    for lane in 1:initial.ring_width
        radius = 1.0 - (initial.ring_width - lane) * 0.24
        lines!(
            ring_axis,
            radius .* cos.(angles),
            radius .* sin.(angles);
            color = RGBf(0.70, 0.70, 0.70),
            linewidth = 2,
        )
    end

    points = lift(_car_points, snapshot)
    rotations = lift(_car_rotations, snapshot)
    colors = lift(state -> _car_colors(state, resolved_color_by), snapshot)
    scatter!(
        ring_axis,
        points;
        marker = :utriangle,
        rotation = rotations,
        color = colors,
        markersize = car_size,
        strokecolor = :white,
        strokewidth = 0.8,
    )

    if resolved_color_by == :direction
        center_text = lift(state -> "$(length(state.cars))\ncars", snapshot)
        text!(
            ring_axis,
            0,
            0;
            text = center_text,
            align = (:center, :center),
            fontsize = 19,
            color = RGBf(0.25, 0.25, 0.25),
        )
    end

    if resolved_color_by == :strategy
        strategy_kinds = sort!(
            unique(something(car.strategy).kind for car in initial.cars); by = Int,
        )
        for kind in strategy_kinds
            scatter!(
                ring_axis,
                [Point2f(NaN, NaN)];
                color = TRAFFIC_STRATEGY_COLORS[kind],
                marker = :utriangle,
                markersize = 13,
                label = STRATEGY_NAMES[kind],
            )
        end
        axislegend(ring_axis; position = :cc, framevisible = false)
    end

    if resolved_color_by == :speed
        for speed in 1:3
            scatter!(
                ring_axis,
                [Point2f(NaN, NaN)];
                color = TRAFFIC_SPEED_COLORS[speed],
                marker = :utriangle,
                markersize = 13,
                label = "Speed $speed",
            )
        end
        axislegend(ring_axis; position = :cc, framevisible = false)
    end

    count_limit = max(1, ceil(Int, 1.1 * length(initial.cars)))
    if capability_mode
        speed_axis = Axis(
            figure[2, 2];
            title = "Current speed distribution",
            xticks = (1:3, ["1", "2", "3"]),
            xlabel = "cells per tick",
            ylabel = "cars",
        )
        speed_counts = lift(_speed_counts, snapshot)
        barplot!(speed_axis, 1:3, speed_counts; color = TRAFFIC_SPEED_COLORS)
        ylims!(speed_axis, 0, count_limit)

        capability_axis = Axis(
            figure[3, 2];
            title = "Capability prevalence",
            xticks = (eachindex(TRAFFIC_CAPABILITY_LABELS), collect(TRAFFIC_CAPABILITY_LABELS)),
            ylabel = "cars",
            xticklabelrotation = π / 8,
        )
        capability_counts = lift(_capability_counts, snapshot)
        barplot!(
            capability_axis, eachindex(TRAFFIC_CAPABILITY_LABELS), capability_counts;
            color = TRAFFIC_CAPABILITY_COLORS,
        )
        ylims!(capability_axis, 0, count_limit)

        side_axis = Axis(
            figure[4, 2];
            title = "Realized side relative to travel",
            xticks = (1:2, ["Left", "Right"]),
            ylabel = "cars",
        )
        side_counts = lift(_relative_side_counts, snapshot)
        barplot!(
            side_axis, 1:2, side_counts;
            color = [Makie.wong_colors()[3], Makie.wong_colors()[4]],
        )
        ylims!(side_axis, 0, count_limit)
    else
        lane_axis = Axis(
            figure[2, 2];
            title = "Cars by lane",
            xticks = (
                1:initial.ring_width,
                ["Lane $lane" for lane in 1:initial.ring_width],
            ),
            ylabel = "cars",
        )
        lane_counts = lift(_lane_counts, snapshot)
        barplot!(lane_axis, 1:initial.ring_width, lane_counts; color = Makie.wong_colors()[3])
        ylims!(lane_axis, 0, count_limit)

        direction_axis = Axis(
            figure[3, 2];
            title = "Travel direction",
            xticks = (1:2, ["Clockwise", "Counter"]),
            ylabel = "cars",
        )
        direction_counts = lift(_direction_counts, snapshot)
        barplot!(
            direction_axis,
            1:2,
            direction_counts;
            color = [
                TRAFFIC_DIRECTION_COLORS[Clockwise],
                TRAFFIC_DIRECTION_COLORS[Counterclockwise],
            ],
        )
        ylims!(direction_axis, 0, count_limit)
    end

    summary = lift(_snapshot_summary, snapshot)
    Label(
        capability_mode ? figure[5, 2] : figure[4:5, 2],
        summary;
        halign = :left,
        valign = :top,
        justification = :left,
        fontsize = 15,
        padding = (14, 14, 10, 10),
    )

    colsize!(figure.layout, 1, Relative(0.66))
    rowgap!(figure.layout, 10)
    colgap!(figure.layout, 18)
    return figure
end

"""
    plot_traffic(snapshot; size = (1180, 760), car_size = 18, color_by = :auto)
    plot_traffic(world; step = 0, ...)

Create a compact dashboard of the ring, every individual car, lane/direction
counts, and key current-state statistics.

Capability snapshots automatically use speed coloring and show speed,
capability, convention, and replacement diagnostics. Legacy snapshots default
to direction coloring. Pass `color_by` to override the automatic choice.
"""
function plot_traffic(snapshot::TrafficSnapshot; kwargs...)
    return _traffic_dashboard(Observable(snapshot); kwargs...)
end

function plot_traffic(world; step::Integer = 0, kwargs...)
    return plot_traffic(traffic_snapshot(world; step = step); kwargs...)
end

function _replacement_rates(history)
    rates = zeros(Float64, length(history))
    @inbounds for index in 2:length(history)
        elapsed_steps = history[index].step - history[index - 1].step
        elapsed_steps > 0 || continue
        replacements = history[index].cumulative_replacements -
                       history[index - 1].cumulative_replacements
        car_steps = length(history[index].cars) * elapsed_steps
        rates[index] = car_steps == 0 ? 0.0 : replacements / car_steps
    end
    return rates
end

"""
    plot_traffic_history(history; size = (1180, 760))

Plot capability-model dynamics: speed shares, convention and habit strength,
replacement rate between captured snapshots, and the population prevalence of
each optional capability.
"""
function plot_traffic_history(
        history::AbstractVector{<:TrafficSnapshot}; size = (1180, 760),
    )
    isempty(history) && throw(ArgumentError("history must contain at least one snapshot"))
    all(_is_capability_snapshot, history) ||
        throw(ArgumentError("plot_traffic_history currently requires capability snapshots"))
    all(snapshot -> snapshot.treatment == first(history).treatment, history) ||
        throw(ArgumentError("history must contain one replacement treatment"))
    issorted(snapshot.step for snapshot in history) ||
        throw(ArgumentError("history steps must be sorted"))

    steps = [snapshot.step for snapshot in history]
    figure = Figure(size = size, backgroundcolor = :white)
    Label(
        figure[1, 1:2],
        "Capability traffic dynamics — $(_treatment_label(first(history))) — " *
        "steps $(first(steps))–$(last(steps))";
        fontsize = 22,
        font = :bold,
        tellwidth = false,
    )

    speed_axis = Axis(
        figure[2, 1];
        title = "Speed distribution",
        xlabel = "step",
        ylabel = "population share",
    )
    for speed in 1:3
        shares = [
            isempty(snapshot.cars) ? 0.0 :
            count(car -> car.speed == speed, snapshot.cars) / length(snapshot.cars)
                for snapshot in history
        ]
        maximum(shares) > 0.0 || continue
        lines!(
            speed_axis, steps, shares;
            color = TRAFFIC_SPEED_COLORS[speed],
            linewidth = 3,
            label = "Speed $speed",
        )
    end
    ylims!(speed_axis, 0, 1)
    axislegend(speed_axis; position = :rc, framevisible = false)

    coordination_axis = Axis(
        figure[2, 2];
        title = "Emergent coordination",
        xlabel = "step",
        ylabel = "strength",
    )
    lines!(
        coordination_axis,
        steps,
        [_convention_strength(snapshot) for snapshot in history];
        color = Makie.wong_colors()[1],
        linewidth = 3,
        label = "Realized convention",
    )
    lines!(
        coordination_axis,
        steps,
        [_mean_abs_habitus(snapshot) for snapshot in history];
        color = Makie.wong_colors()[2],
        linewidth = 3,
        label = "Mean |habitus|",
    )
    social_habit_strength = [_mean_abs_social_habitus(snapshot) for snapshot in history]
    if maximum(social_habit_strength) > 0.0
        lines!(
            coordination_axis,
            steps,
            social_habit_strength;
            color = Makie.wong_colors()[3],
            linewidth = 3,
            label = "Mean |social habit|",
        )
    end
    ylims!(coordination_axis, 0, 1)
    axislegend(coordination_axis; position = :rb, framevisible = false)

    replacement_axis = Axis(
        figure[3, 1];
        title = "Replacement pressure",
        xlabel = "step",
        ylabel = "replacements per car-step",
    )
    replacement_rates = _replacement_rates(history)
    lines!(
        replacement_axis, steps, replacement_rates;
        color = Makie.wong_colors()[6],
        linewidth = 3,
    )
    scatter!(
        replacement_axis, steps, replacement_rates;
        color = Makie.wong_colors()[6],
        markersize = 8,
    )
    ylims!(replacement_axis, 0, max(0.01, 1.1 * maximum(replacement_rates)))

    capability_axis = Axis(
        figure[3, 2];
        title = "Capability composition",
        xlabel = "step",
        ylabel = "population share",
    )
    for capability_index in eachindex(TRAFFIC_CAPABILITY_LABELS)
        shares = [
            isempty(snapshot.cars) ? 0.0 :
            count(car -> _has_capability(car, capability_index), snapshot.cars) /
            length(snapshot.cars)
                for snapshot in history
        ]
        maximum(shares) > 0.0 || continue
        lines!(
            capability_axis, steps, shares;
            color = TRAFFIC_CAPABILITY_COLORS[capability_index],
            linewidth = capability_index <= 3 ? 2 : 3,
            linestyle = capability_index <= 3 ? :dot : :solid,
            label = TRAFFIC_CAPABILITY_LABELS[capability_index],
        )
    end
    ylims!(capability_axis, 0, 1)
    axislegend(capability_axis; position = :rc, framevisible = false)

    rowgap!(figure.layout, 16)
    colgap!(figure.layout, 18)
    return figure
end

"""
    record_traffic(history, filename; framerate = 10, ...)

Record snapshots as an animation. All snapshots must use the same ring geometry.
The output format is inferred from `filename`, for example `plots/traffic.mp4`.
"""
function record_traffic(
    history::AbstractVector{<:TrafficSnapshot},
    filename::AbstractString;
    framerate::Real = 10,
    kwargs...,
)
    isempty(history) && throw(ArgumentError("history must contain at least one snapshot"))
    geometry = (history[1].ring_width, history[1].ring_height)
    all(state -> (state.ring_width, state.ring_height) == geometry, history) ||
        throw(ArgumentError("all snapshots must use the same ring geometry"))

    snapshot = Observable(first(history))
    figure = _traffic_dashboard(snapshot; kwargs...)
    record(figure, filename, history; framerate = framerate) do state
        snapshot[] = state
    end
    return filename
end

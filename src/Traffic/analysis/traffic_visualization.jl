"""State needed to draw one car without retaining the simulation world."""
struct TrafficCarState
    entity::Ark.Entity
    lane::Int
    cell::Int
    direction::Direction
    age::Int
    habitus::Float64
    decision::Float64
    strategy::DriverStrategy
end

"""Immutable, renderer-independent view of the Traffic simulation at one step."""
struct TrafficSnapshot
    step::Int
    ring_width::Int
    ring_height::Int
    cars::Vector{TrafficCarState}
end

"""
    traffic_snapshot(world; step = 0)

Copy the current car and ring state out of an Ark world. The returned snapshot can
be plotted, stored, or passed between tasks without retaining the mutable world.
"""
function traffic_snapshot(world; step::Integer = 0)
    ring = Ark.get_resource(world, Ring)
    cars = TrafficCarState[]

    for (entities, positions, directions, ages, habitus, decisions, strategies) in Query(
        world,
        (Position, Direction, Step, Habitus, LR, DriverStrategy),
    )
        @inbounds for i in eachindex(entities)
            push!(
                cars,
                TrafficCarState(
                    entities[i],
                    positions[i].x,
                    positions[i].y,
                    directions[i],
                    ages[i].val,
                    habitus[i].val,
                    decisions[i].val,
                    strategies[i],
                ),
            )
        end
    end

    sort!(cars; by = car -> (car.cell, car.lane, Int(car.direction)))
    return TrafficSnapshot(Int(step), Int(ring.width), Int(ring.height), cars)
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
    color_by == :strategy &&
        return [TRAFFIC_STRATEGY_COLORS[car.strategy.kind] for car in snapshot.cars]
    throw(ArgumentError("color_by must be :direction or :strategy"))
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

_safe_mean(values) = isempty(values) ? 0.0 : mean(values)

function _snapshot_summary(snapshot::TrafficSnapshot)
    cars = snapshot.cars
    capacity = snapshot.ring_width * snapshot.ring_height
    occupancy = isempty(cars) ? 0.0 : 100 * length(cars) / capacity
    mean_age = _safe_mean(car.age for car in cars)
    mean_habitus = _safe_mean(car.habitus for car in cars)
    mean_abs_habitus = _safe_mean(abs(car.habitus) for car in cars)
    prefer_left = count(car -> car.decision > 0, cars)

    return join(
        [
            "Cars                 $(length(cars)) / $capacity",
            "Road occupancy      $(round(occupancy; digits = 1))%",
            "Mean car age         $(round(mean_age; digits = 1)) steps",
            "Mean habitus         $(round(mean_habitus; digits = 3))",
            "Mean |habitus|       $(round(mean_abs_habitus; digits = 3))",
            "Left-lane intent    $prefer_left",
        ],
        '\n',
    )
end

function _traffic_dashboard(
        snapshot::Observable; size = (1100, 680), car_size = 18,
        color_by::Symbol = :direction,
    )
    initial = snapshot[]
    figure = Figure(size = size, backgroundcolor = :white)

    title = lift(snapshot) do state
        "Traffic simulation overview — step $(state.step)"
    end
    Label(figure[1, 1:2], title; fontsize = 22, font = :bold, tellwidth = false)

    ring_axis = Axis(
        figure[2:4, 1];
        aspect = DataAspect(),
        title = "Individual cars on the two-lane ring (color: $(color_by))",
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
    colors = lift(state -> _car_colors(state, color_by), snapshot)
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

    if color_by == :direction
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

    if color_by == :strategy
        strategy_kinds = sort!(unique(car.strategy.kind for car in initial.cars); by = Int)
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

    lane_axis = Axis(
        figure[2, 2];
        title = "Cars by lane",
        xticks = (1:initial.ring_width, ["Lane $lane" for lane in 1:initial.ring_width]),
        ylabel = "cars",
    )
    lane_counts = lift(_lane_counts, snapshot)
    barplot!(lane_axis, 1:initial.ring_width, lane_counts; color = Makie.wong_colors()[3])
    count_limit = max(1, ceil(Int, 1.1 * length(initial.cars)))
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

    summary = lift(_snapshot_summary, snapshot)
    Label(
        figure[4, 2],
        summary;
        halign = :left,
        valign = :top,
        justification = :left,
        fontsize = 15,
        padding = (14, 14, 10, 10),
    )

    colsize!(figure.layout, 1, Relative(0.68))
    rowgap!(figure.layout, 10)
    colgap!(figure.layout, 18)
    return figure
end

"""
    plot_traffic(snapshot; size = (1100, 680), car_size = 18)
    plot_traffic(world; step = 0, ...)

Create a compact dashboard of the ring, every individual car, lane/direction
counts, and key current-state statistics.
"""
function plot_traffic(snapshot::TrafficSnapshot; kwargs...)
    return _traffic_dashboard(Observable(snapshot); kwargs...)
end

function plot_traffic(world; step::Integer = 0, kwargs...)
    return plot_traffic(traffic_snapshot(world; step = step); kwargs...)
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

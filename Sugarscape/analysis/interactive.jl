"""Renderer-independent state used by the interactive Sugarscape dashboard."""
struct SugarscapeSnapshot
    step::Int64
    landscape::Matrix{Int64}
    citizens::Vector{CitizenState}
    history_steps::Vector{Int64}
    history_gini::Vector{Float64}
    population::Int64
    mean_wealth::Float64
    gini::Float64
    total_agent_sugar::Int64
    total_landscape_sugar::Int64
    moved::Int64
    conflicts::Int64
    harvested::Int64
    deaths::Int64
    births::Int64
    infected::Int64
end

"""
    sugarscape_snapshot(world)

Copy the current state and the logged inequality history out of a mutable Ark world.
The result is safe to retain while the simulation advances.
"""
function sugarscape_snapshot(world)
    clock = Ark.get_resource(world, SimulationClock)
    landscape = Ark.get_resource(world, SugarLandscape)
    events = Ark.get_resource(world, StepEvents)
    logger = Ark.get_resource(world, Logger)
    citizens = citizen_snapshot(world)
    wealth = [citizen.sugar for citizen in citizens]
    return SugarscapeSnapshot(
        clock.step,
        copy(landscape.current),
        citizens,
        copy(logger.step),
        copy(logger.gini),
        length(citizens),
        isempty(wealth) ? NaN : sum(wealth) / length(wealth),
        gini_coefficient(wealth),
        sum(wealth),
        sum(landscape.current),
        events.moved,
        events.conflicts,
        events.harvested,
        events.deaths,
        events.births,
        count(citizen -> citizen.infected, citizens),
    )
end

"""A live Sugarscape world together with its Makie dashboard and playback state."""
mutable struct SugarscapeVisualization{W}
    figure::Figure
    args::ModelArgs
    world::W
    state::Observable{SugarscapeSnapshot}
    running::Observable{Bool}
    frame_delay::Observable{Float64}
    task::Union{Nothing, Task}
end

function _visualization_status(snapshot::SugarscapeSnapshot, running::Bool, horizon::Int)
    mode = running ? "running" : snapshot.step >= horizon ? "complete" : "paused"
    return "Sugarscape — period $(snapshot.step) / $horizon — $mode"
end

function _visualization_summary(snapshot::SugarscapeSnapshot)
    mean_wealth = isnan(snapshot.mean_wealth) ? "—" : string(round(snapshot.mean_wealth; digits = 2))
    gini = isnan(snapshot.gini) ? "—" : string(round(snapshot.gini; digits = 3))
    return join(
        [
            "Citizens                 $(snapshot.population)",
            "Mean wealth              $mean_wealth",
            "Gini coefficient         $gini",
            "Citizen sugar            $(snapshot.total_agent_sugar)",
            "Landscape sugar          $(snapshot.total_landscape_sugar)",
            "Moved / harvested        $(snapshot.moved) / $(snapshot.harvested)",
            "Conflicts                $(snapshot.conflicts)",
            "Births / deaths          $(snapshot.births) / $(snapshot.deaths)",
            "Currently infected       $(snapshot.infected)",
        ],
        '\n',
    )
end

function _build_interactive_figure(
    state::Observable{SugarscapeSnapshot},
    running::Observable{Bool},
    params::ModelParams,
    horizon::Int;
    size = (1250, 780),
    citizen_size::Real = 9,
    framerate::Integer = 5,
)
    figure = Figure(size = size, backgroundcolor = :white)
    title = lift(state, running) do snapshot, is_running
        _visualization_status(snapshot, is_running, horizon)
    end
    Label(figure[1, 1:2], title; fontsize = 23, font = :bold, tellwidth = false)

    landscape_axis = Axis(
        figure[2:4, 1];
        xlabel = "x",
        ylabel = "y",
        title = "Resources and citizens (color: wealth)",
        aspect = DataAspect(),
    )
    heatmap!(
        landscape_axis,
        1:params.width,
        1:params.height,
        lift(snapshot -> snapshot.landscape, state);
        colormap = :YlOrBr,
        colorrange = (0, max(1, params.maximum_patch_sugar)),
    )
    citizen_x = lift(state) do snapshot
        [citizen.position.x for citizen in snapshot.citizens]
    end
    citizen_y = lift(state) do snapshot
        [citizen.position.y for citizen in snapshot.citizens]
    end
    citizen_wealth = lift(state) do snapshot
        [citizen.sugar for citizen in snapshot.citizens]
    end
    citizens = scatter!(
        landscape_axis,
        citizen_x,
        citizen_y;
        color = citizen_wealth,
        colormap = :viridis,
        markersize = citizen_size,
        strokecolor = (:black, 0.5),
        strokewidth = 0.5,
    )
    infected_x = lift(state) do snapshot
        [citizen.position.x for citizen in snapshot.citizens if citizen.infected]
    end
    infected_y = lift(state) do snapshot
        [citizen.position.y for citizen in snapshot.citizens if citizen.infected]
    end
    scatter!(
        landscape_axis,
        infected_x,
        infected_y;
        marker = :xcross,
        color = :red,
        markersize = citizen_size * 1.45,
        strokewidth = 2,
    )
    Colorbar(figure[2:4, 1, Right()], citizens)
    limits!(landscape_axis, 0.5, params.width + 0.5, 0.5, params.height + 0.5)

    inequality_axis = style_axis!(Axis(
        figure[2, 2];
        xlabel = "Period",
        ylabel = "Gini coefficient",
        title = "Wealth inequality",
    ))
    history_steps = lift(snapshot -> snapshot.history_steps, state)
    history_gini = lift(snapshot -> snapshot.history_gini, state)
    lines!(
        inequality_axis,
        history_steps,
        history_gini;
        color = SUGARSCAPE_COLORS[3],
        linewidth = 2.5,
    )
    xlims!(inequality_axis, 0, max(1, horizon))
    ylims!(inequality_axis, 0, 1)

    wealth_axis = style_axis!(Axis(
        figure[3, 2];
        xlabel = "Age",
        ylabel = "Sugar wealth",
        title = "Current age–wealth distribution",
    ))
    citizen_age = lift(state) do snapshot
        [citizen.age for citizen in snapshot.citizens]
    end
    scatter!(
        wealth_axis,
        citizen_age,
        citizen_wealth;
        color = (SUGARSCAPE_COLORS[2], 0.65),
        markersize = 7,
    )
    xlims!(wealth_axis, 0, max(1, params.maximum_lifespan))

    summary = lift(_visualization_summary, state)
    Label(
        figure[4, 2],
        summary;
        halign = :left,
        valign = :top,
        justification = :left,
        fontsize = 14,
        padding = (12, 12, 4, 4),
    )

    controls = GridLayout(figure[5, 1:2])
    reset_button = Button(controls[1, 1]; label = "Reset")
    step_button = Button(controls[1, 2]; label = "Step")
    run_button = Button(controls[1, 3]; label = "Run / pause")
    Label(controls[1, 4], "Playback speed")
    speed = Slider(controls[1, 5]; range = 1:20, startvalue = framerate)
    speed_label = lift(value -> "$(round(Int, value)) steps/s", speed.value)
    Label(controls[1, 6], speed_label; width = 72)
    colsize!(controls, 5, Relative(0.35))

    colsize!(figure.layout, 1, Relative(0.66))
    rowgap!(figure.layout, 10)
    colgap!(figure.layout, 24)
    return figure, reset_button, step_button, run_button, speed
end

function _advance!(visualization::SugarscapeVisualization, steps::Integer)
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    current_step = visualization.state[].step
    remaining = max(0, visualization.args.steps - current_step)
    for _ in 1:min(steps, remaining)
        step!(visualization.world)
    end
    visualization.state[] = sugarscape_snapshot(visualization.world)
    return visualization
end

"""
    step!(visualization::SugarscapeVisualization, steps = 1)

Pause playback and advance the interactive visualization by up to `steps` periods.
The configured `ModelArgs.steps` value is the playback horizon.
"""
function step!(visualization::SugarscapeVisualization, steps::Integer = 1)
    stop!(visualization)
    return _advance!(visualization, steps)
end

"""Stop automatic playback of an interactive Sugarscape visualization."""
function stop!(visualization::SugarscapeVisualization)
    visualization.running[] = false
    return visualization
end

"""Reset an interactive Sugarscape visualization to its seeded initial state."""
function reset!(visualization::SugarscapeVisualization)
    stop!(visualization)
    visualization.world = setup_world(visualization.args)
    visualization.state[] = sugarscape_snapshot(visualization.world)
    return visualization
end

"""Start automatic playback of an interactive Sugarscape visualization."""
function play!(visualization::SugarscapeVisualization)
    visualization.state[].step >= visualization.args.steps && return visualization
    if !isnothing(visualization.task) && !istaskdone(visualization.task)
        visualization.running[] = true
        return visualization
    end
    visualization.running[] = true
    visualization.task = @async begin
        try
            while visualization.running[] &&
                    visualization.state[].step < visualization.args.steps
                _advance!(visualization, 1)
                sleep(visualization.frame_delay[])
            end
        finally
            visualization.running[] = false
        end
    end
    return visualization
end

"""
    interactive_sugarscape(args = ModelArgs(); size = (1250, 780),
                           citizen_size = 9, framerate = 5)

Create an interactive Makie dashboard backed by a live, seeded Sugarscape world. Use
the buttons to reset, advance one period, or run/pause the model, and the slider to
change playback speed. `args.steps` is the maximum displayed period.

For interactive display in a notebook or browser, activate WGLMakie before calling
this function. The return value exposes the dashboard as `visualization.figure` and
can also be advanced programmatically with `step!(visualization, n)`.
"""
function interactive_sugarscape(
    args::ModelArgs = ModelArgs();
    size = (1250, 780),
    citizen_size::Real = 9,
    framerate::Integer = 5,
)
    validate(args)
    args.steps > 0 || throw(ArgumentError("interactive visualization requires at least one step"))
    citizen_size > 0 || throw(ArgumentError("citizen_size must be positive"))
    framerate in 1:20 || throw(ArgumentError("framerate must be between 1 and 20"))

    world = setup_world(args)
    state = Observable(sugarscape_snapshot(world))
    running = Observable(false)
    figure, reset_button, step_button, run_button, speed = _build_interactive_figure(
        state,
        running,
        args.params,
        args.steps;
        size = size,
        citizen_size = citizen_size,
        framerate = framerate,
    )
    visualization = SugarscapeVisualization(
        figure,
        args,
        world,
        state,
        running,
        Observable(1.0 / framerate),
        nothing,
    )

    on(reset_button.clicks) do _
        reset!(visualization)
    end
    on(step_button.clicks) do _
        step!(visualization)
    end
    on(run_button.clicks) do _
        visualization.running[] ? stop!(visualization) : play!(visualization)
    end
    on(speed.value) do value
        visualization.frame_delay[] = 1.0 / value
    end
    return visualization
end

using EBM
using CairoMakie
using Statistics

include(joinpath(@__DIR__, "social_habit_common.jl"))
using .SocialHabitExperiments

const T = EBM.Traffic

env_int(name, default) = parse(Int, get(ENV, name, string(default)))

seed = env_int("TRAFFIC_ANIMATION_SEED", 20260901)
steps = env_int("TRAFFIC_ANIMATION_STEPS", 5_000)
capture_every = env_int("TRAFFIC_ANIMATION_EVERY", 25)
framerate = env_int("TRAFFIC_ANIMATION_FRAMERATE", 15)
# Historical diagnostic under current semantics; keep it separate from retained evidence.
output_dir = get(
    ENV,
    "TRAFFIC_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "plots", "historical_regenerated"),
)

config = ExperimentConfig(
    seeds = seed:seed,
    steps = steps,
    burn_in = min(steps - 1, steps ÷ 5),
)

function mixture_history(config, scenario)
    model = capability_model(config, scenario)
    args = T.ModelArgs(
        seed = first(config.seeds),
        params = SocialHabitExperiments.model_params(config),
        prediction_strategy = model,
        steps = config.steps,
    )
    return T.traffic_history(args; every = capture_every)
end

function convention_strength(snapshot)
    isempty(snapshot.cars) && return 0.0
    return abs(sum(
        T.relative_lane_sign(car.lane, car.direction) for car in snapshot.cars
    ) / length(snapshot.cars))
end

function capability_share(snapshot, capability_index)
    isempty(snapshot.cars) && return 0.0
    mask = UInt8(1) << (capability_index - 1)
    return count(car -> car.capabilities & mask != 0, snapshot.cars) /
           length(snapshot.cars)
end

function all_acquired_share(snapshot)
    isempty(snapshot.cars) && return 0.0
    acquired_mask = sum(UInt8(1) << (index - 1) for index in 4:6)
    return count(
        car -> car.capabilities & acquired_mask == acquired_mask,
        snapshot.cars,
    ) / length(snapshot.cars)
end

function history_metrics(history)
    steps = [snapshot.step for snapshot in history]
    convention = convention_strength.(history)
    speed = [mean(car.speed for car in snapshot.cars) for snapshot in history]
    replacement_rate = zeros(Float64, length(history))
    for index in 2:length(history)
        step_delta = steps[index] - steps[index - 1]
        replacement_delta = history[index].cumulative_replacements -
                            history[index - 1].cumulative_replacements
        replacement_rate[index] = replacement_delta /
                                  (step_delta * length(history[index].cars))
    end
    capabilities = hcat(
        ([capability_share(snapshot, index) for snapshot in history] for index in 4:6)...,
        all_acquired_share.(history),
    )
    return (; steps, convention, speed, replacement_rate, capabilities)
end

function add_trailing_series!(axis, steps, values, frame; color, label, linestyle = :solid)
    visible_steps = lift(index -> steps[1:index], frame)
    visible_values = lift(index -> values[1:index], frame)
    lines!(
        axis,
        visible_steps,
        visible_values;
        color = color,
        linewidth = 3,
        linestyle = linestyle,
        label = label,
    )
    scatter!(
        axis,
        lift(index -> [steps[index]], frame),
        lift(index -> [values[index]], frame);
        color = color,
        markersize = 12,
    )
    return axis
end

function add_phase_series!(axis, replacement_rate, convention, frame; color, label)
    lines!(
        axis,
        lift(index -> replacement_rate[1:index], frame),
        lift(index -> convention[1:index], frame);
        color = (color, 0.55),
        linewidth = 2,
        label = label,
    )
    scatter!(
        axis,
        lift(index -> [replacement_rate[index]], frame),
        lift(index -> [convention[index]], frame);
        color = color,
        markersize = 14,
    )
    return axis
end

mkpath(output_dir)
static_history = mixture_history(config, :mixture_static_replacement)
evolutionary_history = mixture_history(config, :mixture_evolutionary_replacement)
length(static_history) == length(evolutionary_history) ||
    error("mixture histories must contain the same number of frames")

static = history_metrics(static_history)
evolutionary = history_metrics(evolutionary_history)
static.steps == evolutionary.steps || error("mixture histories must use matched steps")

comparison_path = joinpath(output_dir, "social_habit_mixture_comparison.mp4")

static_color = :darkorange2
evolutionary_color = :firebrick2
capability_colors = (:seagreen3, :darkorange2, :dodgerblue3, :purple3)
capability_labels = ("Habit", "Convention", "Social habit", "All three")
frame = Observable(1)
figure = Figure(size = (1500, 850), fontsize = 15, backgroundcolor = :white)
status = lift(frame) do index
    static_snapshot = static_history[index]
    evolutionary_snapshot = evolutionary_history[index]
    "Mixed capabilities: static-entry versus evolutionary replacement — " *
    "step $(static.steps[index])\n" *
    "Static: convention $(round(static.convention[index]; digits = 3)), " *
    "rate $(round(static.replacement_rate[index]; digits = 3)), " *
    "speed $(round(static.speed[index]; digits = 2)), " *
    "cumulative replacements $(static_snapshot.cumulative_replacements)\n" *
    "Evolutionary: convention $(round(evolutionary.convention[index]; digits = 3)), " *
    "rate $(round(evolutionary.replacement_rate[index]; digits = 3)), " *
    "speed $(round(evolutionary.speed[index]; digits = 2)), " *
    "cumulative replacements $(evolutionary_snapshot.cumulative_replacements)"
end
Label(
    figure[1, 1:3],
    status;
    fontsize = 18,
    font = :bold,
    tellwidth = false,
)

coordination_axis = Axis(
    figure[2, 1];
    title = "Emergent lane convention",
    xlabel = "step",
    ylabel = "convention strength",
)
add_trailing_series!(
    coordination_axis,
    static.steps,
    static.convention,
    frame;
    color = static_color,
    label = "Static entry",
)
add_trailing_series!(
    coordination_axis,
    evolutionary.steps,
    evolutionary.convention,
    frame;
    color = evolutionary_color,
    label = "Evolutionary",
)
hlines!(coordination_axis, [config.convention_target]; color = (:black, 0.35), linestyle = :dash)
xlims!(coordination_axis, 0, steps)
ylims!(coordination_axis, 0, 1)
axislegend(coordination_axis; position = :rb, framevisible = false)

replacement_axis = Axis(
    figure[2, 2];
    title = "Collision-driven replacement pressure",
    xlabel = "step",
    ylabel = "replacements per car-step",
)
add_trailing_series!(
    replacement_axis,
    static.steps,
    static.replacement_rate,
    frame;
    color = static_color,
    label = "Static entry",
)
add_trailing_series!(
    replacement_axis,
    evolutionary.steps,
    evolutionary.replacement_rate,
    frame;
    color = evolutionary_color,
    label = "Evolutionary",
)
replacement_limit = 1.05 * max(
    maximum(static.replacement_rate),
    maximum(evolutionary.replacement_rate),
)
xlims!(replacement_axis, 0, steps)
ylims!(replacement_axis, 0, replacement_limit)

speed_axis = Axis(
    figure[2, 3];
    title = "Traffic throughput",
    xlabel = "step",
    ylabel = "mean chosen speed",
)
add_trailing_series!(
    speed_axis,
    static.steps,
    static.speed,
    frame;
    color = static_color,
    label = "Static entry",
)
add_trailing_series!(
    speed_axis,
    evolutionary.steps,
    evolutionary.speed,
    frame;
    color = evolutionary_color,
    label = "Evolutionary",
)
xlims!(speed_axis, 0, steps)
ylims!(speed_axis, 1, 3.05)

static_capability_axis = Axis(
    figure[3, 1];
    title = "Static-entry capability composition",
    xlabel = "step",
    ylabel = "population share",
)
evolutionary_capability_axis = Axis(
    figure[3, 2];
    title = "Evolutionary capability composition",
    xlabel = "step",
    ylabel = "population share",
)
for index in eachindex(capability_labels)
    add_trailing_series!(
        static_capability_axis,
        static.steps,
        static.capabilities[:, index],
        frame;
        color = capability_colors[index],
        label = capability_labels[index],
    )
    add_trailing_series!(
        evolutionary_capability_axis,
        evolutionary.steps,
        evolutionary.capabilities[:, index],
        frame;
        color = capability_colors[index],
        label = capability_labels[index],
    )
end
for axis in (static_capability_axis, evolutionary_capability_axis)
    xlims!(axis, 0, steps)
    ylims!(axis, 0, 1)
end
axislegend(static_capability_axis; position = :rb, framevisible = false)

phase_axis = Axis(
    figure[3, 3];
    title = "Coordination–safety trajectory",
    xlabel = "replacement pressure (lower is better)",
    ylabel = "convention strength (higher is better)",
)
add_phase_series!(
    phase_axis,
    static.replacement_rate,
    static.convention,
    frame;
    color = static_color,
    label = "Static entry",
)
add_phase_series!(
    phase_axis,
    evolutionary.replacement_rate,
    evolutionary.convention,
    frame;
    color = evolutionary_color,
    label = "Evolutionary",
)
xlims!(phase_axis, 0, replacement_limit)
ylims!(phase_axis, 0, 1)
axislegend(phase_axis; position = :rb, framevisible = false)

rowgap!(figure.layout, 14)
colgap!(figure.layout, 18)
record(
    figure,
    comparison_path,
    eachindex(static_history);
    framerate = framerate,
    px_per_unit = 1,
) do index
    frame[] = index
end

for (label, history) in (
        ("static entry", static_history),
        ("evolutionary", evolutionary_history),
    )
    final = last(history)
    acquired_shares = [capability_share(final, index) for index in 4:6]
    println(
        label,
        ": final convention=", round(convention_strength(final); digits = 3),
        ", replacements=", final.cumulative_replacements,
        ", habit/convention/social shares=", round.(acquired_shares; digits = 3),
    )
end
println("wrote analytical comparison animation: $comparison_path")

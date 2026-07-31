using Agents
using EBM
using Random
using Statistics

include(joinpath(@__DIR__, "activation_habit_common.jl"))
using .ActivationHabitExperiments

const T = EBM.Traffic
const S = T.SequentialModel

output_dir = get(ENV, "TRAFFIC_OUTPUT_DIR", joinpath(@__DIR__, "..", "plots"))
csv_path = get(
    ENV,
    "TRAFFIC_RESULTS",
    joinpath(output_dir, "activation_habit_runs.csv"),
)
rows = read_results(csv_path)
seeds = validate_results(rows)

# Verify the controlled two-car case independently of the population runs. The
# cars approach the same longitudinal cell from opposite directions. Equal
# relative-lane choices are safe and unequal choices collide, so neither driver
# has a dominant local action.
params = T.ModelParams(init_agents = 2, ring_y = 10, ϵ = 0.0)
model = S.init_model(params, T.Weights(); seed = 1)
previous_clockwise = (1, 1)
previous_counterclockwise = (1, 3)
actions = (false, true) # right, left relative to travel direction
collision_matrix = [
    S.pair_collides(
        previous_clockwise,
        S.action_position(previous_clockwise, S.Clockwise, action_a, model),
        previous_counterclockwise,
        S.action_position(previous_counterclockwise, S.Counterclockwise, action_b, model),
    )
        for action_a in actions, action_b in actions
]
collision_matrix == Bool[0 1; 1 0] ||
    error("the controlled encounter no longer has the expected anti-coordination structure")

function controlled_encounter(timing; aligned_habit)
    controlled_params = T.ModelParams(
        init_agents = 2,
        ring_y = 10,
        lookahead = 4,
        ϵ = 0.0,
    )
    step_function = timing isa S.SequentialActivation ?
                    S.sequential_step! : S.simultaneous_step!
    controlled_model = StandardABM(
        S.Car,
        GridSpace((2, controlled_params.ring_y));
        model_step! = step_function,
        properties = (
            params = controlled_params,
            timing = timing,
            diagnostics = S.InteractionDiagnostics(),
        ),
        rng = Random.Xoshiro(1),
    )
    weights = T.Weights(
        wₛ = 0.0,
        wₒ = 0.0,
        wₐ = 1.0,
        wₕ = aligned_habit ? 2.0 : 0.0,
    )
    sensitivities = S.Sensitivities(1.0, 1.0, 1.0, 1.0)
    habitus = aligned_habit ? 1.0 : 0.0
    for (position, direction) in (
        ((1, 1), S.Clockwise),
        ((1, 3), S.Counterclockwise),
    )
        add_agent!(
            position,
            controlled_model;
            lr = 0.0,
            sensitivities = sensitivities,
            habitus = habitus,
            weights = weights,
            direction = direction,
            age = 10,
        )
    end
    Agents.step!(controlled_model, 1)
    return only(controlled_model.diagnostics.failed_encounters)
end

controlled_outcomes = (
    sequential_no_habit = controlled_encounter(S.SequentialActivation(); aligned_habit = false),
    simultaneous_no_habit = controlled_encounter(S.SimultaneousActivation(); aligned_habit = false),
    simultaneous_aligned_habit = controlled_encounter(S.SimultaneousActivation(); aligned_habit = true),
)
controlled_outcomes == (
    sequential_no_habit = 0,
    simultaneous_no_habit = 1,
    simultaneous_aligned_habit = 0,
) || error("controlled activation/habit outcomes changed: $controlled_outcomes")

results = hypothesis_results(rows)
report_path = joinpath(output_dir, "activation_habit_verification.md")
mkpath(output_dir)

format_number(value) = isfinite(value) ? string(round(value; digits = 5)) : "n/a"
status(result) = is_supported(result) ? "supported" : "not supported"
function condition_mean(timing, habit, metric)
    values = [
        getproperty(row, metric)
            for row in rows
            if row.timing == timing && row.habit == habit && isfinite(getproperty(row, metric))
    ]
    return mean(values)
end

open(report_path, "w") do io
    println(io, "# Activation timing and habit: verification report")
    println(io)
    println(
        io,
        "This report evaluates ",
        length(seeds),
        " paired seeds (",
        minimum(seeds),
        "–",
        maximum(seeds),
        "). Each seed was run under all four timing × habit conditions for ",
        first(rows).steps,
        " ticks with ",
        first(rows).population,
        " cars on a 2×",
        first(rows).ring_y,
        " torus. Metrics exclude the first ",
        first(rows).burn_in,
        " ticks. Lookahead=",
        first(rows).lookahead,
        ", error rate=",
        first(rows).error_rate,
        ", and active habit weight=",
        first(rows).habit_weight,
        ".",
    )
    println(io)
    println(io, "## Controlled mechanism check")
    println(io)
    println(io, "For two opposing cars approaching the same cell, the collision matrix is:")
    println(io)
    println(io, "| focal \\ other | right | left |")
    println(io, "|---|---:|---:|")
    println(io, "| right | safe | collision |")
    println(io, "| left | collision | safe |")
    println(io)
    println(
        io,
        "Thus the encounter is an anti-coordination problem: both actions can be safe, ",
        "but neither is safe independently of the other driver's simultaneous choice. ",
        "This verifies indeterminacy, not logical impossibility.",
    )
    println(io)
    println(io, "A one-tick mechanism fixture then gives:")
    println(io)
    println(io, "| Timing | Habit state | Failed encounter |")
    println(io, "|---|---|---:|")
    println(io, "| Sequential | none | 0 |")
    println(io, "| Simultaneous | none | 1 |")
    println(io, "| Simultaneous | exogenously aligned | 0 |")
    println(io)
    println(
        io,
        "Sequential activation lets the second driver react to the first driver's committed move. ",
        "With simultaneous activation that ordering signal disappears. A sufficiently strong, shared ",
        "habit restores coordination in the fixture. Because the habit is deliberately aligned here, ",
        "this is a mechanism test—not evidence that alignment emerges by itself.",
    )
    println(io)
    println(io, "## Population hypotheses")
    println(io)
    println(io, "Condition means across paired replications:")
    println(io)
    println(io, "| Timing | Habit | Compatibility | Pre-coordination | Disposition alignment | Disposition predictability | Failure rate |")
    println(io, "|---|---|---:|---:|---:|---:|---:|")
    for timing in ("sequential", "simultaneous"), habit in (false, true)
        println(
            io,
            "| ", timing,
            " | ", habit,
            " | ", format_number(condition_mean(timing, habit, :compatibility_rate)),
            " | ", format_number(condition_mean(timing, habit, :precoordination_rate)),
            " | ", format_number(condition_mean(timing, habit, :disposition_alignment)),
            " | ", format_number(condition_mean(timing, habit, :disposition_predictability)),
            " | ", format_number(condition_mean(timing, habit, :failure_rate)),
            " |",
        )
    end
    println(io)
    println(io, "| ID | Falsifiable claim | Paired effect | 95% bootstrap CI | one-sided p | Result |")
    println(io, "|---|---|---:|---:|---:|---|")
    for result in results
        effect = result.effect
        interval = "[" * format_number(effect.lower) * ", " * format_number(effect.upper) * "]"
        println(
            io,
            "| ", result.id,
            " | ", result.claim,
            " | ", format_number(effect.estimate),
            " | ", interval,
            " | ", format_number(effect.pvalue),
            " | ", status(result),
            " |",
        )
    end
    println(io)
    println(io, "Effects are first condition minus second condition as encoded by each claim; H5 is the difference-in-differences.")
    println(io)
    println(io, "## Interpretation guardrails")
    println(io)
    h2 = only(result for result in results if result.id == "H2")
    h3 = only(result for result in results if result.id == "H3")
    if is_supported(h2)
        println(
            io,
            "Habit increased compatible joint choices conditional on cars meeting. This establishes ",
            "an encounter-level coordination gain, but does not by itself show that the choice was ",
            "settled in advance.",
        )
    else
        println(
            io,
            "Habit did not reliably increase compatible joint choices. On these settings, the central ",
            "coordination claim is not reproduced and should not be asserted.",
        )
    end
    println(
        io,
        is_supported(h3) ?
        "Pre-coordinated encounters—aligned prior dispositions followed by consistent actions—also increased, supporting the claim that habit resolves strategic uncertainty in advance." :
        "Pre-coordinated encounters did not increase reliably. The compatibility gain therefore does not, on its own, verify the proposed reduction of strategic uncertainty through prior habit.",
    )
    println(io)
    println(
        io,
        "The p-values are paired sign-randomization estimates and the intervals are paired bootstrap ",
        "intervals. They are exploratory (no multiple-testing correction) and should be accompanied by ",
        "sensitivity checks over density, error rate, lookahead, habit strength, and replacement policy.",
    )
end

for result in results
    effect = result.effect
    println(
        result.id,
        " ",
        status(result),
        ": effect=",
        format_number(effect.estimate),
        ", CI=[",
        format_number(effect.lower),
        ", ",
        format_number(effect.upper),
        "], p=",
        format_number(effect.pvalue),
    )
end
println("controlled anti-coordination matrix: verified")
println("controlled activation/habit outcomes: $controlled_outcomes")
println("wrote verification report: $report_path")

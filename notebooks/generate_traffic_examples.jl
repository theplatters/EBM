using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using EBM

const OUTPUT_DIR = normpath(joinpath(@__DIR__, "..", "plots", "examples"))
const EXAMPLES = [
    (slug = "per_entity_habitus", strategy = Traffic.PerEntityHabitusStrategy()),
    (slug = "mean_habitus", strategy = Traffic.MeanHabitusStrategy()),
    (slug = "naive", strategy = Traffic.NaiveStrategy()),
    (slug = "two_frame_naive", strategy = Traffic.TwoFrameNaiveStrategy()),
    (slug = "decision_aware", strategy = Traffic.DecisionAwareStrategy()),
]

function main()
    mkpath(OUTPUT_DIR)
    for example in EXAMPLES
        args = Traffic.ModelArgs(
            seed = 42,
            steps = 60,
            prediction_strategy = example.strategy,
        )
        history = Traffic.traffic_history(args; every = 2)
        prefix = joinpath(OUTPUT_DIR, "traffic_$(example.slug)_seed42")
        save("$(prefix)_final.png", Traffic.plot_traffic(last(history)))
        Traffic.record_traffic(history, "$(prefix).mp4"; framerate = 12)
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()

module ElFasol

using Ark
using CairoMakie
using Random

export AttendanceHistory,
    ConstantPredictor,
    LagPredictor,
    Logger,
    MeanPredictor,
    MirrorPredictor,
    ModelArgs,
    ModelParams,
    TrendPredictor,
    evaluate_predictor,
    generate_plot_suite,
    main,
    plot_attendance_dynamics,
    plot_coordination_diagnostics,
    plot_predictor_ecology,
    run_model,
    setup_world,
    step!

include("components/agents.jl")
include("components/predictors.jl")

include("core/parameters.jl")
include("analysis/logger.jl")
include("core/resources.jl")

include("systems/prediction.jl")
include("systems/selection.jl")
include("systems/attendance.jl")
include("systems/scoring.jl")

include("simulation/setup.jl")
include("simulation/step.jl")
include("analysis/runner.jl")
include("analysis/plotting.jl")

end # module ElFasol

module Sugarscape

using Ark
using CairoMakie
using Random

export Age,
    CitizenId,
    CitizenState,
    Female,
    ImmuneProfile,
    Infection,
    InitialEndowment,
    Logger,
    Male,
    MaximumAge,
    Metabolism,
    ModelArgs,
    ModelParams,
    MovementMode,
    Position,
    ProposedPosition,
    ShuffledSequentialMovement,
    Sugar,
    SugarLandscape,
    SynchronousMovement,
    Vision,
    canonical_landscape,
    citizen_snapshot,
    generate_plot_suite,
    gini_coefficient,
    main,
    plot_model_diagnostics,
    plot_sugarscape,
    rebuild_occupancy!,
    run_model,
    setup_world,
    step!

include("components/citizens.jl")

include("core/parameters.jl")
include("analysis/logger.jl")
include("core/resources.jl")

include("systems/growback.jl")
include("systems/movement.jl")
include("systems/disease.jl")
include("systems/lifecycle.jl")
include("systems/reproduction.jl")

include("simulation/setup.jl")
include("simulation/step.jl")
include("analysis/runner.jl")
include("analysis/plotting.jl")

end # module Sugarscape

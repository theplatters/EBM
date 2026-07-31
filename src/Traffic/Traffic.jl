module Traffic

export TrafficCarState,
    TrafficSnapshot,
    plot_traffic,
    plot_traffic_history,
    record_traffic,
    traffic_history,
    traffic_snapshot

using Ark
using Random
using Distributions
using LinearAlgebra
using StatsBase
using CairoMakie
using IterTools
using OpenCL

include("components/agents.jl")
include("components/spatial.jl")
include("components/traits.jl")
include("components/capabilities.jl")


include("core/parameters.jl")
include("simulation/setup.jl")
include("core/resources.jl")
include("core/world.jl")


include("systems/movement.jl")
include("systems/collision.jl")
include("systems/spawning.jl")
include("systems/behavior.jl")
include("systems/habitus.jl")
include("systems/capability_behavior.jl")
include("systems/capability_movement.jl")

include("simulation/step.jl")

include("analysis/logger.jl")

include("analysis/traffic_visualization.jl")

include("analysis/parameter_sweeps.jl")
include("analysis/plotting.jl")
include("analysis/regressions.jl")
include("analysis/runner.jl")

include("sequential_model/agents.jl")

function main(args)
    world = setup_world(args)
    for _ in 1:args.steps
        step!(world, args.prediction_strategy)
    end
    return Ark.get_resource(world, Logger)
end


end

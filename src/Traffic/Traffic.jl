module Traffic

using Ark
using Random
using Distributions
using LinearAlgebra
using StatsBase
using GLMakie
using IterTools
using OpenCL

include("components/agents.jl")
include("components/spatial.jl")
include("components/traits.jl")

include("core/parameters.jl")
include("simulation/setup.jl")
include("core/resources.jl")
include("core/world.jl")


include("systems/movement.jl")
include("systems/collision.jl")
include("systems/spawning.jl")
include("systems/behavior.jl")
include("systems/habitus.jl")

include("simulation/step.jl")

include("analysis/logger.jl")

include("analysis/parameter_sweeps.jl")
include("analysis/plotting.jl")
include("analysis/regressions.jl")
include("analysis/runner.jl")

function main(args)
    world = setup_world(args)
    for _ in 1:args.steps
        step!(world, args.prediction_strategy)
    end
    return Ark.get_resource(world, Logger)
end


end

module Traffic

using Ark
using Random
using Distributions
using LinearAlgebra
using StatsBase
using GLMakie


include("components/spatial.jl")
include("components/agents.jl")
include("components/traits.jl")

include("core/parameters.jl")
include("core/occupancy.jl")
include("core/resources.jl")
include("core/world.jl")

include("systems/movement.jl")
include("systems/collision.jl")
include("systems/spawning.jl")
include("systems/behavior.jl")
include("systems/habitus.jl")

include("simulation/step.jl")

include("analysis/logger.jl")
include("analysis/plotting.jl")

function main(args)
    world = setup_world(args)
    for _ in 1:get!(args, :steps, 30)
        step!(world)
    end
    return Ark.get_resource(world, Logger)
end


end

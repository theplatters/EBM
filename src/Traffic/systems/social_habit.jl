const SUCCESS_TRACE_CUTOFF = 1.0e-8

"""Reduce every stored success trace by the configured retention factor."""
function decay_success_traces!(world)
    model = Ark.get_resource(world, CapabilityModel)
    traces = Ark.get_resource(world, SuccessfulDriverTrace).grid
    traces .*= model.social_trace_retention
    traces[abs.(traces) .< SUCCESS_TRACE_CUTOFF] .= 0.0
    return nothing
end

"""
Deposit lane-choice information along every path completed without collision.

This runs after failed entities have been removed and before their replacements
are spawned, so only successful drivers leave information on the torus.
"""
function deposit_success_traces!(world)
    model = Ark.get_resource(world, CapabilityModel)
    traces = Ark.get_resource(world, SuccessfulDriverTrace).grid
    for (entities, directions, speeds, paths) in Query(
            world, (Direction, Speed, MovementPath),
        )
        @inbounds for index in eachindex(entities)
            direction = directions[index]
            for microstep in 1:speeds[index].val
                position = paths[index].positions[microstep]
                signal = relative_lane_sign(position.x, direction)
                traces[position.x, position.y] = clamp(
                    traces[position.x, position.y] + model.social_trace_deposit * signal,
                    -1.0,
                    1.0,
                )
            end
        end
    end
    return nothing
end

function update_success_traces!(world)
    decay_success_traces!(world)
    deposit_success_traces!(world)
    return nothing
end

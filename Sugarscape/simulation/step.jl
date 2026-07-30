"""
Advance the Sugarscape by one period.

Resource growback precedes movement. Under `ShuffledSequentialMovement`, citizens move
and harvest in a seeded random order, so later citizens observe earlier moves and
harvests. Under `SynchronousMovement`, all proposals use the same initial state;
contested destinations are awarded randomly before movement is committed. Metabolism,
ageing, death, and optional replacement occur only after movement is complete.
"""
function step!(world)
    params = Ark.get_resource(world, ModelParams)
    events = Ark.get_resource(world, StepEvents)
    reset!(events)
    growback!(world)
    rebuild_occupancy!(world)

    if params.movement_mode == ShuffledSequentialMovement
        move_and_harvest_sequential!(world)
    else
        move_and_harvest_synchronously!(world)
    end

    disease!(world)
    lifecycle!(world)
    reproduce!(world)
    Ark.get_resource(world, SimulationClock).step += 1
    logger!(world)
    return nothing
end

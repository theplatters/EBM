function step!(world, strategy::OccupancyStrategy)


    t1 = Threads.@spawn calculate_lr!(world, strategy)
    t2 = Threads.@spawn store_prev_positions!(world)

    wait(t1); wait(t2)
    move!(world)

    new_entities = delete_on_collision!(world)
    spawn_new_entities!(world, new_entities)

    habitus_task = Threads.@spawn update_habitus!(world)
    occupancy_task = Threads.@spawn rebuild_occupancy!(world)

    wait(habitus_task)
    update_mean_habitus!(world)
    prediction_task = Threads.@spawn rebuild_predicted_occupancy!(world, strategy)

    wait(prediction_task); wait(occupancy_task)
    logger!(world)

    return nothing
end

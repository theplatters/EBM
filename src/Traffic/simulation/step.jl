function step!(world, strategy::OccupancyStrategy)


    t1 = Threads.@spawn calculate_lr!(world)
    t2 = Threads.@spawn store_prev_positions!(world)

    wait(t1); wait(t2)
    move!(world)

    new_entities = delete_on_collision!(world)
    spawn_new_entities!(world, new_entities)

    t3 = Threads.@spawn update_habitus!(world)
    t4 = Threads.@spawn rebuild_predicted_occupancy!(world, strategy)
    t5 = Threads.@spawn rebuild_occupancy!(world)

    wait(t3)
    update_mean_habitus!(world)

    wait(t4); wait(t5)
    logger!(world)

    return nothing
end

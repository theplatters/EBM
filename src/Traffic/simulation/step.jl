function step!(world)
    store_prev_positions!(world)
    move!(world)
    occ = Ark.get_resource(world, Occupancy)

    update_habitus!(world)

    rebuild_predicted_occupancy!(world)
    calculate_lr!(world)

    rebuild_predicted_occupancy!(world)
    calculate_lr!(world)

    new_entities = delete_on_collision!(world)
    spawn_new_entities!(world, new_entities)
    rebuild_occupancy!(world)

    logger!(world)
    return nothing
end

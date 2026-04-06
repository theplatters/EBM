@inline lane_to_LR(laneIndex::Int, dir::Direction) =
    dir == Clockwise ?
    (laneIndex == 1 ? 1.0 : -1.0) :
    (laneIndex == 2 ? 1.0 : -1.0)

function update_habitus!(world)
    params = Ark.get_resource(world, ModelParams)

    for (e, pos, dir, hab, step) in Query(world, (Position, Direction, Habitus, Step))
        @inbounds for i in eachindex(e)
            new_hab = Habitus(clamp(hab[i].val + lane_to_LR(pos[i].x, dir[i]) / (params.K + step[i].val), -1.0, 1.0))
            hab[i] = new_hab
        end
    end
    return
end

function update_mean_habitus!(world)
    habitus = Ark.get_resource(world, MeanHabitus)


    total_habitus = 0
    total_abs_habitus = 0
    total_entities = 0

    for (e, habitus) in Query(world, (Habitus,))
        total_abs_habitus = sum(abs.(habitus.val))
        total_habitus = sum(habitus.val)
        total_entities = length(e)
    end

    habitus.abs = total_abs_habitus / total_entities
    habitus.total = total_habitus / total_entities
    return
end

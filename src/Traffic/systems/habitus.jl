@inline relative_lane_sign(x::Int, dir::Direction) =
    dir == Clockwise ?
    (x == 1 ? 1.0 : -1.0) :
    (x == 2 ? 1.0 : -1.0)


function update_habitus!(world)
    params = Ark.get_resource(world, ModelParams)

    for (e, pos, dir, hab, step) in Query(world, (Position, Direction, Habitus, Step))
        @inbounds for i in eachindex(e)
            hab[i] = Habitus(
                clamp(
                    hab[i].val + relative_lane_sign(pos[i].x, dir[i]) / (params.K + step[i].val),
                    -1.0, 1.0
                )
            )
        end
    end

    return nothing
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

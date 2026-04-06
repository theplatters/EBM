function delete_on_collision!(world)
    curr_pos = Dict{Ark.Entity, Position}()
    prev_pos = Dict{Ark.Entity, Position}()
    dir_of = Dict{Ark.Entity, Direction}()

    curr_occ = Dict{Position, Vector{Ark.Entity}}()
    prev_occ = Dict{Position, Vector{Ark.Entity}}()

    for (e, pos, prev, dir) in Query(world, (Position, PrevPosition, Direction))
        @inbounds for i in eachindex(e)
            ent = e[i]
            pcur = pos[i]
            pprev = Position(prev[i])

            curr_pos[ent] = pcur
            prev_pos[ent] = pprev
            dir_of[ent] = dir[i]

            push!(get!(curr_occ, pcur, Ark.Entity[]), ent)
            push!(get!(prev_occ, pprev, Ark.Entity[]), ent)
        end
    end

    kill = Set{Ark.Entity}()
    dirs = Direction[]

    # 1. Same final cell collisions
    for ents in values(curr_occ)
        if length(ents) > 1
            for ent in ents
                if ent ∉ kill
                    push!(kill, ent)
                    push!(dirs, dir_of[ent])
                end
            end
        end
    end

    ents = collect(keys(curr_pos))

    # 2. Pairwise swaps and 3. diagonal crossings
    @inbounds for i in 1:(length(ents) - 1)
        e1 = ents[i]
        p1_prev = prev_pos[e1]
        p1_cur = curr_pos[e1]

        for j in (i + 1):length(ents)
            e2 = ents[j]
            p2_prev = prev_pos[e2]
            p2_cur = curr_pos[e2]

            collide = false

            # Exact swap:
            # A: a -> b, B: b -> a
            if p1_cur == p2_prev && p2_cur == p1_prev
                collide = true
            end

            # Diagonal crossing in 2-lane ring:
            # A: (1,y0)->(2,y1), B:(2,y0)->(1,y1)
            # More generally:
            # same start row, same end row, different start lanes, different end lanes,
            # and each goes to the other's end lane.
            if !collide
                if p1_prev.y == p2_prev.y &&
                        p1_cur.y == p2_cur.y  &&
                        p1_prev.x != p2_prev.x &&
                        p1_cur.x != p2_cur.x  &&
                        p1_prev.x == p2_cur.x &&
                        p2_prev.x == p1_cur.x
                    collide = true
                end
            end

            if collide
                if e1 ∉ kill
                    push!(kill, e1)
                    push!(dirs, dir_of[e1])
                end
                if e2 ∉ kill
                    push!(kill, e2)
                    push!(dirs, dir_of[e2])
                end
            end
        end
    end

    for e in kill
        Ark.remove_entity!(world, e)
    end

    return dirs
end

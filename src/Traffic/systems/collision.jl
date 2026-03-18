function delete_on_collision!(world)
    # Gather current and previous positions
    curr_pos = Dict{Ark.Entity, Position}()
    prev_pos = Dict{Ark.Entity, Position}()
    curr_occ = Dict{Position, Tuple{Ark.Entity, Direction}}()
    prev_occ = Dict{Position, Tuple{Ark.Entity, Direction}}()

    for (e, pos, prev, dir) in Query(world, (Position, PrevPosition, Direction))
        @inbounds for i in eachindex(e)
            ent = e[i]
            pcur = pos[i]
            pprev = Position(prev[i])
            d = dir[i]

            curr_pos[ent] = pcur
            prev_pos[ent] = pprev
            curr_occ[pcur] = (ent, d)
            prev_occ[pprev] = (ent, d)
        end
    end

    kill = Set{Ark.Entity}()
    dirs = Vector{Direction}()

    seen = Dict{Position, Tuple{Entity, Direction}}()
    for (pos, (ent, dir)) in curr_occ
        #Same final cell
        if haskey(seen, pos)
            if ent ∉ kill
                push!(kill, ent)
                push!(dirs, dir)
            end
            if first(seen[pos]) ∉ kill
                push!(kill, first(seen[pos]))
                push!(dirs, last(seen[pos]))
            end
        else
            seen[pos] = (ent, dir)
        end

        if haskey(prev_occ, pos)
            (other_ent, other_dir) = prev_occ[pos]
            if other_ent != ent && curr_pos[other_ent] == prev_pos[ent]
                if ent ∉ kill
                    push!(kill, ent)
                    push!(dirs, dir)
                end
                if other_ent ∉ kill
                    push!(kill, other_ent)
                    push!(dirs, other_dir)
                end
            end
        end
    end


    for e in kill
        Ark.remove_entity!(world, e)
    end
    return dirs
end

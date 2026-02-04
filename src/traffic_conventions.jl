import Ark
using Plots
using Ark: Entity, Query, unpack
using LinearAlgebra: diagind
using StatsBase
using Distributions
using Random


struct Ring
    width::UInt64
    height::UInt64
end

@enum Direction begin
    Clockwise = 1
    Counterclockwise = -1
end

struct Position
    x::Int64
    y::Int64
end

struct PrevPosition
    x::Int64
    y::Int64
end

PrevPosition(p::Position) = PrevPosition(p.x, p.y)
Position(p::PrevPosition) = Position(p.x, p.y)

struct SSensitvity
    val::Float64
end

struct OSensitvity
    val::Float64
end

struct Avoidance
    val::Float64
end

struct Habitgene
    val::Float64
end

struct Habitus
    val::Float64
end

struct LR
    val::Float64
end

mutable struct Step
    val::Float64
end


Base.@kwdef struct Weights
    wₛ::Float64 = 2.1
    wₒ::Float64 = 3.1
    wₐ::Float64 = 1.1
    wₕ::Float64 = 0.1
end

Base.@kwdef struct ModelParams
    δ::Float64 = 0.2
    init_agents::Int64 = 40
    K::Int64 = 3
end


function step(pos::Position, ring::Ring, direction::Direction)
    h = Int(ring.height)
    return mod1(Int(pos.y) + Int(direction), h)
end


Position(t::Tuple{UInt64, UInt64}) = Position(first(t), last(t))

function random_unique_positions(r::Ring, n::Int; rng = Random.default_rng())
    w = r.width
    h = r.height
    total = w * h
    n <= total || throw(ArgumentError("n=$n exceeds ring capacity=$total"))
    idxs = randperm(rng, total)[1:n]
    return Position.([(mod1(i, w), (i - 1) ÷ w + 1) for i in idxs])
end

@inline function spawn_car!(world, pos, dir, ssens, osense, avoid, habitgene, habitus)
    return Ark.new_entity!(
        world, (
            pos,
            PrevPosition(pos),
            dir,
            ssens,
            osense,
            avoid,
            habitgene,
            habitus,
            LR(0.0),
        )
    )
end

function spawn_init_entities!(world)
    ring = Ark.get_resource(world, Ring)
    rng = Ark.get_resource(world, TaskLocalRNG)
    params = Ark.get_resource(world, ModelParams)

    amount = params.init_agents

    positions = random_unique_positions(ring, amount, rng = rng)
    directions = shuffle(rng, repeat([Clockwise, Counterclockwise], amount ÷ 2))
    draws = rand(Random.default_rng(), Normal(1.0, params.δ), amount, 4)

    return @inbounds for i in 1:amount
        spawn_car!(
            world,
            positions[i],
            directions[i],
            SSensitvity(draws[i, 1]),
            OSensitvity(draws[i, 2]),
            Avoidance(draws[i, 3]),
            Habitgene(draws[i, 4]),
            Habitus(0.0),
        )
    end
end


function move!(world)
    ring = Ark.get_resource(world, Ring)

    for (e, pos, lr) in Query(world, (Position, LR))
        @inbounds for i in eachindex(e)
            pos[i] = Position(lr[i].val ≥ 0.0 ? 1 : 2, pos[i].y)
        end
    end
    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            pos[i] = Position(pos[i].x, step(pos[i], ring, dir[i]))
        end
    end
    return
end


function delete_on_collision!(world)
    # Gather current and previous positions
    pos_map = Dict{Ark.Entity, Position}()
    prev_map = Dict{Ark.Entity, Position}()

    for (e, pos, prev) in Query(world, (Position, PrevPosition))
        @inbounds for i in eachindex(e)
            pos_map[e[i]] = pos[i]
            prev_map[e[i]] = Position(prev[i])
        end
    end

    kill = Set{Ark.Entity}()

    # 1) Node collisions (same final cell)
    seen = Dict{Position, Ark.Entity}()
    for (ent, p) in pos_map
        if haskey(seen, p)
            push!(kill, ent)
            push!(kill, seen[p])
        else
            seen[p] = ent
        end
    end

    # 2) Edge collisions (swap positions in same tick)
    # Build lookup from prev position -> entity (assumes unique occupancy per cell)
    prev_occ = Dict{Position, Ark.Entity}()
    for (ent, pprev) in prev_map
        prev_occ[pprev] = ent
    end

    for (a, a_prev) in prev_map
        a_now = pos_map[a]
        b = get(prev_occ, a_now, nothing)   # who used to be where a is now?
        b === nothing && continue
        b == a && continue

        # swapped if b moved to a_prev
        if pos_map[b] == a_prev
            push!(kill, a)
            push!(kill, b)
        end
    end

    for ent in kill
        Ark.remove_entity!(world, ent)
    end
    return length(kill)
end

function spawn_new_entities!(world, amount)
    rng = Ark.get_resource(world, TaskLocalRNG)
    params = Ark.get_resource(world, ModelParams)
    ring = Ark.get_resource(world, Ring)
    unoccupied_positions = Set{Position}(Position(x, y) for x in 1:ring.width, y in 1:ring.height)
    for (e, pos) in Query(world, (Position,))
        @inbounds for i in eachindex(e)
            delete!(unoccupied_positions, pos[i])
        end
    end

    directions = shuffle(rng, repeat([Clockwise, Counterclockwise], amount))
    draws = rand(rng, Normal(2.0, params.δ), amount, 4)

    for i in 1:amount
        position = rand(unoccupied_positions)

        spawn_car!(
            world,
            position,
            directions[i],
            SSensitvity(draws[i, 1]),
            OSensitvity(draws[i, 2]),
            Avoidance(draws[i, 3]),
            Habitgene(draws[i, 4]),
            Habitus(0.0),

        )

        delete!(unoccupied_positions, position)
    end

    return
end

@inline transform_to_unit_range(x) = -2 * x + 3

function update_habitus!(world)
    step = Ark.get_resource(world, Step)
    params = Ark.get_resource(world, ModelParams)

    for (e, pos, hab) in Query(world, (Position, Habitus))
        @inbounds for i in eachindex(e)
            hab[i] = Habitus(hab[i].val + transform_to_unit_range(pos[i].x) / (params.K + step.val))
        end
    end
    return
end


@inline ahead_y(y::Int, dir::Direction, d::Int, h::Int) =
    mod1(y + d * Int(dir), h)
@inline is_left_relative(x::Int, dir::Direction) =
    (dir == Clockwise) ? (x == 1) : (x == 2)

function compute_observations_for!(occ, pos_n::Position, dir_n::Direction, ring::Ring)
    h = Int(ring.height)

    same_total = 0
    same_left = 0
    opp_total = 0
    opp_left = 0

    CL = 0
    CR = 0

    for d in 1:10
        y = ahead_y(pos_n.y, dir_n, d, h)

        # check both lanes at that zone
        for x in 1:2
            key = (x, y)
            v = get(occ, key, nothing)
            v === nothing && continue
            (_, dir_other) = v

            left_rel = is_left_relative(x, dir_n)

            if dir_other == dir_n
                same_total += 1
                same_left += left_rel
            else
                opp_total += 1
                opp_left += left_rel
            end

            # “very close” = exactly one zone ahead
            if d == 1 || d == 2
                if left_rel
                    CL = 1
                else
                    CR = 1
                end
            end
        end
    end

    SL = same_total == 0 ? 0.5 : same_left / same_total
    OL = opp_total == 0 ? 0.5 : opp_left / opp_total
    return SL, OL, CL, CR
end

function store_prev_positions!(world)
    for (e, pos, prev) in Query(world, (Position, PrevPosition))
        @inbounds for i in eachindex(e)
            prev[i] = PrevPosition(pos[i])
        end
    end
    return
end

function build_occupancy(world)
    occ = Dict{Tuple{Int, Int}, Tuple{Ark.Entity, Direction}}()
    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            occ[(pos[i].x, pos[i].y)] = (e[i], dir[i])
        end
    end
    return occ
end

function calculate_lr!(world)
    weights = Ark.get_resource(world, Weights)

    occ = build_occupancy(world)
    ring = Ark.get_resource(world, Ring)

    S_dic = Dict{Entity, Float64}()
    O_dic = Dict{Entity, Float64}()
    CL_dic = Dict{Entity, Float64}()
    CR_dic = Dict{Entity, Float64}()

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            SL, OL, CL, CR = compute_observations_for!(occ, pos[i], dir[i], ring)
            S_dic[e[i]] = SL
            O_dic[e[i]] = OL
            CL_dic[e[i]] = CL
            CR_dic[e[i]] = CR
        end
    end

    for (e, s, o, a, hg, ha, lr) in Query(world, (SSensitvity, OSensitvity, Avoidance, Habitgene, Habitus, LR))
        @inbounds for i in eachindex(e)
            s_val = weights.wₛ * s[i].val * (2 * S_dic[e[i]] - 1)
            o_val = weights.wₒ + o[i].val * (2 * O_dic[e[i]] - 1)
            avoidance_val = weights.wₐ * a[i].val * (CR_dic[e[i]] - CL_dic[e[i]])
            habit_strength = weights.wₕ * hg[i].val + ha[i].val
            lr[i] = LR(s_val + o_val + avoidance_val + habit_strength)
        end
    end

    return
end

function update!(world)
    step = Ark.get_resource(world, Step)
    step.val += 1
    @info step
    return
end

function main(args)
    world = Ark.World(Position, PrevPosition, Direction, SSensitvity, OSensitvity, Avoidance, Habitgene, Habitus, LR)

    Ark.add_resource!(world, Ring(2, 100))
    Ark.add_resource!(world, Random.default_rng())
    Ark.add_resource!(world, ModelParams())
    Ark.add_resource!(world, Weights())
    Ark.add_resource!(world, Step(1))

    spawn_init_entities!(world)
    for _ in 1:20
        update_habitus!(world)
        store_prev_positions!(world)
        move!(world)
        update!(world)
        calculate_lr!(world)

        new_entities = delete_on_collision!(world)
        @info new_entities
        spawn_new_entities!(world, new_entities)
    end
    return world
end
world = main(0)

world
for (e, lr) in Query(world, (Habitus,))
    @inbounds for i in eachindex(e)
        @info lr[i]
    end
end

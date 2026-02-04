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

struct Step
    val::Int64
end


Base.@kwdef struct Weights
    wₛ::Float64 = 0.5
    wₒ::Float64 = 0.5
    wₐ::Float64 = 0.3
    wₕ::Float64 = 0.2
end

Base.@kwdef struct ModelParams
    δ::Float64 = 0.1
    init_agents::Int64 = 40
    K::Float64 = 0.5
    lookahead::Int64 = 10
end


function step_y(pos::Position, ring::Ring, direction::Direction)
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
            Step(1),
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
    for (e, pos, dir, step) in Query(world, (Position, Direction, Step))
        @inbounds for i in eachindex(e)
            pos[i] = Position(pos[i].x, step_y(pos[i], ring, dir[i]))
            step[i] = Step(step[i].val + 1)
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
    draws = rand(rng, Normal(1.0, params.δ), amount, 4)

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


lane_to_LR(laneIndex::Int) = laneIndex == 1 ? 1.0 : -1.0

function update_habitus!(world)
    params = Ark.get_resource(world, ModelParams)

    for (e, pos, hab, step) in Query(world, (Position, Habitus, Step))
        @inbounds for i in eachindex(e)
            hab[i] = Habitus(clamp(hab[i].val + lane_to_LR(pos[i].x) / (params.K + step[i].val), -1.0, 1.0))
        end
    end
    return
end


@inline ahead_y(y::Int, dir::Direction, d::Int, h::Int) =
    mod1(y + d * Int(dir), h)
@inline is_left_relative(x::Int, dir::Direction) =
    (dir == Clockwise) ? (x == 1) : (x == 2)

function compute_observations_for!(world, occ, pos_n::Position, dir_n::Direction, ring::Ring)
    h = Int(ring.height)

    same_total = 0
    same_left = 0
    opp_total = 0
    opp_left = 0

    params = Ark.get_resource(world, ModelParams)

    CL = 0
    CR = 0

    for d in 1:params.lookahead
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
            if d == 1
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
            SL, OL, CL, CR = compute_observations_for!(world, occ, pos[i], dir[i], ring)
            S_dic[e[i]] = SL
            O_dic[e[i]] = OL
            CL_dic[e[i]] = CL
            CR_dic[e[i]] = CR
        end
    end

    for (e, s, o, a, hg, ha, lr) in Query(world, (SSensitvity, OSensitvity, Avoidance, Habitgene, Habitus, LR))
        @inbounds for i in eachindex(e)
            s_val = weights.wₛ * s[i].val * (2 * S_dic[e[i]] - 1)
            o_val = weights.wₒ * o[i].val * (2 * O_dic[e[i]] - 1)
            avoidance_val = weights.wₐ * a[i].val * (CR_dic[e[i]] - CL_dic[e[i]])
            habit_strength = weights.wₕ * hg[i].val + ha[i].val
            lr[i] = LR(s_val + o_val + avoidance_val + habit_strength)
        end
    end

    return
end


function main(args)
    world = Ark.World(Position, PrevPosition, Direction, SSensitvity, OSensitvity, Avoidance, Habitgene, Habitus, LR, Step)

    Ark.add_resource!(world, Ring(2, 200))
    Ark.add_resource!(world, Random.default_rng())
    Ark.add_resource!(world, ModelParams())
    Ark.add_resource!(world, Weights())

    spawn_init_entities!(world)
    for _ in 1:args

        calculate_lr!(world)
        store_prev_positions!(world)
        move!(world)
        update_habitus!(world)

        new_entities = delete_on_collision!(world)
        spawn_new_entities!(world, new_entities)
    end
    return world
end

# -------------------- Descriptive statistics after main --------------------

function convention_stats(world)
    ring = Ark.get_resource(world, Ring)

    # counts
    cw = 0; ccw = 0
    cw_left = 0; ccw_left = 0
    lane1 = 0; lane2 = 0

    # collect habitus for summary
    hab_vals = Float64[]

    for (e, pos, dir, hab) in Query(world, (Position, Direction, Habitus))
        @inbounds for i in eachindex(e)
            x = pos[i].x
            d = dir[i]

            lane1 += (x == 1)
            lane2 += (x == 2)

            if d == Clockwise
                cw += 1
                cw_left += is_left_relative(x, d)  # left relative to movement
            else
                ccw += 1
                ccw_left += is_left_relative(x, d)
            end

            push!(hab_vals, hab[i].val)
        end
    end

    pL_cw = cw == 0 ? NaN : cw_left / cw
    pL_ccw = ccw == 0 ? NaN : ccw_left / ccw

    # convention strength in [0,1]; 0 = no convention (50/50), 1 = perfect (all on same relative side)
    CS = (isnan(pL_cw) || isnan(pL_ccw)) ? NaN :
        (abs(pL_cw - 0.5) + abs(pL_ccw - 0.5))

    # agreement in [0,1]; 1 = both directions choose same relative side frequency
    A = (isnan(pL_cw) || isnan(pL_ccw)) ? NaN :
        (1 - abs(pL_cw - pL_ccw))

    hab_summary = isempty(hab_vals) ? nothing : (
            mean = mean(hab_vals),
            sd = std(hab_vals),
            q25 = quantile(hab_vals, 0.25),
            median = quantile(hab_vals, 0.5),
            q75 = quantile(hab_vals, 0.75),
        )

    return (
        n_agents = cw + ccw,
        cw = cw, ccw = ccw,
        lane1 = lane1, lane2 = lane2,
        p_left_rel_cw = pL_cw,
        p_left_rel_ccw = pL_ccw,
        convention_strength = CS,
        agreement = A,
        habitus = hab_summary,
    )
end

"""
Run simulation and print convention diagnostics.
Usage (REPL):
    world, stats = run_with_stats(10_000)
"""
function run_with_stats(steps::Int)
    world = main(steps)
    stats = convention_stats(world)

    println("\n--- Convention stats (final state) ---")
    println("agents:  $(stats.n_agents)   cw=$(stats.cw)  ccw=$(stats.ccw)")
    println("lane use (absolute): lane1=$(stats.lane1) lane2=$(stats.lane2)")
    println("p(left-relative | cw):   $(stats.p_left_rel_cw)")
    println("p(left-relative | ccw):  $(stats.p_left_rel_ccw)")
    println("convention strength (0..1): $(stats.convention_strength)")
    println("agreement (0..1):          $(stats.agreement)")
    if stats.habitus !== nothing
        h = stats.habitus
        println("habitus: mean=$(h.mean) sd=$(h.sd) q25=$(h.q25) median=$(h.median) q75=$(h.q75)")
    end

    return world, stats
end


# -------------------- Agreement over time + plot --------------------

"Compute agreement A(t) = 1 - |pL_cw(t) - pL_ccw(t)| in [0,1]."
function agreement_now(world)::Float64
    cw = 0; ccw = 0
    cw_left = 0; ccw_left = 0

    for (e, pos, dir) in Query(world, (Position, Direction))
        @inbounds for i in eachindex(e)
            x = pos[i].x
            d = dir[i]
            if d == Clockwise
                cw += 1
                cw_left += is_left_relative(x, d)
            else
                ccw += 1
                ccw_left += is_left_relative(x, d)
            end
        end
    end

    if cw == 0 || ccw == 0
        return NaN
    end
    pL_cw = cw_left / cw
    pL_ccw = ccw_left / ccw
    return 1 - abs(pL_cw - pL_ccw)
end

"""
Run the simulation, log agreement each step, and plot it at the end.
Returns: world, agreement_series
"""
function run_log_agreement(steps::Int)
    world = Ark.World(Position, PrevPosition, Direction, SSensitvity, OSensitvity, Avoidance, Habitgene, Habitus, LR, Step)

    Ark.add_resource!(world, Ring(2, 400))
    Ark.add_resource!(world, Random.default_rng())
    Ark.add_resource!(world, ModelParams())
    Ark.add_resource!(world, Weights())

    spawn_init_entities!(world)

    agreement_series = Vector{Float64}(undef, steps)

    for t in 1:steps
        update_habitus!(world)
        store_prev_positions!(world)
        move!(world)
        calculate_lr!(world)

        new_entities = delete_on_collision!(world)
        spawn_new_entities!(world, new_entities)

        agreement_series[t] = agreement_now(world)
    end

    p = plot(
        1:steps, agreement_series,
        xlabel = "Step",
        ylabel = "Agreement (0..1)",
        title = "Development of agreement over time",
        legend = false,
        ylim = (0, 1),
        lw = 2,
    )


    return world, agreement_series, p
end

# Example run:
world, A, p = run_log_agreement(550)
unicodeplots()
p

run_with_stats(1000)

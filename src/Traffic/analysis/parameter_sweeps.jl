using ProgressMeter

using Agents

struct SweepResult
    weights::Weights
    logger::MeanLogger
end

function simplex_grid(resolution)
    n = resolution - 1
    step = 1.0 / n
    return [
        (i * step, j * step, k * step, (n - i - j - k) * step)
            for i in 0:n
            for j in 0:(n - i)
            for k in 0:(n - i - j)
    ]
end

function sweep_weights(; resolution = 5, depth = 20, strategy::T, seed::Integer = 42) where {T <: OccupancyStrategy}

    # Generate all 4-tuples from the simplex (sum == 1)
    combos = simplex_grid(resolution)
    replicate_seeds = rand(Random.Xoshiro(seed), Int64, depth)
    results = Vector{SweepResult}(undef, length(combos))
    lk = ReentrantLock()

    p = Progress(length(combos); showspeed = true)
    Threads.@threads for combo_index in eachindex(combos)
        ws, wo, wa, wh = combos[combo_index]
        weights = Weights(wₛ = ws, wₒ = wo, wₐ = wa, wₕ = wh)
        result = MeanLogger([
            main(ModelArgs(seed = replicate_seeds[i], prediction_strategy = strategy, weights = weights))
                for i in 1:depth
        ])
        results[combo_index] = SweepResult(weights, result)
        lock(lk) do
            next!(p)
        end
    end
    finish!(p)

    return results
end


struct ABM end

function sweep_weights(
        ::ABM;
        resolution = 5,
        depth = 20,
        seed::Integer = 42,
        steps::Integer = 100,
        params::ModelParams = ModelParams(),
    )
    # Generate all 4-tuples from the simplex (sum == 1)
    combos = simplex_grid(resolution)
    replicate_seeds = rand(Random.Xoshiro(seed), Int64, depth)
    results = Vector{Any}(undef, length(combos))
    lk = ReentrantLock()

    p = Progress(length(combos); showspeed = true)
    Threads.@threads for combo_index in eachindex(combos)
        ws, wo, wa, wh = combos[combo_index]
        weights = Weights(wₛ = ws, wₒ = wo, wₐ = wa, wₕ = wh)
        m = [
            SequentialModel.init_model(params, weights; seed = replicate_seeds[i])
                for i in 1:depth
        ]

        result = Agents.ensemblerun!(m, steps, adata = [:age, :lr, :habitus])
        results[combo_index] = result
        lock(lk) do
            next!(p)
        end
    end
    finish!(p)

    return results
end
MeanLogger(sweep_res::Vector{SweepResult}) = MeanLogger(map(x -> x.logger, sweep_res))

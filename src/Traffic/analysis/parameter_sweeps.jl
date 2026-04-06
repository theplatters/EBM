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

function sweep_weights(; resolution = 5, depth = 20, strategy::T) where {T <: OccupancyStrategy}

    # Generate all 4-tuples from the simplex (sum == 1)
    combos = simplex_grid(resolution)
    results = SweepResult[]
    lk = ReentrantLock()

    p = Progress(length(combos); showspeed = true)
    Threads.@threads for (ws, wo, wa, wh) in combos
        weights = Weights(wₛ = ws, wₒ = wo, wₐ = wa, wₕ = wh)
        result = MeanLogger([main(ModelArgs(prediction_strategy = strategy, weights = weights)) for i in 1:depth])
        lock(lk) do
            push!(results, SweepResult(weights, result))

            next!(p)
        end
    end
    finish!(p)

    return results
end


struct ABM end

function sweep_weights(::ABM; resolution = 5, depth = 20)
    # Generate all 4-tuples from the simplex (sum == 1)
    combos = simplex_grid(resolution)
    results = Vector{Any}()
    lk = ReentrantLock()

    p = Progress(length(combos); showspeed = true)
    Threads.@threads for (ws, wo, wa, wh) in combos
        weights = Weights(wₛ = ws, wₒ = wo, wₐ = wa, wₕ = wh)
        m = [SequentialModel.init_model(ModelParams(), weights) for i in 1:depth]

        result = Agents.ensemblerun!(m, 100, adata = [:age, :lr, :habitus])
        lock(lk) do
            push!(results, result)
            next!(p)
        end
    end
    finish!(p)

    return results
end
MeanLogger(sweep_res::Vector{SweepResult}) = MeanLogger(map(x -> x.logger, sweep_res))

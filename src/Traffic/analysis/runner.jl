function run_all(;resolution=20, depth = 200)
    strategies = [
        PerEntityHabitusStrategy(),
        MeanHabitusStrategy(),
        RandomStrategy(),
        NaiveStrategy(),
        UnsureStrategy(),
        SwitchStrategy(),
    ]

    results = Vector{Vector{SweepResult}}(undef, length(strategies))

    Threads.@threads for i in eachindex(strategies)
        results[i] = sweep_weights(strategy = strategies[i], resolution = resolution, depth = depth)
    end

    return Dict(strategies[i] => results[i] for i in eachindex(strategies))
end

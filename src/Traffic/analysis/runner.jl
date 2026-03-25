function run_all()
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
        results[i] = sweep_weights(strategy = strategies[i], resolution = 10, depth = 100)
    end

    return Dict(strategies[i] => results[i] for i in eachindex(strategies))
end

using BenchmarkTools
using Dates
using EBM
using Printf

const SugarModel = EBM.Sugarscape
const BENCHMARK_RUNNER = Ref{Any}()
const BENCHMARK_ARGS = Ref{Any}()

run_benchmark_case() = BENCHMARK_RUNNER[](BENCHMARK_ARGS[])

function env_integer(name, default)
    value = parse(Int, get(ENV, name, string(default)))
    value > 0 || throw(ArgumentError("$name must be positive"))
    return value
end

function benchmark_runner(label, runner, args; samples)
    runner(args) # compile and warm the complete setup-and-run path
    BENCHMARK_RUNNER[] = runner
    BENCHMARK_ARGS[] = args
    GC.gc()
    benchmark = @benchmarkable run_benchmark_case()
    trial = run(benchmark; samples = samples, evals = 1, seconds = 120)
    estimate = median(trial)
    return (
        label = label,
        milliseconds = time(estimate) / 1.0e6,
        memory_mib = memory(estimate) / 2.0^20,
        allocations = allocs(estimate),
    )
end

function scenario_params(name, movement_mode, width, height, population)
    common = (
        width = width,
        height = height,
        population = population,
        movement_mode = movement_mode,
    )
    if name == :baseline
        return SugarModel.ModelParams(; common...)
    elseif name == :extended
        return SugarModel.ModelParams(
            ;
            common...,
            replace_dead = false,
            reproduction_enabled = true,
            reproduction_probability = 0.02,
            initial_infection_probability = 0.05,
        )
    end
    throw(ArgumentError("unknown scenario: $name"))
end

function benchmark_scenario(name; seed, steps, width, height, population, samples)
    results = NamedTuple[]
    pairs = (
        (
            "ECS sequential",
            SugarModel.run_model,
            "Agents.jl sequential",
            SugarModel.AgentSequential.run_model,
            SugarModel.ShuffledSequentialMovement,
        ),
        (
            "ECS synchronous",
            SugarModel.run_model,
            "Agents.jl synchronous",
            SugarModel.AgentSynchronous.run_model,
            SugarModel.SynchronousMovement,
        ),
    )
    for (ecs_label, ecs_runner, agent_label, agent_runner, movement_mode) in pairs
        params = scenario_params(name, movement_mode, width, height, population)
        args = SugarModel.ModelArgs(seed = seed, params = params, steps = steps)
        push!(results, benchmark_runner(ecs_label, ecs_runner, args; samples = samples))
        push!(results, benchmark_runner(agent_label, agent_runner, args; samples = samples))
    end
    return results
end

function print_results(name, results)
    println("\n### $(titlecase(string(name)))")
    println("\n| Implementation | Median time (ms) | Median memory (MiB) | Allocations |")
    println("|---|---:|---:|---:|")
    for result in results
        @printf(
            "| %s | %.3f | %.3f | %d |\n",
            result.label,
            result.milliseconds,
            result.memory_mib,
            result.allocations,
        )
    end
    for offset in (1, 3)
        ecs = results[offset]
        agent = results[offset + 1]
        ratio = ecs.milliseconds / agent.milliseconds
        @printf(
            "\n%s / %s time ratio: **%.3f×** (above 1 means Agents.jl was faster).\n",
            ecs.label,
            agent.label,
            ratio,
        )
    end
end

function main()
    seed = env_integer("SUGARSCAPE_BENCH_SEED", 20260813)
    steps = env_integer("SUGARSCAPE_BENCH_STEPS", 100)
    width = env_integer("SUGARSCAPE_BENCH_WIDTH", 50)
    height = env_integer("SUGARSCAPE_BENCH_HEIGHT", 50)
    population = env_integer("SUGARSCAPE_BENCH_POPULATION", 400)
    samples = env_integer("SUGARSCAPE_BENCH_SAMPLES", 20)

    println("# Sugarscape benchmark results")
    println("\n- Date: $(today())")
    println("- Julia: $(VERSION)")
    println("- BenchmarkTools: $(pkgversion(BenchmarkTools))")
    println("- CPU: $(first(Sys.cpu_info()).model) ($(Sys.CPU_NAME))")
    println("- Julia threads: $(Threads.nthreads())")
    println("- Seed: $seed")
    println("- Grid/population/steps: $(width)×$(height) / $population / $steps")
    println("- Samples: $samples (one complete setup plus run per sample; median reported)")

    for scenario in (:baseline, :extended)
        results = benchmark_scenario(
            scenario;
            seed = seed,
            steps = steps,
            width = width,
            height = height,
            population = population,
            samples = samples,
        )
        print_results(scenario, results)
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()

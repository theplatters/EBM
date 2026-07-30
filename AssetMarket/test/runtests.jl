using Ark
using EBM
using Test

const SFI = EBM.AssetMarket

function component_count(world, component_types)
    return sum(length(first(result)) for result in Ark.Query(world, component_types))
end

function total_holdings(world)
    total = 0.0
    for (entities, holdings) in Ark.Query(world, (SFI.AssetHolding,))
        @inbounds for i in eachindex(entities)
            total += holdings[i].val
        end
    end
    return total
end

@testset "Asset market equilibrium and setup" begin
    params = SFI.ModelParams(population = 8, predictors_per_trader = 12)
    equilibrium = SFI.homogeneous_equilibrium(params)
    @test equilibrium.price_slope ≈ 0.95 / 0.15
    @test equilibrium.payoff_variance ≈ 4.0 atol = 0.01
    @test equilibrium.forecast_a == params.dividend_persistence
    @test equilibrium.forecast_b ≈ 4.5 atol = 0.01

    world = SFI.setup_world(SFI.ModelArgs(seed = 10, params = params, steps = 0))
    @test component_count(world, (SFI.TraderId,)) == params.population
    @test component_count(world, (SFI.PredictorId,)) ==
          params.population * params.predictors_per_trader
    @test component_count(world, (SFI.DefaultPredictor,)) ==
          params.population * params.predictors_per_trader

    defaults = 0
    for (entities, markers) in Ark.Query(world, (SFI.DefaultPredictor,))
        defaults += count(marker -> marker.val, markers)
    end
    @test defaults == params.population

    selected = Int64[]
    for (_, predictor_ids) in Ark.Query(world, (SFI.SelectedPredictor,))
        append!(selected, predictor_id.val for predictor_id in predictor_ids)
    end
    @test length(selected) == params.population
    @test all(!=(0), selected)
end

@testset "Asset market clearing and timing" begin
    params = SFI.ModelParams(population = 10, predictors_per_trader = 20)
    world = SFI.setup_world(SFI.ModelArgs(seed = 22, params = params, steps = 0))
    history = Ark.get_resource(world, SFI.MarketHistory)
    initial_history_length = length(history.prices)

    SFI.step!(world)
    state = Ark.get_resource(world, SFI.MarketState)
    @test total_holdings(world) ≈ SFI.total_share_supply(params) atol = 1e-10
    @test state.price > 0.0
    @test state.volume >= 0.0
    @test length(history.prices) == initial_history_length + 1
    @test last(history.prices) == state.price
    @test last(history.dividends) == state.dividend
    @test Ark.get_resource(world, SFI.SimulationClock).step == 1

    matched = 0
    for (entities, flags) in Ark.Query(world, (SFI.Matched,))
        matched += count(flag -> flag.val, flags)
    end
    @test matched >= params.population
end

@testset "Asset market evolution and reproducibility" begin
    evolving_params = SFI.ModelParams(
        population = 6,
        predictors_per_trader = 10,
        evolution_interval = 1.0,
    )
    world = SFI.setup_world(
        SFI.ModelArgs(seed = 41, params = evolving_params, steps = 0),
    )
    SFI.step!(world)
    @test Ark.get_resource(world, SFI.EvolutionEvents).count == evolving_params.population

    args = SFI.ModelArgs(
        seed = 2026,
        params = SFI.ModelParams(population = 8, predictors_per_trader = 20),
        steps = 50,
    )
    first = SFI.main(args)
    second = SFI.main(args)
    @test first.prices == second.prices
    @test first.volumes == second.volumes
    @test first.technical_usage == second.technical_usage
    @test first.evolution_events == second.evolution_events
    @test all(isfinite, first.prices)
    @test all(>(0.0), first.prices)
end

@testset "Asset market visualizations" begin
    params = SFI.ModelParams(population = 6, predictors_per_trader = 10)
    logger = SFI.main(SFI.ModelArgs(seed = 7, params = params, steps = 20))
    @test !isnothing(SFI.plot_market_dynamics(logger; rolling_window = 5))
    @test !isnothing(
        SFI.plot_volatility_and_volume(
            logger;
            burn_in = 5,
            volatility_window = 5,
            maximum_lag = 5,
        ),
    )
    @test !isnothing(SFI.plot_belief_ecology(logger; rolling_window = 5))
    @test_throws ArgumentError SFI.plot_market_dynamics(SFI.Logger())
end

@testset "Asset market scenario analysis" begin
    scenarios = SFI.default_scenarios()
    @test length(scenarios) == 4
    @test unique(getproperty.(scenarios, :slug)) == getproperty.(scenarios, :slug)

    analysis = SFI.run_scenarios(
        scenarios[[1, 4]];
        steps = 20,
        burn_in = 5,
        replicates = 2,
        seed = 19,
    )
    @test length(analysis.runs) == 4
    @test length(analysis.summaries) == 2
    @test analysis.runs[1].logger.dividends == analysis.runs[3].logger.dividends
    @test all(
        isfinite(getproperty(run.metrics, metric))
            for run in analysis.runs for metric in SFI.SCENARIO_METRIC_NAMES
    )
    @test !isnothing(SFI.plot_scenario_price_paths(analysis))
    @test !isnothing(SFI.plot_scenario_metric_comparison(analysis))
    @test !isnothing(SFI.plot_scenario_tradeoff(analysis))

    mktempdir() do output_dir
        tables = SFI.write_scenario_tables(analysis, output_dir)
        @test all(isfile, tables)
    end
end

@testset "Asset market validation" begin
    @test_throws ArgumentError SFI.setup_world(
        SFI.ModelArgs(params = SFI.ModelParams(predictors_per_trader = 1)),
    )
    @test_throws ArgumentError SFI.setup_world(SFI.ModelArgs(steps = -1))
end

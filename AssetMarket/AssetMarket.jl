module AssetMarket

using Ark
using CairoMakie
using Random

export Logger,
    MarketHistory,
    MarketState,
    ModelArgs,
    ModelParams,
    ScenarioAnalysis,
    ScenarioMetrics,
    ScenarioSpec,
    calculate_scenario_metrics,
    complex_market_params,
    default_scenarios,
    fundamental_price,
    generate_plot_suite,
    generate_scenario_analysis,
    homogeneous_equilibrium,
    main,
    plot_belief_ecology,
    plot_market_dynamics,
    plot_scenario_metric_comparison,
    plot_scenario_price_paths,
    plot_scenario_tradeoff,
    plot_volatility_and_volume,
    run_model,
    run_scenarios,
    setup_world,
    slow_market_params,
    step!

include("components/traders.jl")
include("components/predictors.jl")

include("core/parameters.jl")
include("analysis/logger.jl")
include("core/resources.jl")

include("systems/dividends.jl")
include("systems/descriptors.jl")
include("systems/expectations.jl")
include("systems/clearing.jl")
include("systems/scoring.jl")
include("systems/evolution.jl")

include("simulation/setup.jl")
include("simulation/step.jl")
include("analysis/runner.jl")
include("analysis/plotting.jl")
include("analysis/scenarios.jl")
include("analysis/scenario_plotting.jl")

end # module AssetMarket

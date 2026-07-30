# Santa Fe artificial stock market in Ark.jl

This directory contains an Ark.jl reconstruction of Arthur, Holland, LeBaron,
Palmer, and Tayler's artificial stock market. Twenty-five heterogeneous traders
allocate wealth between a risky dividend-paying asset and a risk-free bond. Each
trader owns a population of condition/forecast rules that evolve asynchronously.

The implementation includes the paper's central mechanisms:

- CARA demand under conditionally Gaussian payoff forecasts;
- exact specialist clearing of a fixed stock supply;
- an AR(1) dividend process;
- twelve fundamental, technical, and control descriptors;
- wildcard condition strings and linear price-plus-dividend forecasts;
- exponentially weighted squared forecast error;
- asynchronous replacement of the worst twenty percent of rules through tournament
  selection, crossover, and mutation;
- slow and medium exploration parameterizations.

See `ECS_ADAPTATION.md` for the mapping to entities, components, resources, systems,
and the implementation choices needed to make the original model explicit in ECS.

## Running the model

```julia
using EBM

args = EBM.AssetMarket.ModelArgs(seed = 2026, steps = 2_500)
logger = EBM.AssetMarket.main(args)
logger.prices
```

`run_model(args)` returns the final Ark world. `main(args)` returns its aggregate
logger. Use `slow_market_params()` for the paper's slow-exploration regime and
`complex_market_params()` for the medium-exploration defaults.

## Plotting suite

```julia
result = EBM.AssetMarket.generate_plot_suite(
    EBM.AssetMarket.ModelArgs(seed = 2026, steps = 2_500);
    output_dir = "AssetMarket/plots",
    burn_in = 500,
)
```

This produces:

- `market_dynamics.png`: market and homogeneous-equilibrium prices, relative
  mispricing, and dividends;
- `volatility_and_volume.png`: returns, rolling volatility, volume, and raw versus
  absolute-return autocorrelation;
- `belief_ecology.png`: selected fundamental, technical, and control information,
  forecast/holding heterogeneity, and asynchronous evolution events.

Run `julia --project=. AssetMarket/generate_plots.jl` to reproduce the included
figures.

## Scenario analysis

`SCENARIOS.md` defines four matched-shock experiments: selection without genetic
replacement, the paper's slow-exploration regime, its medium-exploration regime, and a
rapid-exploration stress case. Run the ensemble analysis with:

```bash
julia --project=. AssetMarket/run_scenarios.jl
```

The command writes replicate and summary CSV files plus price-path, metric-comparison,
and adaptation–volatility figures to `AssetMarket/scenarios/`. The resulting economic
interpretation is recorded in `SCENARIO_ANALYSIS.md`.

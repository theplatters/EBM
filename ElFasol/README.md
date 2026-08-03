# El Farol bar model in Ark.jl

This directory contains a self-contained entity-component-system implementation of
Arthur's El Farol bar problem. `ElFasol.jl` is the module entry point. The spelling
`ElFasol` follows the repository directory requested for this implementation; the
economic model is the El Farol model.

Participants and predictor instances are separate entities. A participant is composed
from identity, forecast, decision, selected-predictor, and payoff components. Every
predictor entity has an owner, forecast, cumulative squared error, and deterministic
tie breaker. Predictor behavior is supplied by one of five heterogeneous components:
lag, mean, mirror, trend, or constant.

The systems implement a synchronous weekly schedule:

1. Every predictor forecasts from the public attendance history.
2. Every participant selects its lowest-error predictor.
3. Participants simultaneously decide whether to attend.
4. The model aggregates realized attendance and assigns payoffs.
5. All predictors receive a virtual squared-error update, selected or not.

The bar is enjoyable only when realized attendance is strictly below `capacity`.
Agents attend only when their forecast is strictly below `capacity`; equality is
therefore treated as crowding on both sides of the decision.

## Module structure

`ElFasol.jl` is included by `src/EBM.jl` and exposed as `EBM.ElFasol`:

```text
components/       participants and predictor-family components
core/             parameters, attendance history, RNG, and shared resources
systems/          forecasting, predictor selection, attendance, and scoring
simulation/       setup and synchronous weekly schedule
analysis/         logger, diagnostic plots, and model runner
test/runtests.jl  deterministic predictor, timing, reproducibility, and plot tests
plots/            checked-in standard figures
```

Predictor entities and participant entities are deliberately separate. Preserve
virtual scoring for every predictor, including predictors that were not selected
in a week, and preserve the synchronous decision boundary when changing the
schedule.

From the repository root:

```julia
using EBM

args = EBM.ElFasol.ModelArgs(seed = 2026, steps = 500)
logger = EBM.ElFasol.main(args)
logger.attendance
```

Use `run_model(args)` instead of `main(args)` when access to the final Ark world and
its components is needed.

## Plotting suite

The plotting suite records more than aggregate attendance. It tracks cross-agent
forecast dispersion, the share of active predictors belonging to each component
family, family-wide virtual forecast RMSE, and successful decisions. Generate the
standard figures with:

```julia
using EBM

args = EBM.ElFasol.ModelArgs(seed = 2026, steps = 750)
result = EBM.ElFasol.generate_plot_suite(
    args;
    output_dir = "ElFasol/plots",
    burn_in = 150,
    rolling_window = 25,
)
result.paths
```

The suite writes three figures:

- `attendance_dynamics.png` shows weekly attendance, convergence of its rolling mean,
  mean active expectations, and forecast dispersion relative to bar capacity.
- `coordination_diagnostics.png` shows forecast calibration, the attendance
  distribution, attendance autocorrelation, and the frequency of under-capacity and
  crowded regimes after burn-in.
- `predictor_ecology.png` shows the changing composition of selected predictor
  components and the virtual RMSE of every predictor family, including rules that were
  not selected.

Each plot is also available independently through `plot_attendance_dynamics`,
`plot_coordination_diagnostics`, and `plot_predictor_ecology`. Running
`julia --project=. ElFasol/generate_plots.jl` reproduces the checked-in figures.

## Testing

From the repository root:

```sh
julia --project=. ElFasol/test/runtests.jl
```

The complete package suite, including this model, is
`julia --project=. test/runtests.jl`.

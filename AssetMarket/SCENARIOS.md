# Asset-market scenarios

The scenarios isolate the speed and scope of adaptive belief formation while holding
market structure, preferences, asset supply, and the exogenous dividend process fixed.
Matched replicate seeds deliver the same dividend shocks to every scenario.

## 1. Selection only

Agents retain their initially endowed predictor populations. Forecast-error learning
and switching among existing rules continue, but genetic replacement is effectively
disabled by setting its mean interval far beyond the experiment horizon.

This is not a homogeneous rational-expectations equilibrium. It is a control for the
incremental effect of discovering new rules: heterogeneous initial beliefs may still
be selected and coordinated upon.

Expected diagnostic: information use and market behavior can change through rule
selection, but no genuinely new conditions or coefficients enter the population.

## 2. Slow exploration

This follows the paper's slow-learning regime:

- forecast-error update rate: `1/150`;
- mean genetic-replacement interval: 1,000 periods per trader;
- crossover probability: `0.30`.

Expected diagnostic: inaccurate rules lose influence gradually, genetic innovations
rarely coordinate across traders, and prices should remain comparatively close to the
homogeneous-expectations benchmark.

## 3. Medium exploration

This is the paper's complex-regime parameterization:

- forecast-error update rate: `1/75`;
- mean genetic-replacement interval: 250 periods per trader;
- crossover probability: `0.10`.

Expected diagnostic: new rules appear often enough for mutually reinforcing technical
beliefs to survive, potentially increasing mispricing, turnover, and persistent
volatility.

## 4. Rapid exploration

This is an explicit stress scenario rather than a calibration reported in the paper:

- forecast-error update rate: `1/30`;
- mean genetic-replacement interval: 100 periods per trader;
- condition mutation probability: `0.05`;
- coefficient mutation standard deviations: `0.075` and `1.5`.

Expected diagnostic: faster adaptation need not stabilize the market. More frequent
belief turnover may raise trading activity and forecast dispersion, although rapid
selection can also eliminate poor innovations before they coordinate.

## Comparison design

The standard analysis uses five matched-seed replicates of 1,500 periods and discards
the first 300 periods. It reports means and 95% normal confidence intervals for:

- signed and absolute price deviation from the homogeneous-equilibrium benchmark;
- return volatility and excess kurtosis;
- mean trading volume;
- lag-one raw- and absolute-return autocorrelation;
- selected technical and control descriptor usage;
- maximum drawdown.

The scenario analysis is comparative, not a claim of empirical calibration. Five
replicates are sufficient for a repository diagnostic but too few for precise tail-risk
inference. The CSV output retains replicate-level results so larger ensembles can be
run without changing the analysis code.

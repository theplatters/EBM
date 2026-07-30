# Scenario analysis: adaptive expectations in the asset market

## Experimental design

The analysis compares four belief-learning regimes over 1,500 periods, discards the
first 300 periods, and uses five matched replicate seeds. Within a replicate, every
scenario receives exactly the same AR(1) dividend innovations. Differences therefore
come from predictor selection and evolution rather than different fundamental shocks.

The reported interval is the ensemble mean plus or minus a 95% normal confidence
interval. With only five replicates, it should be read as a diagnostic of between-run
variation rather than precise tail inference. Full results are retained in
`scenarios/scenario_summary.csv` and `scenarios/scenario_runs.csv`.

## Main result: genetic exploration creates persistent overpricing

The strongest result is a monotonic increase in price deviation as genetic exploration
accelerates.

| Scenario | Mean signed mispricing | Mean absolute mispricing |
|---|---:|---:|
| Selection only | -0.43% ± 1.28 | 4.75% ± 0.86 |
| Slow exploration | 14.17% ± 0.61 | 14.34% ± 0.65 |
| Medium exploration | 20.44% ± 0.22 | 20.44% ± 0.22 |
| Rapid exploration | 21.76% ± 0.37 | 21.77% ± 0.37 |

Selection among a fixed heterogeneous rule population leaves the ensemble centered
near the homogeneous-expectations benchmark, although individual prices still deviate.
Once genetic replacement is enabled, prices develop a persistent positive premium.
The price-path figure shows that this is not produced by one isolated crash or bubble:
the ensemble paths move toward distinct regime-specific plateaus.

The slow regime in this implementation therefore does **not** converge as closely to
the rational-expectations benchmark as the original paper reports. That is a model
validation result, not something to hide. Plausible explanations include the much
shorter horizon, the operationalized coefficient-mutation distributions, protected
default rules, and use of the appendix's single best active predictor rather than an
`H`-rule combination.

## Trading activity responds more strongly than volatility

Mean trading volume rises sharply and monotonically:

| Scenario | Mean shares traded per period |
|---|---:|
| Selection only | 0.45 ± 0.06 |
| Slow exploration | 0.90 ± 0.16 |
| Medium exploration | 2.41 ± 0.52 |
| Rapid exploration | 4.71 ± 0.26 |

This is the clearest behavioral effect of faster adaptation. New and more frequently
revised rules create greater disagreement in desired holdings, even though total
holdings continue to clear exactly to the fixed stock supply.

Return volatility does not move monotonically with volume. It falls from 0.66% in the
selection-only scenario to 0.58% and 0.57% under slow and medium exploration, then rises
to 0.62% under rapid exploration. The scenario comparison therefore separates three
phenomena that could otherwise be conflated:

- genetic exploration increases the price level relative to fundamentals;
- genetic exploration increases reallocative trading strongly;
- short-horizon volatility responds nonlinearly and is not simply proportional to
  turnover or mispricing.

Maximum drawdown is also non-monotonic: approximately 36.1%, 29.9%, 30.4%, and 32.4%
from selection-only through rapid exploration. The fixed-rule control can have large
temporary losses even though its average price is close to fundamental value.

## Evidence for technical trading is suggestive, not decisive

Selected technical-condition use averages 5.31%, 4.15%, 5.60%, and 6.59% of available
technical bits across the four scenarios. Rapid exploration has the highest mean, but
the confidence intervals overlap substantially. More importantly, control-bit use is
also noisy—between 4.5% and 6.0% on average with wide intervals.

The correct inference is therefore limited: rapid exploration creates more scope for
technical rules to become active, but these experiments do not establish that technical
information is selected more strongly than uninformative controls. A convincing
emergence test requires more replicates, much longer runs, and a paired within-run
comparison of technical and control-bit incidence.

The adaptation–volatility plot reinforces this qualification. Replicates with more
technical-bit use are not ordered cleanly by volatility, while the marker sizes show a
much clearer scenario-level increase in absolute mispricing.

## Stylized facts not reproduced in this experiment

All scenarios exhibit very high first-order return autocorrelation (about 0.85–0.88)
and absolute-return autocorrelation (about 0.70–0.73). Excess kurtosis is close to zero
or slightly negative rather than strongly positive. These 1,500-period experiments
therefore do not reproduce the low linear return predictability and fat tails associated
with empirical financial returns and reported for the long complex-regime experiments
in the original paper.

This matters for how the current model should be used. It is already suitable as an ECS
case study of heterogeneous beliefs, exact simultaneous clearing, rule selection, and
asynchronous evolution. It should not yet be presented as a validated reproduction of
the Santa Fe market's empirical return signatures.

## Conclusions

Three findings are robust within this experiment:

1. Genetic rule discovery, rather than selection among a fixed initial rule set, is
   responsible for persistent positive mispricing.
2. Faster exploration produces a large and monotonic increase in trading volume.
3. More adaptation does not mechanically imply more return volatility or clearer
   selection of technical information.

The next validation stage should run the original 250,000-period horizon with at least
25 replicates, add a clamped homogeneous-expectations diagnostic, reproduce the genetic
operator details more exactly, and vary scoring speed, evolution frequency, and mutation
strength one at a time. Those experiments would distinguish finite-horizon behavior
from differences introduced by the ECS reconstruction's explicitly documented
operational choices.

# Accommodating the Santa Fe asset market to ECS

The original model is described in terms of traders who each contain one hundred
predictors and a specialist who aggregates their demands. The ECS implementation
keeps the economic timing and equations but changes where state and behavior live.

## Representation changes

### Traders contain current state, not behavior

A trader is an entity with focused components for identity, selected predictor,
expected payoff, perceived variance, desired holding, and settled holding. It has no
`step` method and does not own a mutable predictor array. Systems select precisely the
trader state needed for expectation assignment, clearing, or settlement.

### Predictors are entities

Each condition/forecast rule is a separate entity with owner, condition bits,
forecast coefficients, current forecast, exponentially weighted error variance,
demand variance, matching state, default-rule marker, and deterministic tie breaker.
The owner component provides the relationship to a trader without nesting mutable
objects. This makes predictor evaluation and scoring homogeneous population passes,
while genetic replacement changes the components of a predictor slot.

Stable predictor entities are deliberately reused during genetic replacement. The
scientific model replaces rules, for which identity has no economic meaning; replacing
their components avoids unnecessary archetype churn and keeps the hot predictor table
contiguous.

### The market is shared resource state

Price, dividend, volume, public histories, twelve descriptor bits, random-number streams,
the specialist's reduction buffers, predictor registry, and asynchronous evolution
schedule are resources. These quantities belong to the market rather than to any one
trader.

Dividend generation and predictor learning use separate RNG resources derived from the
same master seed. This permits matched-seed scenario experiments in which all regimes
receive identical exogenous dividend shocks even though their genetic algorithms
consume different numbers of random draws.

## Explicit system schedule

One period is decomposed as:

```text
reveal dividend
    -> clear price from previously submitted expectations
    -> compute desired holdings
    -> settle trades and volume
    -> score every previously matching predictor
    -> update public market descriptors
    -> evolve scheduled predictor populations
    -> match and evaluate predictors
    -> select each trader's next expectation
    -> log the market
```

The clearing system is a reduction over traders. Settlement is separate, so no
trader's new holding can affect another trader's demand in the same period. Predictor
matching/evaluation and forecast-error updates are bulk systems over predictor
components and are natural candidates for parallel execution; the clearing reduction
and genetic replacement remain synchronization boundaries.

## Operational choices and qualifications

- The appendix says only the most accurate matching predictor is used, while the main
  text also mentions combining the `H` most accurate rules. This implementation uses
  the appendix's single-rule specification.
- Every matching rule receives the exponentially weighted squared-error update. The
  selected rule's error variance is used for demand and is refreshed at that trader's
  genetic-algorithm event, matching the appendix's two-timescale description.
- Genetic events use a geometric waiting time with the requested mean interval. This
  makes asynchronous updating explicit and reproducible under the model RNG.
- The paper specifies that coefficient mutations add random values but does not report
  their distributions. The implementation exposes Gaussian mutation scales as
  `a_mutation_std` and `b_mutation_std` parameters.
- The fixed twelve descriptors are stored as `NTuple{12, Int8}` condition components:
  `-1` is wildcard, `0` is false, and `1` is true. This fixed-width representation is
  compact, immutable, and suitable for archetype iteration.
- A protected all-wildcard default predictor remains available to each trader, ensuring
  expectation formation is total for every possible market state.

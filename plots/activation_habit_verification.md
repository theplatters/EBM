# Activation timing and habit: verification report

This report evaluates 30 paired seeds (20260801–20260830). Each seed was run under all four timing × habit conditions for 5000 ticks with 120 cars on a 2×300 torus. Metrics exclude the first 1000 ticks. Lookahead=20, error rate=0.01, and active habit weight=0.5.

## Controlled mechanism check

For two opposing cars approaching the same cell, the collision matrix is:

| focal \ other | right | left |
|---|---:|---:|
| right | safe | collision |
| left | collision | safe |

Thus the encounter is an anti-coordination problem: both actions can be safe, but neither is safe independently of the other driver's simultaneous choice. This verifies indeterminacy, not logical impossibility.

A one-tick mechanism fixture then gives:

| Timing | Habit state | Failed encounter |
|---|---|---:|
| Sequential | none | 0 |
| Simultaneous | none | 1 |
| Simultaneous | exogenously aligned | 0 |

Sequential activation lets the second driver react to the first driver's committed move. With simultaneous activation that ordering signal disappears. A sufficiently strong, shared habit restores coordination in the fixture. Because the habit is deliberately aligned here, this is a mechanism test—not evidence that alignment emerges by itself.

## Population hypotheses

Condition means across paired replications:

| Timing | Habit | Compatibility | Pre-coordination | Disposition alignment | Disposition predictability | Failure rate |
|---|---|---:|---:|---:|---:|---:|
| sequential | false | 0.97758 | 0.75166 | 0.82212 | 0.84846 | 0.02242 |
| sequential | true | 0.98012 | 0.97329 | 0.99356 | 0.9862 | 0.01988 |
| simultaneous | false | 0.97446 | 0.90178 | 0.93996 | 0.94265 | 0.02554 |
| simultaneous | true | 0.9792 | 0.97395 | 0.99441 | 0.98692 | 0.0208 |

| ID | Falsifiable claim | Paired effect | 95% bootstrap CI | one-sided p | Result |
|---|---|---:|---:|---:|---|
| H1 | simultaneous timing reduces compatible joint choices without habit | -0.00312 | [-0.00351, -0.00272] | 5.0e-5 | supported |
| H2 | habit increases compatible joint choices under simultaneous timing | 0.00474 | [0.00433, 0.00512] | 5.0e-5 | supported |
| H3 | habit increases pre-coordinated encounters | 0.07216 | [0.05435, 0.09112] | 5.0e-5 | supported |
| H4 | habit improves compatibility more under simultaneous than sequential timing | 0.0022 | [0.0017, 0.00265] | 5.0e-5 | supported |

Effects are first condition minus second condition as encoded by each claim; H5 is the difference-in-differences.

## Interpretation guardrails

Habit increased compatible joint choices conditional on cars meeting. This establishes an encounter-level coordination gain, but does not by itself show that the choice was settled in advance.
Pre-coordinated encounters—aligned prior dispositions followed by consistent actions—also increased, supporting the claim that habit resolves strategic uncertainty in advance.

The p-values are paired sign-randomization estimates and the intervals are paired bootstrap intervals. They are exploratory (no multiple-testing correction) and should be accompanied by sensitivity checks over density, error rate, lookahead, habit strength, and replacement policy.
